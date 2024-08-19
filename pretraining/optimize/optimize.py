import argparse
import optuna
import numpy as np
import torch
import torch.optim as optim
import yaml
import wandb 
import ipdb
import os
from transformers import  ViTMAEForPreTraining, ViTMAEConfig
from common.scheduler import CosineAnnealingWithWarmUp
from pretraining.validate import validate
from pretraining.data_utils import get_default_train_transform, get_default_val_transform, get_dataloader


def load_config(config_path):
    with open(config_path, 'r') as file:
        train_config = yaml.safe_load(file)
        return train_config
        
def objective(trial, train_config):
    ### dataset
    train_dataset = train_config['dataset']['train']
    val_dataset = train_config['dataset']['val']
    #### wandb projectname
    group = train_config['group']
    wandb_project_name = train_config['wandb_project']
    ### hyperparams
    steps = train_config['eval_every_x_steps']
    mean = train_config['transforms']['normalize']['mean']
    std = train_config['transforms']['normalize']['std']
    num_epochs = train_config['num_epochs']
    learning_rate_min = float(train_config['learning_rate_range']['min'])
    learning_rate_max = float(train_config['learning_rate_range']['max'])
    optimizers = train_config['optimizers']
    batch_size_min = train_config['batch_size_range']['min']
    batch_size_max = train_config['batch_size_range']['max']
    warmup_fraction_min = float(train_config['warmup_fraction']['min'])
    warmup_fraction_max = float(train_config['warmup_fraction']['max'])
    weight_decay_min = float(train_config['weight_decay']['min'])
    weight_decay_max = float(train_config['weight_decay']['max'])
    warmup_epochs_min = train_config['warmup_epochs']['min']
    warmup_epochs_max = train_config['warmup_epochs']['max']
    momentum_min = train_config['momentum']['min']
    momentum_max = train_config['momentum']['max']
    beta2_min = train_config['beta2']['min']
    beta2_max = train_config['beta2']['max']
    batch_sizes = [2**i for i in range(int(np.log2(batch_size_min)),int(np.log2(batch_size_max))+1)]


    device = torch.device("cuda")
    learning_rate = trial.suggest_float('learning_rate', learning_rate_min, learning_rate_max, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', optimizers)
    momentum= trial.suggest_float('momentum', momentum_min, momentum_max) # momentum in sgd or beta l in adam(W)
    beta2 = trial.suggest_float('beta2', beta2_min, beta2_max)
    batch_size = trial.suggest_categorical('batch_size', batch_sizes)
    weight_decay = trial.suggest_float('weight_decay', weight_decay_min, weight_decay_max, log=True)
    warmup_fraction = trial.suggest_float("warmup_fraction", warmup_fraction_min, warmup_fraction_max, log=True)
    warmup_epochs = trial.suggest_float("warmup_epochs", warmup_epochs_min, warmup_epochs_max)
    # num of eppochs before lr stabilizes and keeps constant
    #total_scheduling_epochs = trial.suggest_float("total_scheduling_epochs", warmup_epochs, num_epochs)
    total_scheduling_epochs = num_epochs
    eta_min = trial.suggest_float("eta_min", 0, learning_rate)
    
    train_transform = get_default_train_transform(mean, std)
    val_transform = get_default_val_transform(mean, std)


    config = ViTMAEConfig(norm_pix_loss = True,  #corresponding vit-mae-large layers
                            hidden_size = 1024,
                            intermediate_size = 4096,
                            num_attention_heads = 16,
                            num_hidden_layers = 24,
                            num_channels = 1
                        )
       
    model = ViTMAEForPreTraining(config).to(device) # use the same parameters as mae-large

    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(momentum, beta2))
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(momentum, beta2))
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

    train_dataloader = get_dataloader(root=train_dataset,
                                       transform=train_transform,
                                       batch_size=batch_size)
    val_dataloader = get_dataloader(root=val_dataset,
                                        transform=val_transform,
                                        batch_size=batch_size)

    warmup_steps = warmup_epochs*len(train_dataloader)
    total_steps = total_scheduling_epochs*len(train_dataloader)
    scheduler = CosineAnnealingWithWarmUp(warmup_steps=warmup_steps,
                                            total_steps=total_steps,
                                            optimizer=optimizer,
                                            warmup_fraction=warmup_fraction,
                                            eta_min=eta_min)

    best_loss = float('inf')
    trial_config = dict(trial.params)
    trial_config["trial.number"] = trial.number
    wandb.init(
        project=wandb_project_name,
        entity="katja-sivertsen",
        config=trial_config,
        group=group,
        reinit=True,
    )   
    step_count = 0
    report_step = 0
    try:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for images, _ in train_dataloader:
                step_count += 1
                images = images.to(device)
                outputs = model(images)
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

                if (step_count*batch_size/64) % steps == 0:
                    val_loss = validate(model, val_dataloader, device)
                    normilized_step_count = int(step_count*batch_size/64)
                    print(f"Step (batch normalized) {normilized_step_count}, Valid Loss: {val_loss}")
                    report_step += 1
                    trial.report(val_loss, report_step)
                    wandb.log(data={"loss": val_loss}, step=report_step)
                    if trial.should_prune() or np.isnan(val_loss):
                        wandb.run.summary["state"] = "pruned"
                        wandb.finish(quiet=True)
                        raise optuna.exceptions.TrialPruned()
                    if val_loss < best_loss:
                        best_loss = val_loss
                    model.train()
            
        val_loss = validate(model, val_dataloader, device)
        wandb.log(data={"final loss": val_loss}, step=report_step+1)
        trial.report(val_loss, report_step+1)
        print(f"Final Valid Loss: {val_loss}")
        if val_loss < best_loss:
            best_loss = val_loss
        wandb.run.summary["final loss"] = val_loss
        wandb.run.summary["state"] = "complated"
        wandb.finish(quiet=True)
        return best_loss
    except RuntimeError as e:
        if 'out of memory' in str(e):
            torch.cuda.empty_cache()

            return float('inf')
        else:
            raise e  # Re-raise unexpected errors


def main(config_path):
    train_config = load_config(config_path)
    storage = train_config.get('storage', None)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    study = optuna.create_study(direction='minimize',
                                study_name=train_config['group'],
                                pruner=pruner,
                                storage = storage,
                                load_if_exists=True)
    study.optimize(lambda trial: objective(trial, train_config), n_trials=1000)
    print(study.best_params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help="Path to the configuration YAML file.")
    args = parser.parse_args()
    main(args.config_path)