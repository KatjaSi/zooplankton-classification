import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import csv
import time
import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import yaml
import numpy as np
import copy
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import WeightedRandomSampler
from train_utils import one_iter


def get_loaders(device, pretrained, input_size, mean, std):
    train_transform = transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(input_size),
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
       # transforms.Resize(256),
       # transforms.CenterCrop(input_size),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = 'ZooScan77_small'

    train_dataset = ImageFolder(root=f"datasets/{dataset}/train", transform=train_transform)
    val_dataset = ImageFolder(root=f"datasets/{dataset}/val", transform=val_transform)
    test_dataset  = ImageFolder(root=f"datasets/{dataset}/test", transform=val_transform)

    train_labels = np.array(train_dataset.targets)
    class_counts = np.bincount(train_labels)
    class_weights = 1. / class_counts
    # Apply weights to each sample
    weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(weights, len(weights))

    batch_size = 32*torch.cuda.device_count()#if pretrained else 64
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=torch.cuda.device_count())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,  num_workers=torch.cuda.device_count())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,  num_workers=torch.cuda.device_count())

    return train_loader, val_loader, test_loader

def main():
    results = []
    pretrained_results_file = 'preliminary_study/pretrained_vit_model_evaluation_results_attempt11.csv'
    if not os.path.exists(pretrained_results_file):
        with open(pretrained_results_file, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file,
            fieldnames=['model_name',
                        'pretrained', 
                        'val_loss', 
                        'val_accuracy', 
                        'val_balanced_accuracy',
                        'val_macro_avg_precision',
                        'val_macro_avg_f1_score',
                        'test_loss', 
                        'test_accuracy', 
                        'test_balanced_accuracy', 
                        'test_macro_avg_precision',
                        'test_macro_avg_f1_score', 
                        'best_epoch',
                        'training_time_seconds']) 
            dict_writer.writeheader()

    with open('preliminary_study/vit_models.yaml', 'r') as file:
        config = yaml.safe_load(file)
        pretrained = True
        device = "cuda"
        for model in config['models'][30:]:
            model_name = model['name']
            torch.cuda.empty_cache()
            print(f"Training {model_name}:")
            try:
                model = timm.create_model(model_name, pretrained=pretrained, num_classes=77)
            except RuntimeError as e:
                print(f"Skipping {model_name} with pretrained={pretrained} due to error: {e}")
                continue
            model = model.to(device)
            
            
            # Get default input size, mean, and std for the model
            input_size = model.default_cfg['input_size'][1]
            mean = model.default_cfg['mean']
            std = model.default_cfg['std']

            train_loader, val_loader, test_loader = get_loaders(device, pretrained, input_size, mean, std)
            criterion = nn.CrossEntropyLoss()

            lr = 0.00005 if pretrained else 0.0005 #0.001
            optimizer = optim.Adam(params=model.parameters(), lr=lr)

            num_epochs = 100 #20?
            best_epoch = 0
            best_val_loss = float('inf')
            best_val_balanced_accuracy = float('-inf')
            best_val_accuracy = float('inf')
            best_val_macro_avg_precision = float('-inf')
            best_val_macro_avg_f1_score = float('-inf')
            patience = 2
            patience_count = patience
            best_model_weights = None
            start_time = time.time()
            for epoch in range(num_epochs):
                result = one_iter(model, criterion, train_loader,
                            device,
                            train=True,
                            optimizer=optimizer,
                            monitoring_metrics=['accuracy', 'balanced_accuracy'])
                loss = result['loss']
                accuracy = result['accuracy']
                balanced_accuracy = result['balanced_accuracy']
                print(f"Epoch {epoch+1}, \
                    Train loss: {loss}, \
                    train accuracy: {accuracy:.6f},\
                    balanced train accuracy: {balanced_accuracy:.6f}")

                # validation
                monitoring_metrics=['accuracy', 'balanced_accuracy', 'macro_avg_precision', 'macro_avg_f1_score']
                result = one_iter(model, criterion, val_loader,
                                device,
                                train=False,
                                monitoring_metrics=monitoring_metrics)
                val_loss = result['loss']
                accuracy = result['accuracy']
                balanced_accuracy = result['balanced_accuracy']
                macro_avg_precision = result['macro_avg_precision']
                macro_avg_f1_score = result['macro_avg_f1_score']
                print(f"Epoch {epoch+1}, \
                    Valid loss: {val_loss}, \
                    valid accuracy: {accuracy:.6f},\
                    balanced valid accuracy: {balanced_accuracy:.6f}")

                if best_val_balanced_accuracy < balanced_accuracy:
                    best_val_balanced_accuracy = balanced_accuracy
                    best_val_loss = val_loss # probably not the best loss, but the one corresponding to the best early stopping metric
                    best_val_accuracy = accuracy
                    best_val_macro_avg_precision = macro_avg_precision
                    best_val_macro_avg_f1_score = macro_avg_f1_score
                    best_model_weights = copy.deepcopy(model.state_dict())
                    best_epoch = epoch+1
                    patience_count = patience # this was forgotten
                else:
                    patience_count -= 1
                    if patience_count == 0:
                        break

            print('Finished Training')
            time_spent = time.time() - start_time

            # test
            model.load_state_dict(best_model_weights)

            result = one_iter(model, criterion, test_loader,
                                    device,
                                    train=False,
                                    monitoring_metrics=monitoring_metrics)
            loss = result['loss']
            accuracy = result['accuracy']
            balanced_accuracy = result['balanced_accuracy']
            macro_avg_precision = result['macro_avg_precision']
            macro_avg_f1_score = result['macro_avg_f1_score']
            print(f"Test loss: {loss}, \
                    test accuracy: {accuracy:.6f},\
                    balanced test accuracy: {balanced_accuracy:.6f}")


            # Save results to the appropriate CSV file
            results = {
                'model_name': model_name,
                'pretrained': pretrained,
                'val_loss': f"{best_val_loss:.4f}",
                'val_accuracy': f"{best_val_accuracy:.4f}",
                'val_balanced_accuracy':f"{best_val_balanced_accuracy:.4f}",
                'val_macro_avg_precision': f"{best_val_macro_avg_precision:.4f}",
                'val_macro_avg_f1_score': f"{best_val_macro_avg_f1_score:.4f}",
                'test_loss': f"{loss:.4f}",
                'test_accuracy': f"{accuracy:.4f}",
                'test_balanced_accuracy': f"{balanced_accuracy:.4f}",
                'test_macro_avg_precision':f"{macro_avg_precision:.4f}",
                'test_macro_avg_f1_score': f"{macro_avg_f1_score:.4f}",
                'best_epoch': best_epoch,
                'training_time_seconds': f"{time_spent:.2f}"
            }
                        
            with open(pretrained_results_file, 'a', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=results.keys())
                dict_writer.writerow(results)

if __name__ == "__main__":
    main()