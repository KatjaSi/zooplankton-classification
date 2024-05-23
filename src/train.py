import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import shutil
import os
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import WeightedRandomSampler
from plots import plot_confusion_matrix
from parsers import TrainConfigParser # type: ignore
from transforms import apply_clahe, resize_and_pad
from datetime import datetime

import torch.optim as optim

import os
import pandas as pd
import copy


from train_utils import one_iter


def main():

    parser = TrainConfigParser()

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    batch_size = parser.get_batch_size()
    max_num_epochs = parser.get_max_num_epochs()
    dataset = parser.get_dataset_name()
    is_enable_report = parser.is_enable_report()
    report_frequency = parser.get_report_frequency()
    is_enable_confusion_matrix = parser.is_enable_confusion_matrix()
    is_enable_stats_per_class = parser.is_enable_stats_per_class()
    num_workers = parser.get_num_workers()
    is_checkpoint = parser.is_checkpoint()
    early_stopping_metric = parser.get_early_stopping_metric()
    compare_op = parser.get_compare_operator()

    # early stopping
    best_metric = float('-inf') if early_stopping_metric in ["accuracy", "balanced_accuracy"] else float('inf')
    best_model_weights = None
    patience = parser.get_patience()
    best_epoch = 0

    checkpoint_path = os.path.join('checkpoints', parser.get_model_name(), datetime.now().strftime('%Y-%m-%d-%H-%M'))

    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)
    if is_enable_report or is_checkpoint:
        os.makedirs(checkpoint_path)

    training_parameters_path = os.path.join(checkpoint_path, 'training_parameters')
    stats_df_path = os.path.join(checkpoint_path, 'stats_df.csv')
    best_model_path = os.path.join(checkpoint_path, "best_model.pth")
    report_path = os.path.join(checkpoint_path, "report.txt")

    if is_enable_stats_per_class:
        columns = ["Epoch", "Class ID", "Recall", "Precision", "F1_Score"]
        stats_df = pd.DataFrame(columns=columns)

    train_transform = transforms.Compose([
       # transforms.Resize(256),
       # transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(180),
        transforms.Lambda(lambda img: resize_and_pad(img)),  
        transforms.RandomAffine(180, (0.1, 0.1)), # rotation + translation
        transforms.ToTensor(),
        #transforms.Grayscale(num_output_channels=1),
        #transforms.Lambda(apply_clahe),
        #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Lambda(lambda img: resize_and_pad(img)),  
       # transforms.Resize(256),
       # transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Grayscale(num_output_channels=1),
       # transforms.Lambda(apply_clahe),
        #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = ImageFolder(root=f"datasets/{dataset}/train", transform=train_transform)
    val_dataset = ImageFolder(root=f"datasets/{dataset}/val", transform=val_transform)
    test_dataset  = ImageFolder(root=f"datasets/{dataset}/test", transform=val_transform)


    # weighted random sampler
    train_labels = np.array(train_dataset.targets)
    class_counts = np.bincount(train_labels)
    class_weights = 1. / class_counts
    # Apply weights to each sample
    weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = parser.get_model().to(device)
    model = nn.DataParallel(model) 
    criterion = nn.CrossEntropyLoss()
    optimizer = parser.get_optimizer(model) 
    scheduler = parser.get_scheduler(optimizer)
    num_epochs = max_num_epochs
    patience_count = patience
    for epoch in range(max_num_epochs):
        result = one_iter(model, criterion, train_loader,
                            device, 
                            train=True, 
                            optimizer=optimizer,
                            scheduler=scheduler,
                            monitoring_metrics=['accuracy', 'balanced_accuracy'])
        loss = result['loss']
        accuracy = result['accuracy']
        balanced_accuracy = result['balanced_accuracy']
        print(f"Epoch {epoch+1}, \
            Train loss: {loss}, \
            train accuracy: {accuracy:.6f},\
            balanced train accuracy: {balanced_accuracy:.6f}")

        
        ### validation ###
        monitoring_metrics=['accuracy', 'balanced_accuracy', 'macro_avg_precision', 'macro_avg_f1_score']
        if is_enable_report and (epoch + 1) % report_frequency == 0:
            monitoring_metrics += [
                metric for metric, enabled in [
                    ("confusion_matrix", is_enable_confusion_matrix),
                    ("recall_per_class", is_enable_stats_per_class),
                    ("precision_per_class", is_enable_stats_per_class),
                    ("f1_score_per_class", is_enable_stats_per_class)
                ] if enabled
            ]
        
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

        if "confusion_matrix" in result:
            cm = result["confusion_matrix"]
            class_names = val_dataset.classes
            plot_confusion_matrix(cm, class_names, epoch, checkpoint_path)

        if is_enable_stats_per_class:
            recall_per_class = result["recall_per_class"]
            precision_per_class = result["precision_per_class"]
            f1_score_per_class = result["f1_score_per_class"]
            epoch_df = pd.DataFrame({
                    "Epoch": epoch + 1,
                    "Class ID": range(1, len(recall_per_class) + 1),  
                    "Class Name": val_dataset.classes,
                    "Recall": recall_per_class,
                    "Precision": precision_per_class,
                    "F1_Score": f1_score_per_class
                })
            stats_df = epoch_df if stats_df.empty else pd.concat([stats_df, epoch_df], ignore_index=True)

        # early stopping
        metric_value = result[early_stopping_metric]
        if compare_op(metric_value, best_metric):
            best_metric = metric_value
            best_model_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            patience_count = patience
        else:
            patience_count -= 1
            if patience_count == 0:
                num_epochs = epoch + 1
                break
        

    print('Finished Training')
    if is_checkpoint:
        with open(report_path, "a") as report_file:
            report_file.write("Valid Set Metrics:\n")
            report_file.write(f"Accuracy: {balanced_accuracy:.6f}\n")
            report_file.write(f"Balanced Accuracy: {accuracy:.6f}\n")
            report_file.write(f"Macro Avg Precision: {macro_avg_precision:.6f}\n")
            report_file.write(f"Macro Avg F1 Score: {macro_avg_f1_score:.6f}\n")
 
    # test
    model.load_state_dict(best_model_weights)

    result = one_iter(model, criterion, test_loader,
                            device,
                            train=False,
                            monitoring_metrics=['accuracy', 'balanced_accuracy', 'macro_avg_precision', 'macro_avg_f1_score'])
    loss = result['loss']
    accuracy = result['accuracy']
    balanced_accuracy = result['balanced_accuracy']
    macro_avg_precision = result['macro_avg_precision']
    macro_avg_f1_score = result['macro_avg_f1_score']
    print(f"Test loss: {loss}, \
            test accuracy: {accuracy:.6f},\
            balanced test accuracy: {balanced_accuracy:.6f}")

    if is_enable_stats_per_class:
        stats_df.to_csv(stats_df_path, index=False)
    if is_checkpoint:
        torch.save(model.state_dict(), best_model_path)
        parser.save_training_parameters(training_parameters_path,
                                num_epochs=num_epochs,
                                best_epoch=best_epoch)
        with open(report_path, "a") as report_file:
            report_file.write("Test Set Metrics:\n")
            report_file.write(f"Accuracy: {balanced_accuracy:.6f}\n")
            report_file.write(f"Balanced Accuracy: {accuracy:.6f}\n")
            report_file.write(f"Macro Avg Precision: {macro_avg_precision:.6f}\n")
            report_file.write(f"Macro Avg F1 Score: {macro_avg_f1_score:.6f}\n")


if __name__ == "__main__":
    main()