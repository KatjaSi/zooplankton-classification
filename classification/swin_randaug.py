from torchvision.transforms import RandAugment, Compose, Normalize, ToTensor
from torchvision.transforms.functional import InterpolationMode
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import shutil
import os
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import WeightedRandomSampler
from plots import plot_confusion_matrix
from parsers import TrainConfigParser # type: ignore
from common.transforms import apply_clahe, resize_and_pad, ResizeAndPadTorch
from datetime import datetime
from transformers import ViTMAEModel, ViTForImageClassification, ViTConfig, ViTMAEForPreTraining
import torch.optim as optim
import csv
import pandas as pd
import copy
import ipdb

from train_utils import one_iter
from common.dataset import ZooScanImageFolderTorch

def main():

    parser = TrainConfigParser()

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    batch_size = parser.get_batch_size() * torch.cuda.device_count()
    max_num_epochs = parser.get_max_num_epochs()
    dataset = parser.get_dataset_name()
    is_enable_report = parser.is_enable_report()
    report_frequency = parser.get_report_frequency()
    num_workers = parser.get_num_workers()
    is_checkpoint = parser.is_checkpoint()
    early_stopping_metric = parser.get_early_stopping_metric()
    compare_op = parser.get_compare_operator()
    mean = parser.get_transforms_normalize_mean()
    std = parser.get_transforms_normalize_std()

    # early stopping setup
    for i in range(10):
        best_metric = float('-inf') if early_stopping_metric in ["accuracy", "balanced_accuracy"] else float('inf')
        best_model_weights = None
        patience = parser.get_patience()
        best_epoch = 0
        best_val_accuracy = float('-inf')
        best_val_balanced_accuracy = float('-inf')
        best_val_macro_avg_precision = float('-inf')
        best_val_macro_avg_f1_score = float('-inf')

        checkpoint_path = os.path.join('checkpoints', 'swin_randaug', f"run_{i+1}")

        if is_enable_report:
            os.makedirs(checkpoint_path, exist_ok=True)

        training_parameters_path = os.path.join(checkpoint_path, 'training_parameters')
        stats_df_path = os.path.join(checkpoint_path, 'stats_df.csv')
        best_model_path = os.path.join(checkpoint_path, "best_model.pth")
        report_path = os.path.join(checkpoint_path, "report.txt")
        epoch_log_path = os.path.join(checkpoint_path, "epoch_metrics.csv")


        train_transform = Compose([
            ResizeAndPadTorch(224, fill=255),
            RandAugment(num_ops=2, magnitude=9),  # Adjust num_ops and magnitude as needed
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])

 
        val_transform = Compose([
            ResizeAndPadTorch(size=224, fill=255),
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])

        train_dataset = ZooScanImageFolderTorch(root=f"datasets/{dataset}/train", transform=train_transform, grayscale=False)
        val_dataset = ZooScanImageFolderTorch(root=f"datasets/{dataset}/val", transform=val_transform, grayscale=False)
        test_dataset  = ZooScanImageFolderTorch(root=f"datasets/{dataset}/test", transform=val_transform, grayscale=False)

        if is_enable_report:
            columns = ["Epoch", "Class ID", "Recall", "Precision", "F1_Score"] \
                        +  [f"Misclassification {i+1}" for i in range(len(val_dataset.classes))]
            stats_df = pd.DataFrame(columns=columns)


        # weighted random sampler
        train_labels = np.array(train_dataset.targets)
        class_counts = np.bincount(train_labels)
        class_weights = 1. / class_counts
        # Apply weights to each sample
        weights = class_weights[train_labels]
        sampler = WeightedRandomSampler(weights, len(weights))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=20)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=20)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=20)

        model = parser.get_model().to(device)

        model.to(device)
        model = nn.DataParallel(model)
        #model.load_state_dict(state_dict, strict=False)
        criterion = nn.CrossEntropyLoss()
        optimizer =  torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-8)
        scheduler = None
        num_epochs = max_num_epochs
        patience_count = patience
        with open(epoch_log_path, mode='w', newline='') as log_file:
            csv_writer = csv.writer(log_file)
            csv_writer.writerow([
                "Epoch", 
                "Train Loss", "Train Accuracy", "Train Balanced Accuracy", 
                "Validation Loss", "Validation Accuracy", "Validation Balanced Accuracy"
            ])
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
                            "confusion_matrix",
                            "recall_per_class", 
                            "precision_per_class", 
                            "f1_score_per_class"
                        ]
                
                result = one_iter(model, criterion, val_loader,
                                        device,
                                        train=False,
                                        monitoring_metrics=monitoring_metrics)
                val_loss = result['loss']
                val_accuracy = result['accuracy']
                val_balanced_accuracy = result['balanced_accuracy']
                macro_avg_precision = result['macro_avg_precision']
                macro_avg_f1_score = result['macro_avg_f1_score']
                print(f"Epoch {epoch+1}, \
                    Valid loss: {val_loss}, \
                    valid accuracy: {val_accuracy:.6f},\
                    balanced valid accuracy: {val_balanced_accuracy:.6f}")
                    
                if is_enable_report:
                    cm = result["confusion_matrix"]
                    class_names = val_dataset.classes
                    plot_confusion_matrix(cm, class_names, epoch, checkpoint_path)
                    recall_per_class = result["recall_per_class"]
                    precision_per_class = result["precision_per_class"]
                    f1_score_per_class = result["f1_score_per_class"]

                    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    rows = []
                    for class_idx, class_name in enumerate(val_dataset.classes):
                        row = {
                            "Epoch": epoch + 1,
                            "Class ID": class_idx + 1,
                            "Class Name": class_name,
                            "Recall": recall_per_class[class_idx],
                            "Precision": precision_per_class[class_idx],
                            "F1_Score": f1_score_per_class[class_idx]
                        }
                        for i, confusion_percentage in enumerate(cm_normalized[class_idx]):
                            row[f"Misclassification {i+1}"] = confusion_percentage
                        rows.append(row)
                    
                    epoch_df = pd.DataFrame(rows)
                    stats_df = epoch_df if stats_df.empty else pd.concat([stats_df, epoch_df], ignore_index=True)

                    csv_writer.writerow([
                    epoch + 1,
                    loss, accuracy, balanced_accuracy,
                    val_loss, val_accuracy, val_balanced_accuracy
                        ])

                # early stopping
                metric_value = result[early_stopping_metric]
                if compare_op(metric_value, best_metric):
                    best_metric = metric_value
                    best_model_weights = copy.deepcopy(model.state_dict())
                    best_epoch = epoch + 1
                    patience_count = patience

                    best_val_accuracy = val_accuracy
                    best_val_balanced_accuracy = val_balanced_accuracy
                    best_val_macro_avg_precision = macro_avg_precision
                    best_val_macro_avg_f1_score = macro_avg_f1_score
                else:
                    patience_count -= 1
                    if patience_count == 0:
                        num_epochs = epoch + 1
                        break
                

        print('Finished Training')
        if is_checkpoint:
            with open(report_path, "a") as report_file:
                report_file.write("Valid Set Metrics:\n")
                report_file.write(f"Accuracy: {best_val_accuracy:.6f}\n")
                report_file.write(f"Balanced Accuracy: {best_val_balanced_accuracy:.6f}\n")
                report_file.write(f"Macro Avg Precision: {best_val_macro_avg_precision:.6f}\n")
                report_file.write(f"Macro Avg F1 Score: {best_val_macro_avg_f1_score:.6f}\n")
    
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

        if is_enable_report:
            stats_df.to_csv(stats_df_path, index=False)
        if is_checkpoint:
            torch.save(model.state_dict(), best_model_path)
            parser.save_training_parameters(training_parameters_path,
                                    num_epochs=num_epochs,
                                    best_epoch=best_epoch)
            with open(report_path, "a") as report_file:
                report_file.write("Test Set Metrics:\n")
                report_file.write(f"Accuracy: {accuracy:.6f}\n")
                report_file.write(f"Balanced Accuracy: {balanced_accuracy:.6f}\n")
                report_file.write(f"Macro Avg Precision: {macro_avg_precision:.6f}\n")
                report_file.write(f"Macro Avg F1 Score: {macro_avg_f1_score:.6f}\n")


if __name__ == "__main__":
    main()