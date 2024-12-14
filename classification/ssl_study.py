import sys
import os
import csv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import yaml
import numpy as np
import albumentations as A
from common.transforms import apply_clahe, resize_and_pad, ResizeAndPad
from albumentations.pytorch import ToTensorV2
import copy
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import WeightedRandomSampler
from train_utils import one_iter
from parsers import TrainConfigParser
from transformers import ViTMAEModel, ViTForImageClassification, ViTConfig, ViTMAEForPreTraining
from common.dataset import ZooScanImageFolder

import ipdb

def main():
    parser = TrainConfigParser()

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    batch_size = parser.get_batch_size()*torch.cuda.device_count() 
    max_num_epochs = parser.get_max_num_epochs()
    dataset = parser.get_dataset_name()
    is_enable_report = parser.is_enable_report()
    report_frequency = parser.get_report_frequency()
    num_workers = parser.get_num_workers()

    early_stopping_metric = parser.get_early_stopping_metric()
    compare_op = parser.get_compare_operator()
    mean = parser.get_transforms_normalize_mean()
    std = parser.get_transforms_normalize_std()
    patience = parser.get_patience()
    train_transform = A.Compose([
        ResizeAndPad(224, fill=255), 
        A.OneOf([
                            A.HorizontalFlip(p=0.5),
                            A.RandomRotate90(p=0.5),
                            A.VerticalFlip(p=0.5),
        ], p=1),
        A.Normalize(mean=mean, std=std),
        ToTensorV2() ])

    val_transform = A.Compose([
        ResizeAndPad(size=224, fill=255),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    train_dataset = ZooScanImageFolder(root=f"datasets/{dataset}/train", transform=train_transform, grayscale=True)
    val_dataset = ZooScanImageFolder(root=f"datasets/{dataset}/val", transform=val_transform, grayscale=True)
    test_dataset  = ZooScanImageFolder(root=f"datasets/{dataset}/test", transform=val_transform, grayscale=True)

    train_labels = np.array(train_dataset.targets)
    class_counts = np.bincount(train_labels)
    class_weights = 1. / class_counts
    weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=20)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=20)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=20)


    results_file = f"ssl_study/finetuning/vit_mae_base_npl_tru_in_donain.csv"

    if not os.path.exists(results_file):
        with open(results_file, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(
                output_file,
                fieldnames=[
                        'attempt',
                        'val_loss',
                        'val_accuracy',
                        'val_balanced_accuracy',
                        'val_macro_avg_precision',
                        'val_macro_avg_f1_score',
                        'test_loss',
                        'test_accuracy',
                        'test_balanced_accuracy',
                        'test_macro_avg_precision',
                        'test_macro_avg_f1_score'
                    ]
                )
            dict_writer.writeheader()
        

    for i in range(10):
        best_metric = float('-inf') if early_stopping_metric in ["accuracy", "balanced_accuracy"] else float('inf')
        best_model_weights = None
        best_epoch = 0
        patience_count = patience
        best_val_loss = float('inf')
        best_val_accuracy = float('-inf')
        best_val_balanced_accuracy = float('-inf')
        best_val_macro_avg_precision = float('-inf')
        best_val_macro_avg_f1_score = float('-inf')

        
        #model = ViTForImageClassification.from_pretrained("facebook/vit-mae-base", num_labels=77, ignore_mismatched_sizes=True)
        config = ViTConfig(
            num_labels=77,
            num_channels=1,
         #   hidden_size = 1024,
         #   intermediate_size = 4096,
         #   num_attention_heads = 16,
         #   num_hidden_layers = 24,
            ignore_mismatched_sizes=True
    )
        model = ViTForImageClassification(config)
        state_dict = torch.load("checkpoints/checkpoints_mae_base_npl/swa_final_model.pth")
        model.to(device)
        model = nn.DataParallel(model)
        # Fine tuning
        for param in model.module.parameters():
            param.requires_grad = False

        for param in model.module.classifier.parameters():
            param.requires_grad = True

        ####

        model.load_state_dict(state_dict, strict=False)
        criterion = nn.CrossEntropyLoss()
        #optimizer = parser.get_optimizer(model)
        optimizer = optim.Adam(params=model.module.classifier.parameters(), lr=1e-4)
        scheduler = parser.get_scheduler(optimizer)
        num_epochs = max_num_epochs
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
                
            
            metric_value = result[early_stopping_metric]
            if compare_op(metric_value, best_metric):
                best_metric = metric_value
                best_model_weights = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                patience_count = patience

                best_val_accuracy = accuracy
                best_val_balanced_accuracy = balanced_accuracy
                best_val_macro_avg_precision = macro_avg_precision
                best_val_macro_avg_f1_score = macro_avg_f1_score
                best_val_loss = val_loss
            else:
                patience_count -= 1
                if patience_count == 0:
                    num_epochs = epoch + 1
                    break
            

        print('Finished Training')
   
        model.load_state_dict(best_model_weights)

        result = one_iter(model, criterion, test_loader,
                                device,
                                train=False,
                                monitoring_metrics=['accuracy', 'balanced_accuracy', 'macro_avg_precision', 'macro_avg_f1_score'])
        test_loss = result['loss']
        test_accuracy = result['accuracy']
        test_balanced_accuracy = result['balanced_accuracy']
        test_macro_avg_precision = result['macro_avg_precision']
        test_macro_avg_f1_score = result['macro_avg_f1_score']
        print(f"Test loss: {test_loss}, \
                test accuracy: {test_accuracy:.6f},\
                balanced test accuracy: {test_balanced_accuracy:.6f}")
        
        with open(results_file, 'a', newline='') as output_file:
            dict_writer = csv.DictWriter(
                output_file,
                fieldnames=[
                    'attempt',
                    'val_loss',
                    'val_accuracy',
                    'val_balanced_accuracy',
                    'val_macro_avg_precision',
                    'val_macro_avg_f1_score',
                    'test_loss',
                    'test_accuracy',
                    'test_balanced_accuracy',
                    'test_macro_avg_precision',
                    'test_macro_avg_f1_score'
                ]
            )

            dict_writer.writerow({
                'attempt': i + 1,
                'val_loss': best_val_loss,
                'val_accuracy': best_val_accuracy,
                'val_balanced_accuracy': best_val_balanced_accuracy,
                'val_macro_avg_precision': best_val_macro_avg_precision,
                'val_macro_avg_f1_score': best_val_macro_avg_f1_score,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'test_balanced_accuracy': test_balanced_accuracy,
                'test_macro_avg_precision': test_macro_avg_precision,
                'test_macro_avg_f1_score': test_macro_avg_f1_score
            })

            

if __name__ == "__main__":
    main()