import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import os
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import WeightedRandomSampler
from plots import plot_confusion_matrix
from parsers import TrainConfigParser # type: ignore
from common.transforms import  resize_and_pad, ResizeAndPad
import torch.optim as optim
import csv
import pandas as pd
import copy
import ipdb

from train_utils import one_iter
from common.dataset import ZooScanImageFolder

def main():

    parser = TrainConfigParser()

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    batch_size = parser.get_batch_size()*torch.cuda.device_count()
    dataset = parser.get_dataset_name()
    num_workers = parser.get_num_workers()
    mean = parser.get_transforms_normalize_mean()
    std = parser.get_transforms_normalize_std()
    val_transform = A.Compose([
            ResizeAndPad(size=224, fill=255),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

    test_dataset  = ZooScanImageFolder(root=f"datasets/{dataset}/test", transform=val_transform, grayscale=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=20)
    model = parser.get_model().to(device)
    model.to(device)
    model = nn.DataParallel(model)

    for i in range(10,11):
        state_dict = torch.load(f"checkpoints/swin_manual_augs/run_{i}/best_model.pth")
        stats_df_test_path = f"checkpoints/swin_manual_augs/run_{i}/stats_df_test.csv"
        model.load_state_dict(state_dict)

        columns = ["Class ID", "Recall", "Precision", "F1_Score"] \
                            +  [f"Misclassification {i+1}" for i in range(len(test_dataset.classes))]
        stats_df_test = pd.DataFrame(columns=columns)
        criterion = nn.CrossEntropyLoss()
        result = one_iter(model, criterion, test_loader,
                                            device,
                                            train=False,
                                            monitoring_metrics=[
                                                "accuracy",
                                                "balanced_accuracy",
                                                "confusion_matrix",
                                                "recall_per_class", 
                                                "precision_per_class", 
                                                "f1_score_per_class",
                                                "confusion_matrix"
                                            ])

        test_accuracy = result['accuracy']
        test_balanced_accuracy = result['balanced_accuracy']
            
        print(f"test accuracy: {test_accuracy:.6f},\
                        balanced test accuracy: {test_balanced_accuracy:.6f}")

        class_names = test_dataset.classes
        recall_per_class = result["recall_per_class"]
        precision_per_class = result["precision_per_class"]
        f1_score_per_class = result["f1_score_per_class"]
        cm = result["confusion_matrix"]
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        rows = []
        for class_idx, class_name in enumerate(test_dataset.classes):
            row = { "Class ID": class_idx + 1,
                    "Class Name": class_name,
                    "Recall": recall_per_class[class_idx],
                    "Precision": precision_per_class[class_idx],
                    "F1_Score": f1_score_per_class[class_idx]
                        }
            for i, confusion_percentage in enumerate(cm_normalized[class_idx]):
                row[f"Misclassification {i+1}"] = confusion_percentage
            rows.append(row)
                        
            stats_df_test = pd.DataFrame(rows)
            stats_df_test.to_csv(stats_df_test_path, index=False)
                        

                  

if __name__ == "__main__":
    main()