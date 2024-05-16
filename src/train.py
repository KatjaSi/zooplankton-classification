import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import sklearn.metrics as metrics
import shutil
import os
import json
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import WeightedRandomSampler
from plots import plot_confusion_matrix
from parser import Parser
from transforms import apply_clahe, resize_and_pad
from datetime import datetime

import torch.optim as optim

import os



if __name__ == "__main__":

    parser = Parser()

    device = torch.device("cuda")

    num_classes = parser.get_num_classes()
    batch_size = parser.get_batch_size()
    num_epochs = parser.get_num_epochs()
    dataset = parser.get_dataset_name()
    is_enable_report = parser.is_enable_report() 
    report_frequency = parser.get_report_frequency()
    is_enable_confusion_matrix = parser.is_enable_confusion_matrix()
    num_workers = parser.get_num_workers()

    graph_path = os.path.join('graphs', parser.get_model_name(), datetime.now().strftime('%Y-%m-%d %H:%M'))

    if os.path.exists(graph_path):
        shutil.rmtree(graph_path)
    os.makedirs(graph_path)

    if torch.cuda.is_available():
        gpu_types = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    else:
        gpu_types = []

    training_parameters = {
        'num_gpus': torch.cuda.device_count(),
        'gpu_types': gpu_types
    }

    file_path = os.path.join(graph_path, 'training_parameters')

    with open(file_path, 'w') as json_file:
        json.dump(training_parameters, json_file, indent=4)


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
    test_dataset  = ImageFolder(root=f"datasets/{dataset}/test")


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
    
    for epoch in range(num_epochs):  
        running_loss = 0.0
        count = 0.0
        model.train()

        train_pred = []
        train_true = []
        for data, labels in (train_loader):
            batch_size = len(labels)
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            preds = outputs.max(dim=1)[1]

            count += batch_size
            running_loss += loss.item() * batch_size

            train_true.append(labels.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        if parser.is_enable_scheduler():
            scheduler.step()

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        accuracy = metrics.accuracy_score(train_true, train_pred)
        balanced_accuracy = metrics.balanced_accuracy_score(train_true, train_pred)
        print(f"Epoch {epoch+1}, \
            Train loss: {running_loss*1.0/count}, \
            train accuracy: {accuracy:.6f},\
            balanced train accuracy: {balanced_accuracy:.6f}")

        
        ### validation ###
        val_loss = 0.0
        count = 0.0
        model.eval()
        val_pred = []
        val_true = []

        for data, labels in (val_loader):
            batch_size = len(labels)
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            preds = outputs.max(dim=1)[1]
            count += batch_size
            val_loss += loss.item() * batch_size
            val_true.append(labels.cpu().numpy())
            val_pred.append(preds.detach().cpu().numpy())

        val_true = np.concatenate(val_true)
        val_pred = np.concatenate(val_pred)
        accuracy = metrics.accuracy_score(val_true, val_pred)
        balanced_accuracy = metrics.balanced_accuracy_score(val_true, val_pred)

        if is_enable_report and (epoch + 1) % report_frequency == 0:
            if is_enable_confusion_matrix:
                cm = metrics.confusion_matrix(val_true, val_pred)
                class_names = val_dataset.classes
                plot_confusion_matrix(cm, class_names, epoch, graph_path)
        
        print(f"Epoch {epoch+1}, \
            Valid loss: {val_loss*1.0/count}, \
            valid accuracy: {accuracy:.6f},\
            balanced valid accuracy: {balanced_accuracy:.6f}\n")


    print('Finished Training')