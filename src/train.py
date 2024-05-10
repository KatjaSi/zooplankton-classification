import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import sklearn.metrics as metrics
import shutil
import os
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from plots import plot_confusion_matrix
from parser import Parser
from transforms import apply_clahe, resize_and_pad

import torch.optim as optim

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'



if __name__ == "__main__":

    parser = Parser()

    device = torch.device("cuda")

    graph_path = 'graphs'
    if os.path.exists(graph_path):
        shutil.rmtree(graph_path)
    os.makedirs(graph_path)

    in_channels = 1
    num_classes = parser.get_num_classes()
    batch_size = parser.get_batch_size()
    num_epochs = parser.get_num_epochs()
    dataset = parser.get_dataset_name()
    is_enable_report = parser.is_enable_report() 
    report_frequency = parser.get_report_frequency()
    is_enable_confusion_matrix = parser.is_enable_confusion_matrix()


    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda img: resize_and_pad(img)),  
        transforms.ToTensor(),
       # transforms.Lambda(apply_clahe),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Lambda(lambda img: resize_and_pad(img)),  
        transforms.ToTensor(),
       # transforms.Lambda(apply_clahe),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = ImageFolder(root=f"datasets/{dataset}/train", transform=train_transform)
    val_dataset = ImageFolder(root=f"datasets/{dataset}/val", transform=val_transform)
    test_dataset  = ImageFolder(root=f"datasets/{dataset}/test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    model = parser.get_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = parser.get_optimizer(model) #optim.Adam(params=model.parameters(), lr=0.0005)
    
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