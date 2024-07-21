import yaml
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import json
import operator
import os
import re
import timm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.datasets import ImageFolder

from transformers import ViTForImageClassification, ViTConfig, DeiTForImageClassification
from transformers import  DeiTForImageClassification, DeiTConfig, DeiTForImageClassificationWithTeacher
from transformers import SwinForImageClassification, SwinConfig


class TrainConfigParser():

    def __init__(self, config_file_path='src/config.yaml'):
        with open(config_file_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def get_num_classes(self):
        return self.config['num_classes']

    def get_num_workers(self):
        return self.config['num_workers']
    
    def get_batch_size(self):
        return self.config['batch_size']

    def get_max_num_epochs(self):
        return self.config['max_num_epochs']
    
    def get_patience(self):
        return self.config['early_stopping']['patience']

    def get_early_stopping_metric(self):
        return self.config['early_stopping']['early_stopping_metric']

    def get_dataset_name(self):
        return self.config['dataset']

    def is_enable_report(self):
        return self.config['reports']['enable']
    
    def get_report_frequency(self):
        return self.config['reports']['frequency']

    def is_model_distilled(self):
        return self.config['distilled']

    def get_model(self):
        model_name = self.config['model']['name']
        pretrained = self.config['model']['pretrained']
        num_classes = self.get_num_classes()
        if model_name == 'vgg16':
            model = torchvision.models.vgg16(weights=pretrained)
            model.classifier[-1] = nn.Linear(in_features=4096, out_features=num_classes)
        elif model_name == 'resnet50':
            model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None)
            model.fc = nn.Linear(in_features=2048, out_features=num_classes)
        elif model_name == 'resnet34':
            model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT if pretrained else None)
            model.fc = nn.Linear(in_features=512, out_features=num_classes)
        elif model_name == 'googlenet':
            model = torchvision.models.googlenet(pretrained=pretrained)
            model.fc = nn.Linear(in_features=1024, out_features=num_classes)
        elif model_name == 'vit':
            if pretrained:
                model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', #'google/vit-base-patch16-224',
                                                                    num_labels=num_classes,
                                                                    ignore_mismatched_sizes=True)
            else:
                config = ViTConfig()
                config.num_labels = num_classes
                model = ViTForImageClassification(config)
        elif model_name == 'deit':
            distilled = self.is_model_distilled()
            if distilled:
                if pretrained:
                    model = DeiTForImageClassificationWithTeacher.from_pretrained(
                        'facebook/deit-base-distilled-patch16-224',
                        num_labels=num_classes,
                        ignore_mismatched_sizes=True
                    )
                else:
                    config = DeiTConfig()
                    config.num_labels = num_classes
                    model = DeiTForImageClassificationWithTeacher(config)
            else:
                if pretrained:
                    model = DeiTForImageClassification.from_pretrained('facebook/deit-base-distilled-patch16-224',
                                                                       num_labels=num_classes,
                                                                      ignore_mismatched_sizes=True)
                else:
                    config = DeiTConfig()
                    config.num_labels = num_classes
                    model = DeiTForImageClassification(config)
        elif model_name == "swin":
            if pretrained:
                model = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224",
                                                                    num_labels=num_classes,
                                                                    ignore_mismatched_sizes=True)
            else:
                config = SwinConfig()
                config.num_labels = num_classes
                model = SwinForImageClassification(config)

        else:
            model = timm.create_model(model_name, pretrained=pretrained, num_classes=77)
            #raise ValueError(f"Unsupported model name: {model_name}")

        if not self.config['model']['fine_tune']:
            found_layer = False
            for name, param in model.named_parameters():
                if self.config['model']['freeze_until'] in name:
                    found_layer = True
                    print(f"Freezing up to {name}")
                    break
                param.requires_grad = False
            assert found_layer, f"Layer named {self.config['model']['freeze_until']} not found in the model."
        return model

    def get_model_name(self):
        return self.config['model']['name']

    def get_optimizer(self, model):
        optimizer = self.config['optimizer']
        optimizer_type = optimizer['type']
        lr = optimizer['lr']
        if optimizer_type == 'Adam':
            return optim.Adam(params=model.parameters(), lr=lr)
        elif optimizer_type=='SGD':
            return optim.SGD(params=model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    def get_scheduler(self, optimizer):
        if self.config['scheduler']['enable']:
            if self.config['scheduler']['type'] == 'CosineAnnealingLR':
                scheduler = CosineAnnealingLR(optimizer, T_max=self.config['scheduler']['T_max'], eta_min=self.config['scheduler']['eta_min'])
                return scheduler
            else:
                raise ValueError(f"Unsupported scheduler type: {self.config['scheduler']['type']}")
        return None

    def is_enable_scheduler(self):
        return self.config['scheduler']['enable']


    def save_training_parameters(self, file_path, num_epochs=None, best_epoch=None):
        num_gpus = 0
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpu_types = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        else:
            gpu_types = []
        num_epochs = num_epochs if num_epochs is not None else self.get_max_num_epochs()
        model = {
            'name': self.get_model_name(),
            'pretrained': self.config['model']['pretrained'],
            'fine_tune': self.config['model']['fine_tune'],
            'freeze_until': self.config['model']['freeze_until']
        }
        training_parameters = {
            'num_gpus': num_gpus,
            'gpu_types': gpu_types,
            'num_workers': self.get_num_workers(),
            'batch_size': self.get_batch_size(),
            'num_epochs': num_epochs,
            'model': model,
            'optimizer': self.config['optimizer'],
            'scheduler': self.config['scheduler'],
            'dataset': self.get_dataset_name()
        }
        if self.is_model_distilled() is not None:
            training_parameters['distilled'] = self.is_model_distilled()
        if best_epoch is not None:
            training_parameters['best_epoch'] = best_epoch
        with open(file_path, 'w') as json_file:
            json.dump(training_parameters, json_file, indent=4) # type: ignore

    def is_checkpoint(self):
        return self.config['checkpoint']

    def get_compare_operator(self):
        metric_name = self.get_early_stopping_metric()
        if metric_name in ["accuracy", "balanced_accuracy"]:
            return operator.gt
        elif metric_name == "loss":
            return operator.lt
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")

    def get_transforms_normalize_mean(self):
        return self.config['transforms']['normalize']['mean']
    
    def get_transforms_normalize_std(self):
        return self.config['transforms']['normalize']['std']
    

class MetricPlotterConfigParser():

    def __init__(self, config_file_path='src/metric_plotter_config.yaml'):
        with open(config_file_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def get_csv_file_path(self):
        return os.path.join(self.config['csv_file_folder_path'], 'stats_df.csv')

    def get_output_img_folder_path(self):
        return self.config['output_img_folder_path']

    def get_metric(self):
        return self.config['metric']

    def get_category(self):
        return self.config['category']

    def get_classes(self):
        category = self.get_category()
        if category is None:
            return self.config['classes']
        else:
            all_classes = ImageFolder(root=f"datasets/ZooScan77/train").classes
            pattern = re.compile(rf"^{category}[1-9][0-9]?_")
            classes = [cls for cls in all_classes if pattern.match(cls)]
            return classes

    def get_true_class(self):
        """
        Apply for confusion trends plot
        """
        return self.config['true_class']
            
    def get_top_N(self):
        return self.config["top_N"]

    def get_class_names(self):
        return ImageFolder(root=f"datasets/ZooScan77/train").classes

    def get_epoch_until(self):
        return self.config["epoch_until"]