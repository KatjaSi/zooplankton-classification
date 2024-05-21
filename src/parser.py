import yaml
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.optim.lr_scheduler import CosineAnnealingLR


class Parser():

    def __init__(self, config_file_path='src/config.yaml'):
        with open('src/config.yaml', 'r') as file:
            config_file = yaml.safe_load(file)
            self.num_classes = config_file['num_classes']
            self.num_workers = config_file['num_workers']
            self.batch_size = config_file['batch_size']
            self.num_epochs = config_file['num_epochs']
            self.report_config = config_file['reports']
            self.dataset = config_file['dataset']
            self.model_name = config_file['model']['name']
            self.pretrained = config_file['model']['pretrained']
            self.optimizer = config_file['optimizer']
            self.scheduler = config_file['scheduler']
            self.fine_tune = config_file['model']['fine_tune']
            self.freeze_until = config_file['model']['freeze_until']

    def get_num_classes(self):
        return self.num_classes

    def get_num_workers(self):
        return self.num_workers
    
    def get_batch_size(self):
        return self.batch_size

    def get_num_epochs(self):
        return self.num_epochs

    def get_dataset_name(self):
        return self.dataset

    #def get_report_config(self):
     #   return self.report_config

    def is_enable_report(self):
        return self.report_config['enable']
    
    def get_report_frequency(self):
        return self.report_config['frequency']

    def is_enable_confusion_matrix(self):
        return self.report_config['types']['confusion_matrix']

    def is_enable_stats_per_class(self):
        return self.report_config['types']['stats_per_class']

    def get_model(self):
        model_name = self.model_name
        pretrained = self.pretrained
        num_classes = self.num_classes
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
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        if not self.fine_tune:
            found_layer = False
            for name, param in model.named_parameters():
                if self.freeze_until in name:
                    found_layer = True
                    print(f"Freezing up to {name}")
                    break
                param.requires_grad = False
            assert found_layer, f"Layer named {self.freeze_until} not found in the model."
        return model

    def get_model_name(self):
        return self.model_name

    def get_optimizer(self, model):
        optimizer_type = self.optimizer['type']
        lr = self.optimizer['lr']
        if optimizer_type == 'Adam':
            return optim.Adam(params=model.parameters(), lr=lr)
        elif optimizer_type=='SGD':
            return optim.SGD(params=model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    def get_scheduler(self, optimizer):
        if self.scheduler['enable']:
            if self.scheduler['type'] == 'CosineAnnealingLR':
                scheduler = CosineAnnealingLR(optimizer, T_max=self.scheduler['T_max'], eta_min=self.scheduler['eta_min'])
                return scheduler
            else:
                raise ValueError(f"Unsupported scheduler type: {self.scheduler['type']}")
        return None

    def is_enable_scheduler(self):
        return self.scheduler['enable']


    def save_training_parameters(self, file_path, num_epochs=None):
        num_gpus = 0
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpu_types = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        else:
            gpu_types = []
        num_epochs = num_epochs if num_epochs is not None else self.num_epochs
        model = {
            'name': self.get_model_name(),
            'pretrained': self.pretrained,
            'fine_tune': self.fine_tune,
            'freeze_until': self.freeze_until
        }
        training_parameters = {
            'num_gpus': num_gpus,
            'gpu_types': gpu_types,
            'num_workers': self.num_workers,
            'batch_size': self.batch_size,
            'num_epochs': num_epochs,
            'model': model,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'dataset': self.dataset
        }

        with open(file_path, 'w') as json_file:
            json.dump(training_parameters, json_file, indent=4) # type: ignore