import yaml
import torchvision
import torch.nn as nn
import torch.optim as optim
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
