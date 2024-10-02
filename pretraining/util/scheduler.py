import math
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR


class CosineAnnealingWithWarmUpWithLLRD():

    def __init__(self, warmup_steps, total_steps, optimizer, base_lr, min_lr):
        
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.step_count = 0
        self.base_lr = base_lr
        self.lr = base_lr
        self.min_lr = min_lr

    def step(self):
        self.step_count += 1
        if self.step_count < self.warmup_steps:
            self.lr = self.base_lr*self.step_count/self.warmup_steps
            
        elif self.step_count < self.total_steps:
            self.lr = self.min_lr + (self.base_lr - self.min_lr)*0.5*\
                (1.+math.cos(math.pi*(self.step_count-self.warmup_steps)/(self.total_steps-self.warmup_steps)))
        else:
            self.lr = self.min_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr*param_group.get('lr_scale', 1.0)

        return self.lr

    def state_dict(self):
        return {
            'step_count': self.step_count,
            'base_lr': self.base_lr,
            'min_lr': self.min_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps
        }

    def load_state_dict(self, state_dict):
        self.step_count = state_dict['step_count']
        self.base_lr = state_dict['base_lr']
        self.min_lr = state_dict['min_lr']
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']

        self.step()

