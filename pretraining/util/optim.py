from torch.optim import AdamW
import ipdb
def get_optimizer(lr, weight_decay, param_groups):
    optimizer = AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    for param_group in param_groups:
        param_group['lr'] =param_group['lr']*param_group['lr_scale']
    return optimizer

