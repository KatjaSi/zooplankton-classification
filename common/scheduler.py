from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR


class CosineAnnealingWithWarmUp():

    def __init__(self, warmup_steps, total_steps, optimizer, warmup_fraction, eta_min=0):
        """
        Initializes the Cosine Annealing with Warm Up scheduler.

        Parameters:
        - warmup_steps: Number of steps for the warmup phase.
        - total_steps: Total number of steps for the entire training process.
        - optimizer: The optimizer being used.
        - warmup_start_factor: Starting factor for the warmup phase.
        - eta_min: Minimum learning rate for the cosine annealing phase. Default: 0

        After total_steps, the learning rate will remain unchanged.
        """
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.warmup_fraction = warmup_fraction
        self.total_steps = total_steps
        self.warmup_scheduler = LinearLR(optimizer, start_factor=warmup_fraction, end_factor=1.0, total_iters=warmup_steps)
        annealing_steps = total_steps-warmup_steps
        self.annealing_scheduler = CosineAnnealingLR(optimizer, T_max=annealing_steps, eta_min=eta_min, last_epoch=-1)
        self.step_count = 0

    def step(self):
        if self.step_count < self.warmup_steps:
            self.warmup_scheduler.step()
        else:
            if self.step_count < self.total_steps:
                self.annealing_scheduler.step()
        self.step_count += 1

    def state_dict(self):
        return {
            'step_count': self.step_count,
            'warmup_scheduler': self.warmup_scheduler.state_dict(),
            'annealing_scheduler': self.annealing_scheduler.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.step_count = state_dict['step_count']
        self.warmup_scheduler.load_state_dict(state_dict['warmup_scheduler'])
        self.annealing_scheduler.load_state_dict(state_dict['annealing_scheduler'])

