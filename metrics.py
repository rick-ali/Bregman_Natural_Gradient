import torch
from torch import nn 

# bregman metrics
class BregmanMetric():
    def __init__(self, **kwargs):
        pass

    def __call__(self, p, q):
        return self.hessian(p, q)

    def hessian(self, p, q):
        pass

class SquaredMetric(BregmanMetric):
    def hessian(self, p, q):
        return torch.full_like(p, 1.)
    
