import torch
import torch.optim as optim

class NGD(optim.SGD):
    def _init_(self, params, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super(NGD, self)._init_(params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)

    def step(self, metric=None, df_dw=None, closure=None, zeta=0.1):
        if metric is None:
            super(NGD, self).step(closure)
        else:
            # Perform some custom logic before calling the base class's step method
            for param_group in self.param_groups:
                for i, param in enumerate(param_group['params']):
                    if param.grad is None:
                        continue 
                    df_dw[i] = df_dw[i].view(-1, 1)
                    G = df_dw[i] @ metric @ df_dw[i].T
                    G = G + zeta * torch.eye(len(G))
                    G_inv = torch.inverse(G)
                    dp = G_inv @ param.grad.view(-1, 1)
                    param.grad = dp.view(param.grad.shape)
            super(NGD, self).step(closure)