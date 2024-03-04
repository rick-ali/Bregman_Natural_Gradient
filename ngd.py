import torch
import torch.optim as optim


class NGD(optim.SGD):
    def __init__(
        self, params, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False
    ):
        super(NGD, self).__init__(
            params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    def step(self, closure=None, zeta=0.1):
        if "metric" not in self.defaults:
            super(NGD, self).step(closure)
        else:
            # Perform some custom logic before calling the base class's step method
            for param_group in self.param_groups:
                for i, param in enumerate(param_group["params"]):
                    if param.grad is None:
                        continue
                    G = self.defaults["metric"][i]
                    G_ = G + zeta * torch.eye(len(G), device=G.device)
                    G_inv = torch.inverse(G_)
                    dp = G_inv @ param.grad.view(-1, 1)
                    param.grad = dp.view(param.grad.shape)
            super(NGD, self).step(closure)
