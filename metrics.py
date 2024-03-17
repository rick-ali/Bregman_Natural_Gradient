import torch


# bregman metrics
class BregmanMetric:
    def __init__(self, **kwargs):
        pass

    def __call__(self, p, q):
        return self.hessian(p, q)

    def hessian(self, p, q):
        pass


class SquaredMetric(BregmanMetric):
    def hessian(self, p, q):
        return torch.full_like(p, 1.0)
        # return p - q

class BCEMetric(BregmanMetric):
    def hessian(self, p, q):
        hessian = torch.zeros_like(p)
        hessian[q == 0] = p[q == 0] ** -2
        hessian[q == 1] = (1 - p[q == 1]) ** -2
        return hessian

class KLMetric(BregmanMetric):
    def hessian(self, p, q):
        # hessian = 1 / p ** 2
        hessian = 1/p
        hessian = 1 + torch.log(p)
        return hessian