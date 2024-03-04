from dataclasses import dataclass

@dataclass
class MaxMSERegConfig:
    name = 'maxmsereg'
    bop = 'max'
    num_samples = 1000
    model_type = 'simple'
    loss_type = 'mse'
    lr = 0.01
    num_epochs = 100
    pullback = False

class MaxMSEPullConfig:
    name = 'maxmsepull'
    bop = 'max'
    num_samples = 1000
    model_type = 'simple'
    loss_type = 'mse'
    lr = 0.01
    num_epochs = 100
    pullback = True
