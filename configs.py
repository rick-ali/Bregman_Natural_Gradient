from dataclasses import dataclass

@dataclass
class ExperimentConfig_1:
    name = 'config_1'
    bop = 'max'
    num_samples = 1000
    model_type = 'simple'
    loss_type = 'mse'
    lr = 0.01
    num_epochs = 100
