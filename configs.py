from dataclasses import dataclass


@dataclass
class MaxMSERegConfig:
    name = "maxmsereg"
    # function to learn
    bop = "max"
    # training hyperparams
    num_samples = 1000
    loss_type = "mse"
    lr = 0.01
    num_epochs = 100
    pullback = False
    # static architecture
    input_dim = 2
    output_dim = 1
    # dynamic architecture
    model_type = "simple"
    hidden_dim = 10


class MaxMSEPullConfig:
    name = "maxmsepull"
    # function to learn
    bop = "max"
    # training hyperparams
    num_samples = 1000
    loss_type = "mse"
    lr = 0.01
    num_epochs = 100
    pullback = True
    # static architecture
    input_dim = 2
    output_dim = 1
    # dynamic architecture
    model_type = "simple"
    hidden_dim = 10
