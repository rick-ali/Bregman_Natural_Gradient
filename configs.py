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

@dataclass
class AddMSERegConfig:
    name = "addmsereg"
    # function to learn
    bop = "add"
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


class AddMSEPullConfig:
    name = "addmsepull"
    # function to learn
    bop = "add"
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

@dataclass
class SubMSERegConfig:
    name = "submsereg"
    # function to learn
    bop = "sub"
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


class SubMSEPullConfig:
    name = "submsepull"
    # function to learn
    bop = "sub"
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

@dataclass
class UnitBCERegConfig:
    name = "unitbcereg"
    # function to learn
    bop = "unit"
    # training hyperparams
    num_samples = 1000
    loss_type = "bce"
    lr = 0.01
    num_epochs = 2000
    pullback = False
    # static architecture
    input_dim = 2
    output_dim = 1
    # dynamic architecture
    model_type = "binary"
    hidden_dim = 10


class UnitBCEPullConfig:
    name = "unitbcepull"
    # function to learn
    bop = "unit"
    # training hyperparams
    num_samples = 1000
    loss_type = "bce"
    lr = 0.01
    num_epochs = 2000
    pullback = True
    # static architecture
    input_dim = 2
    output_dim = 1
    # dynamic architecture
    model_type = "binary"
    hidden_dim = 10

@dataclass
class UnitKLRegConfig:
    name = "unitklreg"
    # function to learn
    bop = "unit"
    # training hyperparams
    num_samples = 1000
    loss_type = "kl"
    lr = 0.01
    num_epochs = 1000
    pullback = False
    # static architecture
    input_dim = 2
    output_dim = 1
    # dynamic architecture
    model_type = "binary"
    hidden_dim = 10


class UnitKLPullConfig:
    name = "unitklpull"
    # function to learn
    bop = "unit"
    # training hyperparams
    num_samples = 1000
    loss_type = "kl"
    lr = 0.01
    num_epochs = 1000
    pullback = True
    # static architecture
    input_dim = 2
    output_dim = 1
    # dynamic architecture
    model_type = "binary"
    hidden_dim = 10