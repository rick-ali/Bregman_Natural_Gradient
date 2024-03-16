from dataclasses import dataclass


@dataclass
class MaxMSESGDConfig:
    name = "maxmsesgd"
    # function to learn
    bop = "max"
    # training hyperparams
    num_samples = 1000
    loss_type = "mse"
    lr = 0.01
    num_epochs = 100
    optimizer = 'sgd'
    # static architecture
    input_dim = 2
    output_dim = 1
    # dynamic architecture
    model_type = "simple"
    hidden_dim = 10

@dataclass
class MaxMSEPullConfig:
    name = "maxmsepull"
    # function to learn
    bop = "max"
    # training hyperparams
    num_samples = 1000
    loss_type = "mse"
    lr = 0.01
    num_epochs = 100
    optimizer = 'ngd'
    # static architecture
    input_dim = 2
    output_dim = 1
    # dynamic architecture
    model_type = "simple"
    hidden_dim = 10

@dataclass
class MaxMSEAdamConfig:
    name = "maxmseadam"
    # function to learn
    bop = "max"
    # training hyperparams
    num_samples = 1000
    loss_type = "mse"
    lr = 0.01
    num_epochs = 100
    optimizer = 'adam'
    # static architecture
    input_dim = 2
    output_dim = 1
    # dynamic architecture
    model_type = "simple"
    hidden_dim = 10

@dataclass
class AddMSESGDConfig:
    name = "addmsesgd"
    # function to learn
    bop = "add"
    # training hyperparams
    num_samples = 1000
    loss_type = "mse"
    lr = 0.01
    num_epochs = 100
    optimizer = 'sgd'
    # static architecture
    input_dim = 2
    output_dim = 1
    # dynamic architecture
    model_type = "simple"
    hidden_dim = 10

@dataclass
class AddMSEPullConfig:
    name = "addmsepull"
    # function to learn
    bop = "add"
    # training hyperparams
    num_samples = 1000
    loss_type = "mse"
    lr = 0.01
    num_epochs = 100
    optimizer = 'ngd'
    # static architecture
    input_dim = 2
    output_dim = 1
    # dynamic architecture
    model_type = "simple"
    hidden_dim = 10

@dataclass
class AddMSEAdamConfig:
    name = "addmseadam"
    # function to learn
    bop = "add"
    # training hyperparams
    num_samples = 1000
    loss_type = "mse"
    lr = 0.01
    num_epochs = 100
    optimizer = 'adam'
    # static architecture
    input_dim = 2
    output_dim = 1
    # dynamic architecture
    model_type = "simple"
    hidden_dim = 10

@dataclass
class SubMSESGDConfig:
    name = "submsesgd"
    # function to learn
    bop = "sub"
    # training hyperparams
    num_samples = 1000
    loss_type = "mse"
    lr = 0.01
    num_epochs = 100
    optimizer = 'sgd'
    # static architecture
    input_dim = 2
    output_dim = 1
    # dynamic architecture
    model_type = "simple"
    hidden_dim = 10

@dataclass
class SubMSEPullConfig:
    name = "submsepull"
    # function to learn
    bop = "sub"
    # training hyperparams
    num_samples = 1000
    loss_type = "mse"
    lr = 0.01
    num_epochs = 100
    optimizer = 'ngd'
    # static architecture
    input_dim = 2
    output_dim = 1
    # dynamic architecture
    model_type = "simple"
    hidden_dim = 10

@dataclass
class SubMSEAdamConfig:
    name = "submseadam"
    # function to learn
    bop = "sub"
    # training hyperparams
    num_samples = 1000
    loss_type = "mse"
    lr = 0.01
    num_epochs = 100
    optimizer = 'adam'
    # static architecture
    input_dim = 2
    output_dim = 1
    # dynamic architecture
    model_type = "simple"
    hidden_dim = 10

@dataclass
class UnitBCESGDConfig:
    name = "unitbcesgd"
    # function to learn
    bop = "unit"
    # training hyperparams
    num_samples = 1000
    loss_type = "bce"
    lr = 0.01
    num_epochs = 100
    optimizer = 'sgd'
    # static architecture
    input_dim = 2
    output_dim = 1
    # dynamic architecture
    model_type = "binary"
    hidden_dim = 10

@dataclass
class UnitBCEPullConfig:
    name = "unitbcepull"
    # function to learn
    bop = "unit"
    # training hyperparams
    num_samples = 1000
    loss_type = "bce"
    lr = 0.01
    num_epochs = 100
    optimizer = 'ngd'
    # static architecture
    input_dim = 2
    output_dim = 1
    # dynamic architecture
    model_type = "binary"
    hidden_dim = 10

@dataclass
class UnitBCEAdamConfig:
    name = "unitbceadam"
    # function to learn
    bop = "unit"
    # training hyperparams
    num_samples = 1000
    loss_type = "bce"
    lr = 0.01
    num_epochs = 100
    optimizer = 'adam'
    # static architecture
    input_dim = 2
    output_dim = 1
    # dynamic architecture
    model_type = "binary"
    hidden_dim = 10

@dataclass
class UnitKLSGDConfig:
    name = "unitklsgd"
    # function to learn
    bop = "unit"
    # training hyperparams
    num_samples = 1000
    loss_type = "kl"
    lr = 0.01
    num_epochs = 100
    optimizer = 'sgd'
    # static architecture
    input_dim = 2
    output_dim = 1
    # dynamic architecture
    model_type = "binary"
    hidden_dim = 10

@dataclass
class UnitKLPullConfig:
    name = "unitklpull"
    # function to learn
    bop = "unit"
    # training hyperparams
    num_samples = 1000
    loss_type = "kl"
    lr = 0.01
    num_epochs = 100
    optimizer = 'ngd'
    # static architecture
    input_dim = 2
    output_dim = 1
    # dynamic architecture
    model_type = "binary"
    hidden_dim = 10

@dataclass
class UnitKLAdamConfig:
    name = "unitkladam"
    # function to learn
    bop = "unit"
    # training hyperparams
    num_samples = 1000
    loss_type = "kl"
    lr = 0.01
    num_epochs = 100
    optimizer = 'adam'
    # static architecture
    input_dim = 2
    output_dim = 1
    # dynamic architecture
    model_type = "binary"
    hidden_dim = 10