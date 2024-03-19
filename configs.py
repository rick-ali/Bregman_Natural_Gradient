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
class MaxMSEBGDConfig:
    name = "maxmsebgd"
    # function to learn
    bop = "max"
    # training hyperparams
    num_samples = 1000
    loss_type = "mse"
    lr = 0.01
    num_epochs = 100
    optimizer = 'bgd'
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
class AddMSEBGDConfig:
    name = "addmsebgd"
    # function to learn
    bop = "add"
    # training hyperparams
    num_samples = 1000
    loss_type = "mse"
    lr = 0.01
    num_epochs = 100
    optimizer = 'bgd'
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
class SubMSEBGDConfig:
    name = "submsebgd"
    # function to learn
    bop = "sub"
    # training hyperparams
    num_samples = 1000
    loss_type = "mse"
    lr = 0.01
    num_epochs = 100
    optimizer = 'bgd'
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
class UnitKLNGDConfig:
    name = "unitklngd"
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

@dataclass
class UnitKLBGDConfig:
    name = "unitklbgd"
    # function to learn
    bop = "unit"
    # training hyperparams
    num_samples = 1000
    loss_type = "kl"
    lr = 0.01
    num_epochs = 100
    optimizer = 'bgd'
    # static architecture
    input_dim = 2
    output_dim = 1
    # dynamic architecture
    model_type = "binary"
    hidden_dim = 10

@dataclass
class BinAdd8MSEBGDConfig:
    name = "binaddmsebgd"
    # function to learn
    bop = "binadd"
    # training hyperparams
    num_samples = 100
    loss_type = "mse"
    lr = 0.01
    num_epochs = 500
    optimizer = 'bgd'
    # static architecture
    input_dim = 8
    output_dim = 1
    # dynamic architecture
    model_type = "simple"
    hidden_dim = 10    

@dataclass
class BinAdd8MSESGDConfig:
    name = "binadd8msesgd"
    # function to learn
    bop = "binadd"
    # training hyperparams
    num_samples = 100
    loss_type = "mse"
    lr = 0.01
    num_epochs = 80000
    optimizer = 'sgd'
    # static architecture
    input_dim = 8
    output_dim = 1
    # dynamic architecture
    model_type = "simple"
    hidden_dim = 10      

@dataclass
class BinAdd8KLBGDConfig:
    name = "binaddklbgd"
    # function to learn
    bop = "binadd"
    # training hyperparams
    num_samples = 100
    loss_type = "kl"
    lr = 0.01
    num_epochs = 1000
    optimizer = 'bgd'
    # static architecture
    input_dim = 8
    output_dim = 1
    # dynamic architecture
    model_type = "simple"
    hidden_dim = 10    

@dataclass
class BinAdd8KLNGDConfig:
    name = "binadd8klngd"
    # function to learn
    bop = "binadd"
    # training hyperparams
    num_samples = 100
    loss_type = "kl"
    lr = 0.01
    num_epochs = 1000
    optimizer = 'ngd'
    # static architecture
    input_dim = 8
    output_dim = 1
    # dynamic architecture
    model_type = "simple"
    hidden_dim = 10      