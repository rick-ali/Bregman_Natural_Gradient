from train import train, test
from configs import (
    MaxMSESGDConfig,
    MaxMSEAdamConfig,
    MaxMSEBGDConfig,
    MaxMSENGDConfig,

    AddMSESGDConfig,
    AddMSEAdamConfig,
    AddMSEBGDConfig,

    SubMSESGDConfig,
    SubMSEAdamConfig,
    SubMSEBGDConfig,

    # UnitBCESGDConfig,
    # UnitBCEAdamConfig,
    # UnitBCEPullConfig,
    UnitKLSGDConfig,
    UnitKLAdamConfig,
    UnitKLNGDConfig,
    UnitKLBGDConfig,
    UnitKLP2Config
)
from lightning.pytorch.loggers import CSVLogger
from accelerate import Accelerator
import torch
import numpy as np


def experiment():
    seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    configs = [
        # MaxMSESGDConfig(),
        # MaxMSEAdamConfig(),
        # MaxMSEBGDConfig(),
        # AddMSESGDConfig(),
        # AddMSEAdamConfig(),
        # AddMSEBGDConfig(),
        # SubMSESGDConfig(),
        # SubMSEAdamConfig(),
        # SubMSEBGDConfig(),
        # UnitKLSGDConfig(),
        # UnitKLAdamConfig(),
        # UnitKLBGDConfig(),
        UnitKLP2Config(),
        # UnitKLNGDConfig(),
    ]
    for config in configs:
        for seed in seeds:
            # Set random seed for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)
            train_logger = CSVLogger("logs", name="{}_train_seed_{}".format(config.name, seed))
            # metric_logger = CSVLogger("logs", name="{}_train_seed_{}/metric".format(config.name, seed))
            device = torch.device('cpu')
            # model = train(config=config, device=device, logger=train_logger, log_G=True, G_logger=metric_logger)
            model, X = train(config=config, device=device, logger=train_logger)
            test_logger = CSVLogger("logs", name="{}_test_seed_{}".format(config.name, seed))
            test(config=config, device=device, model=model, logger=test_logger)


if __name__ == "__main__":
    experiment()
