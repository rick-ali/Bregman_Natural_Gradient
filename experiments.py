from train import train
from configs import (
    MaxMSESGDConfig,
    MaxMSEAdamConfig,
    MaxMSEPullConfig,
    AddMSESGDConfig,
    AddMSEAdamConfig,
    AddMSEPullConfig,
    SubMSESGDConfig,
    SubMSEAdamConfig,
    SubMSEPullConfig,
    UnitBCESGDConfig,
    UnitBCEAdamConfig,
    UnitBCEPullConfig,
    UnitKLSGDConfig,
    UnitKLAdamConfig,
    UnitKLPullConfig,
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
        # MaxMSEPullConfig(),
        # AddMSESGDConfig(),
        # AddMSEAdamConfig(),
        # AddMSEPullConfig(),
        # SubMSESGDConfig(),
        # SubMSEAdamConfig(),
        # SubMSEPullConfig(),
        # UnitBCESGDConfig(),
        # UnitBCEPullConfig(),
        UnitKLSGDConfig(),
        UnitKLAdamConfig(),
        UnitKLPullConfig(),
    ]
    for config in configs:
        for seed in seeds:
            # Set random seed for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)
            logger = CSVLogger("logs", name="{}_seed_{}".format(config.name, seed))
            device = torch.device('cpu')
            train(config=config, device=device, logger=logger)


if __name__ == "__main__":
    experiment()
