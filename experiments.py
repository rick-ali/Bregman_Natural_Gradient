from train import train
from configs import (
    MaxMSERegConfig,
    MaxMSEPullConfig,
    AddMSERegConfig,
    AddMSEPullConfig,
    SubMSERegConfig,
    SubMSEPullConfig,
    UnitBCERegConfig,
    UnitBCEPullConfig,
    UnitKLRegConfig,
    UnitKLPullConfig,
)
from lightning.pytorch.loggers import CSVLogger
from accelerate import Accelerator
import torch
import numpy as np


def experiment():
    seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    accelerator = Accelerator()
    configs = [
        MaxMSERegConfig(),
        MaxMSEPullConfig(),
        AddMSERegConfig(),
        AddMSEPullConfig(),
        SubMSERegConfig(),
        SubMSEPullConfig(),
        # UnitBCERegConfig(),
        # UnitBCEPullConfig(),
        # UnitKLRegConfig(),
        # UnitKLPullConfig(),
    ]
    for config in configs:
        for seed in seeds:
            # Set random seed for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)
            logger = CSVLogger("logs", name="{}_seed_{}".format(config.name, seed))
            train(config=config, logger=logger, accelerator=accelerator)


if __name__ == "__main__":
    experiment()
