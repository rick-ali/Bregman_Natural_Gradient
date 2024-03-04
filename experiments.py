from train import train 
from configs import ExperimentConfig_1
from lightning.pytorch.loggers import CSVLogger

def experiment():
    seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    configs = [ExperimentConfig_1()]
    for config in configs:
        for seed in seeds:
            logger = CSVLogger("logs", name="{}_seed_{}_regular".format(config.name, seed))
            train(bop=config.bop,
                  num_samples=config.num_samples,
                  model_type=config.model_type,
                  loss_type=config.loss_type,
                  num_epochs=config.num_epochs,
                  pullback=False,
                  lr=config.lr,
                  logger=logger)
            
    for config in configs:
        for seed in seeds:
            logger = CSVLogger("logs", name="{}_seed_{}_pullback".format(config.name, seed))
            train(bop=config.bop,
                  num_samples=config.num_samples,
                  model_type=config.model_type,
                  loss_type=config.loss_type,
                  num_epochs=config.num_epochs,
                  pullback=True,
                  lr=config.lr,
                  logger=logger)

if __name__ == '__main__':
    experiment()