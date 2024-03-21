import pandas as pd
import seaborn as sns
from os import listdir
import os
import click
import matplotlib.pyplot as plt
import numpy as np

@click.command()
@click.option("--path", default="logs/", help="Folder directory.")
@click.option("--name", default="maxmse", help="Config name.")
@click.option(
    "--version",
    default=0,
    help="Version number (results are same across versions for fixed seed).",
)
def plot_reg_pull_diff(path, name, version):
    folders = listdir(path)
    data = None
    hue_order = set()
    losses = {'sgd': [],
                'bgd': [],
                'adam': []}
    if 'unit' in name:
        losses['ngd'] = []
    for folder in folders:
        if 'train' in folder:
            continue
        if 'p2' in folder:
            continue
        if name in folder:
            task = ''
            tasks = ['max', 'sub', 'add', 'unit']
            for task_ in tasks:
                if task_ in name:
                    task = task_
            _, seed = folder.split("_seed_")
            seed = int(seed)
            data_path =  path + folder + "/" + "version_{}".format(version) + "/" + "metrics.csv"
            if not os.path.isfile(data_path):
                continue
            if "kl" in folder:
                df = pd.read_csv(data_path)["accuracy"]
            else:
                df = pd.read_csv(data_path)["loss"]
            
            if "sgd" in folder:
                losses['sgd'].append(df[0])
            elif "ngd" in folder:
               losses['ngd'].append(df[0])
            elif "bgd" in folder:
               losses['bgd'].append(df[0])
            elif "adam" in folder:
               losses['adam'].append(df[0])
    
    for method in losses:
        vals = np.array(losses[method])
        print(f'method: {method} mean {np.mean(vals):.4}+{np.std(vals):.4}')


if __name__ == "__main__":
    plot_reg_pull_diff()
