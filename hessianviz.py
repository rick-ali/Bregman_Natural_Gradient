import pandas as pd
import seaborn as sns
from os import listdir
import os
import click
import matplotlib.pyplot as plt
import torch

@click.command()
@click.option("--path", default="logs/", help="Folder directory.")
@click.option("--name", default="maxmse", help="Config name.")
@click.option("--step", default=0, help="Step.")
@click.option(
    "--version",
    default=0,
    help="Version number (results are same across versions for fixed seed).",
)
def plot_reg_pull_diff(path, name, step, version):
    pd.set_option('display.max_colwidth', None)
    folders = listdir(path)
    data = None
    for folder in folders:
        if 'test' in folder:
            continue
        if name in folder:
            _, seed = folder.split("_seed_")
            seed = int(seed)
            data_path =  path + folder + "/metric/" + "version_{}".format(version) + "/" + "metrics.csv"
            if not os.path.isfile(data_path):
                continue
            df = pd.read_csv(data_path)
            df = df[df['step'] == step]
            if step >= 10:
                hessian = str(df['metric'])[8:-30]
            else:
                hessian = str(df['metric'])[7:-30]
            hessian = hessian.split('], [')
            for i in range(len(hessian)):
                hessian[i] = [float(el) for el in hessian[i].split(', ')]
            hessian = torch.tensor(hessian)
            sns.heatmap(hessian)
            plt.show()

    # sns.set_theme(style="darkgrid")
    # sns.lineplot(x="epoch", y="loss", hue="grad", data=data).set_title(f'{name}')
    # plt.show()


if __name__ == "__main__":
    plot_reg_pull_diff()
