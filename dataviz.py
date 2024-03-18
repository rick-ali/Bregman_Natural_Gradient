import pandas as pd
import seaborn as sns
from os import listdir
import os
import click
import matplotlib.pyplot as plt


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
    for folder in folders:
        if name in folder:
            _, seed = folder.split("_seed_")
            seed = int(seed)
            data_path =  path + folder + "/" + "version_{}".format(version) + "/" + "metrics.csv"
            if not os.path.isfile(data_path):
                continue
            df = pd.read_csv(data_path)[["epoch", "loss"]]
            if "sgd" in folder:
                df = df.assign(grad="sgd")
            elif "ngd" in folder:
                df = df.assign(grad="ngd")
            elif "bgd" in folder:
                df = df.assign(grad="bgd (ours)")
            elif "adam" in folder:
                df = df.assign(grad="adam")
            if data is None:
                data = df
            else:
                data = pd.concat([data, df])

    sns.set_theme(style="darkgrid")
    sns.lineplot(x="epoch", y="loss", hue="grad", data=data)
    plt.show()


if __name__ == "__main__":
    plot_reg_pull_diff()
