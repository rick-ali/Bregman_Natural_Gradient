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
    hue_order = set()
    for folder in folders:
        if 'test' in folder:
            continue
        if name in folder:
            if 'p2' in folder:
                continue
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
            df = pd.read_csv(data_path)[["epoch", "loss"]]
            if "sgd" in folder:
                hue_order.add('sgd')
                df = df.assign(grad="sgd")
            elif "ngd" in folder:
                hue_order.add('ngd')
                df = df.assign(grad="fim")
            elif "bgd" in folder:
                hue_order.add('bgd')
                df = df.assign(grad="bgd")
            elif "adam" in folder:
                hue_order.add('adam')
                df = df.assign(grad="adam")
            elif "p2" in folder:
                hue_order.add('p^2')
                df = df.assign(grad="p^2")
            if data is None:
                data = df
            else:
                data = pd.concat([data, df])

    sns.set_theme(style="darkgrid")
    hue_order = sorted(list(hue_order))
    if 'ngd' in hue_order:
        hue_order = ["adam", "bgd", "sgd", "ngd"]
    sns.lineplot(x="epoch", y="loss", hue="grad", hue_order=hue_order, data=data).set_title(f'{task}')
    plt.show()


if __name__ == "__main__":
    plot_reg_pull_diff()
