import io
import os
from typing import List, Optional

import seaborn as sns
from matplotlib import pyplot as plt
from PIL import Image

import wandb
from loggers.exp_logger import ExperimentLogger

os.environ['WANDB_START_METHOD'] = 'thread'


class Logger(ExperimentLogger):
    """Characterizes a Weights and Biases wandb logger
    Assumes prior wandb login (wandb login)"""

    def __init__(
            self,
            exp_path: str,
            exp_name: Optional[str] = None,
            tags: Optional[List[str]] = None,
    ):
        super(Logger, self).__init__(exp_path, exp_name)

        wandb.init(group=exp_name, tags=tags)
        self.metrics = []

    def log_scalar(self, task, iter, name, value, group=None, curtime=None):
        if task is not None:
            key = "{}_{}/t_{}".format(group, name, task)
        else:
            key = "{}/{}".format(group, name)

        iter_key = "_iter/" + key.replace("/", "_")
        if key not in self.metrics:
            wandb.define_metric(iter_key, hidden=True)
            wandb.define_metric(key, step_metric=iter_key)
            self.metrics.append(key)
        wandb.log({key: value, iter_key: iter})

    def log_args(self, args):
        wandb.config.update(args.__dict__)

    def log_result(self, array, name, step, **kwargs):
        if array.ndim <= 1:
            array = array[None]

        plt.cla()
        plt.clf()
        plt.figure(dpi=300)

        if "cbar" in kwargs:
            cbar = kwargs["cbar"]
        else:
            cbar = False
        if "vmin" in kwargs:
            vmin = kwargs["vmin"]
        else:
            vmin = 0

        if "vmax" in kwargs:
            vmax = kwargs["vmax"]
        else:
            vmax = 1

        if "annot" in kwargs:
            annot = kwargs["annot"]
        else:
            annot = True

        if "cmap" in kwargs:
            cmap = kwargs["cmap"]
        else:
            cmap = None

        plot = sns.heatmap(array, annot=annot, cmap=cmap, cbar=cbar, vmin=vmin, vmax=vmax)

        if "title" in kwargs:
            plot.set_title(kwargs["title"])
        if "xlabel" in kwargs:
            plot.set_xlabel(kwargs["xlabel"])
        if "ylabel" in kwargs:
            plot.set_ylabel(kwargs["ylabel"])

        wandb.log({name: _plot_to_wandb(plot)})

    def log_figure(self, name, iter, figure, curtime=None):
        wandb.log({name: _plot_to_wandb(figure)})

    def __del__(self):
        wandb.finish()


def _plot_to_wandb(plot):
    buffer = io.BytesIO()
    plot.get_figure().savefig(buffer)

    with Image.open(buffer) as img:
        img = wandb.Image(img)

    return img
