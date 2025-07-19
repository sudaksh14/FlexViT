import matplotlib.axis
import matplotlib.figure
import wandb
import re
import experiments

import matplotlib.pyplot as plt

import networks
import numpy as np
import networks.config
import utils

import itertools

import matplotlib

api = wandb.Api()


def plot_acc_history(path: str, stage: str = "val"):
    run = api.run(path)

    for i in itertools.count(0):
        key = f"{stage}_level{i}_acc"
        if key not in run.summary.keys():
            break
        history = run.history(keys=["epoch", key])
        x = history["epoch"]
        y = history[key]

        config = experiments.resolve_from_str(run.name)
        conf = config.model_config.create_base_config(i)
        model = conf.no_prebuilt().make_model()
        params = utils.count_parameters(model)

        plt.ylim(0.0, 1.0)
        plt.plot(x, y, label=f"{params // 100000 / 10}M parameters")

    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("top 1 accuracy")
    plt.show()


def plot_acc(path: str, stage: str = "val"):
    run = api.run(path)
    size_to_acc = dict()

    for i in itertools.count(0):
        key = f"{stage}_level{i}_acc"
        if key not in run.summary.keys():
            break

        y = run.summary[key]

        config = experiments.resolve_from_str(run.name)
        conf = config.model_config.create_base_config(i)
        model = conf.no_prebuilt().make_model()
        params = utils.count_parameters(model)

        size_to_acc[params / 1000000] = y

    plt.plot(size_to_acc.keys(), size_to_acc.values(), marker='o')
    plt.xlabel("model size in millions of parameters")
    plt.ylabel("top 1 accuracy")
    plt.ylim(0.0, 1.0)
    plt.show()


def moving_avg(axis, data, size=10):
    data = np.array(data)
    mavg = np.convolve(data, np.ones((size,), dtype=data.dtype) / size, 'valid')
    axis = np.array(axis)
    axis = axis[size // 2: -(size // 2) + ((size + 1) % 2)]
    return axis, mavg


def plot_acc_val_and_train(path, level):
    run = api.run(path)
    size_to_acc = dict()

    train_key = f"train_level{level}_acc"
    val_key = f"val_level{level}_acc"

    train = run.history(keys=["epoch", train_key], pandas=True)
    val = run.history(keys=["epoch", val_key], pandas=True)


    train_lines = plt.plot(train['epoch'], train[train_key], ls='--', alpha=0.5, label="training accuracy")
    val_lines = plt.plot(val['epoch'], val[val_key], ls='--', alpha=0.5, label="validation accuracy")
    
    axs, mavg = moving_avg(train['epoch'], train[train_key], 20)
    plt.plot(axs, mavg, color=train_lines[0].get_color())

    axs, mavg = moving_avg(val['epoch'], val[val_key], 20)
    plt.plot(axs, mavg, val_lines[0].get_color())

    plt.xlabel("epochs")
    plt.ylabel("top 1 accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    matplotlib.rc("font", size=20)
    # plot_acc_history("robbieman4-university-of-amsterdam/b/runs/hgwwsqy5")
    # plot_acc("robbieman4-university-of-amsterdam/b/runs/hgwwsqy5", "test")
    # plot_acc_val_and_train(
    #     "robbieman4-university-of-amsterdam/b/runs/hgwwsqy5", 4)
    
    plot_acc_history("robbieman4-university-of-amsterdam/b/runs/t5ggk71u")
    plot_acc("robbieman4-university-of-amsterdam/b/runs/t5ggk71u", "test")
