#!/usr/bin/python3
import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import wandb
import utils

import config.paths as paths
import run_experiment
import training


class Wandb:
    api: wandb.Api = None

    @staticmethod
    def get() -> wandb.Api:
        if __class__.api is None:
            __class__.api = wandb.Api()
        return __class__.api


def get_experiment(name: str):
    print(f"retrieving experiment '{name}'", file=sys.stderr)
    experiment: training.TrainerBuilder = run_experiment.resolve_from_str(name)
    if not isinstance(experiment, run_experiment.TrainerBuilder):
        return
    entity = Wandb.get().default_entity
    project = experiment.training_context.wandb_project_name
    runs = Wandb.get().runs(f"{entity}/{project}", order="-created_at")
    run = next(filter(lambda r: r.name == name, runs))
    return experiment, run


def plot_acc_history(name: str, stage: str = "val"):
    print(
        f"plotting acc history of experiment '{name}' as stage '{stage}'", file=sys.stderr)
    exp, run = get_experiment(name)

    for i in itertools.count(0):
        key = f"{stage}_level{i}_acc"
        if key not in run.summary.keys():
            break
        history = run.history(keys=["epoch", key], pandas=True)
        x = history["epoch"]
        y = history[key]

        reg_config = exp.model_config.create_base_config(i)
        model = reg_config.no_prebuilt().make_model()
        params = utils.count_parameters(model)

        plt.ylim(0.0, 1.0)
        plt.plot(x, y, label=f"{params // 100000 / 10}M parameters")

    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("top 1 accuracy")


def plot_acc(name: str, stage: str = "val"):
    print(
        f"plotting accuracies of experiment '{name}' at stage '{stage}'", file=sys.stderr)
    exp, run = get_experiment(name)
    size_to_acc = dict()

    for i in itertools.count(0):
        key = f"{stage}_level{i}_acc"
        if key not in run.summary.keys():
            break

        y = run.summary[key]

        reg_config = exp.model_config.create_base_config(i)
        model = reg_config.no_prebuilt().make_model()
        params = utils.count_parameters(model)

        size_to_acc[params / 1000000] = y

    _, _, test_loader = exp.training_context.loader_function()
    model = exp.model_config.create_base_config(
        exp.model_config.max_level()).make_model()

    if isinstance(exp.training_context, training.FlexTrainingContext):
        if exp.training_context.load_from is not None:
            lmodel = utils.load_model(
                exp.training_context.load_from.get_filename_safe_description(), "prebuild").submodel
            utils.flexible_model_copy(lmodel, model)

    acc = utils.evaluate_model(model, test_loader, utils.get_device())
    max_x = max(*size_to_acc.keys())
    min_x = min(*size_to_acc.keys())

    lines = plt.plot([min_x, max_x], [acc, acc], ls='--')
    plt.plot(size_to_acc.keys(), size_to_acc.values(),
             marker='o', color=lines[0].get_color())
    plt.xlabel("model size in millions of parameters")
    plt.ylabel("top 1 accuracy")
    plt.ylim(0.0, 1.0)


def moving_avg(axis, data, size=10):
    data = np.array(data)
    mavg = np.convolve(data, np.ones(
        (size,), dtype=data.dtype) / size, 'valid')
    axis = np.array(axis)
    axis = axis[size // 2: -(size // 2) + ((size + 1) % 2)]
    return axis, mavg


def plot_acc_val_and_train(name, level):
    print(
        f"plotting validation and train accuracies of '{name}' at level {level}", file=sys.stderr)

    exp, run = get_experiment(name)

    train_key = f"train_level{level}_acc"
    val_key = f"val_level{level}_acc"

    train = run.history(keys=["epoch", train_key], pandas=True)
    val = run.history(keys=["epoch", val_key], pandas=True)

    train_lines = plt.plot(
        train['epoch'], train[train_key], ls='--', alpha=0.5, label="training accuracy")
    val_lines = plt.plot(val['epoch'], val[val_key],
                         ls='--', alpha=0.5, label="validation accuracy")

    axs, mavg = moving_avg(train['epoch'], train[train_key], 20)
    plt.plot(axs, mavg, color=train_lines[0].get_color())

    axs, mavg = moving_avg(val['epoch'], val[val_key], 20)
    plt.plot(axs, mavg, val_lines[0].get_color())

    plt.xlabel("epochs")
    plt.ylabel("top 1 accuracy")
    plt.legend()


def savefig(name: str):
    print(f"saving figure '{name}'")
    plt.savefig(paths.FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":
    matplotlib.rc("font", size=13)

    plot_acc_history("flexvit,imagenet")
    savefig("vit_imagenet_history")

    plot_acc("flexvit,imagenet", "test")
    savefig("vit_imagenet_acc")

    plot_acc_val_and_train("flexvit,imagenet", 4)
    savefig("overfitting")

    plot_acc_history("flexvit,cifar10.5levels")
    savefig("cifar10_acc_history")

    plot_acc("flexvit,cifar10.5levels", "test")
    savefig("cifar10_acc")
