import wandb
import re

import experiments

import matplotlib.pyplot as plt

import training

from networks import vggadapt, resnetadapt
import numpy as np
import utils

# Initialize API and target project
api = wandb.Api()

# Get all runs from the project
runs = api.runs("robbieman4-university-of-amsterdam/a")

# Pattern to match 'test_levelX' keys
pattern = re.compile(r"test_level(\d+)_acc")


fig = plt.figure()
ax10 = plt.subplot(141)
ax100 = plt.subplot(142)
ax10res = plt.subplot(143)
ax100res = plt.subplot(144)

sizes = np.array([6, 8, 10, 12, 14, 16])
sizes = sizes * sizes
sizes = sizes / sizes.max()

_, _, test_loader = utils.load_data()
_, _, test_loader100 = utils.load_data100()

for run in runs:
    # You can also use run.history() to get intermediate steps,
    # but run.summary is faster and often contains the last logged values.
    levels = []
    values = []

    summary = run.summary
    for key in summary.keys():
        match = pattern.match(key)
        if match:
            level = int(match.group(1))
            value = summary[key]
            levels.append(level)
            values.append(value)

    if len(levels) == 3:
        levels = [1, 3, 5]
    levels = sizes[levels]

    # if len(levels) == 6:
    #     levels = levels[1:]
    #     values = values[1:]

    model = experiments.resolve_from_str(run.name)
    model = model()
    model: training.AdaptiveModelTrainer

    label = run.name.split(',')[-1]

    if model.model_config.num_classes == 10:
        if isinstance(model.model_config, vggadapt.VGGConfig):
            line, *_ = ax10.plot(levels, values, label=label)
            acc = utils.evaluate_model(model, test_loader, utils.get_device())
            ax10.plot([0,1],[acc, acc], color=line.get_color(), ls='--')
        else:
            line, *_ = ax10res.plot(levels, values, label=label)
            acc = utils.evaluate_model(model, test_loader, utils.get_device())
            ax10res.plot([0,1],[acc, acc], color=line.get_color(), ls='--')
    else:
        if isinstance(model.model_config, vggadapt.VGGConfig):
            line, *_ = ax100.plot(levels, values, label=label)
            acc = utils.evaluate_model(model, test_loader100, utils.get_device())
            ax100.plot([0,1],[acc, acc], color=line.get_color(), ls='--')
        else:
            line, *_ = ax100res.plot(levels, values, label=label)
            acc = utils.evaluate_model(model, test_loader100, utils.get_device())
            ax100res.plot([0,1],[acc, acc], color=line.get_color(), ls='--')

ax10.legend()
ax10.set_title('vgg 10')
ax10.set_ylim(0, 1.0)
ax10.set_xlim(0, 1)


ax10res.legend()
ax10res.set_title('resnet 10')
ax10res.set_ylim(0, 1.0)
ax10res.set_xlim(0, 1)


ax100.legend()
ax100.set_title('vgg 100')
ax100.set_ylim(0, 1.0)
ax100.set_xlim(0, 1)


ax100res.legend()
ax100res.set_title('resnet 100')
ax100res.set_ylim(0, 1.0)
ax100res.set_xlim(0, 1)

plt.show()
