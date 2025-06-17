#!/usr/bin/python3
from networks import resnetadapt
import sys

from training import TrainingContext

from torch.optim.lr_scheduler import StepLR, ExponentialLR, CyclicLR, CosineAnnealingLR, ReduceLROnPlateau
import torch.optim as optim

import paths
import utils


class ModelTraining(TrainingContext):
    def __init__(self, device):
        super().__init__(device, *utils.load_data(paths.DATA_PATH,
                                                  paths.TMPDIR), patience=10, epochs=500)

    def make_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=1e-5)

    def make_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=1e-5)


class ModelTraining100(TrainingContext):
    def __init__(self, device):
        super().__init__(device, *utils.load_data(paths.DATA_PATH,
                                                  paths.TMPDIR), patience=10, epochs=500)

    def make_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=1e-5)

    def make_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=1e-5)


CONFIGS = {
    "resnetadapt": [
        resnetadapt.Config()
        .set_training_context(ModelTraining),
        resnetadapt.Config()
        .set_training_context(ModelTraining100)
        .set_num_classes(100),
        resnetadapt.Config()
        .set_small_channels((6, 8, 10, 12, 14, 16))
        .set_mid_channels((12, 16, 20, 24, 28, 32))
        .set_large_channels((24, 32, 40, 48, 56, 64))
        .set_training_context(ModelTraining),
        resnetadapt.Config()
        .set_small_channels((6, 8, 10, 12, 14, 16))
        .set_mid_channels((12, 16, 20, 24, 28, 32))
        .set_large_channels((24, 32, 40, 48, 56, 64))
        .set_training_context(ModelTraining100)
        .set_num_classes(100),
        resnetadapt.Config()
        .set_num_blocks((9, 9, 9))
        .set_training_context(ModelTraining),
        resnetadapt.Config()
        .set_num_blocks((9, 9, 9))
        .set_training_context(ModelTraining100)
        .set_num_classes(100),
        resnetadapt.Config()
        .set_num_blocks((9, 9, 9))
        .set_small_channels((6, 8, 10, 12, 14, 16))
        .set_mid_channels((12, 16, 20, 24, 28, 32))
        .set_large_channels((24, 32, 40, 48, 56, 64))
        .set_training_context(ModelTraining),
        resnetadapt.Config()
        .set_num_blocks((9, 9, 9))
        .set_small_channels((6, 8, 10, 12, 14, 16))
        .set_mid_channels((12, 16, 20, 24, 28, 32))
        .set_large_channels((24, 32, 40, 48, 56, 64))
        .set_training_context(ModelTraining100)
        .set_num_classes(100),
    ]
}


def resolve_from_str(config):
    config = config.split(',')
    SUBPART = CONFIGS
    for i in config:
        try:
            i = int(i)
        except ValueError:
            pass
        SUBPART = SUBPART[i]
    return SUBPART


def iter_over_conf(conf, basestr):
    if isinstance(conf, dict):
        for key, val in conf.items():
            for s in iter_over_conf(val, basestr + f",{key}"):
                yield s
    if isinstance(conf, list):
        for idx, val in enumerate(conf):
            for s in iter_over_conf(val, basestr + f",{idx}"):
                yield s
    else:
        yield basestr


def print_all_conf_paths(conf, basestr, file=sys.stdout):
    for s in iter_over_conf(conf, basestr):
        print(s, file=file)


if __name__ == "__main__":
    command, conf = sys.argv[1:]
    # command = "run"
    # conf = "resnetadapt,0"
    res = resolve_from_str(conf)
    if command == "list":
        print_all_conf_paths(res, conf)
    elif command == "run":
        res.run_training()
