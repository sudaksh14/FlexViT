#!/usr/bin/python3
from networks import resnetadapt, vggadapt, resnet, vgg
import sys

from training import TrainingContext, AdaptiveTrainingContext, AdaptiveModelTrainer, BaseTrainer, ZeroOutTrainer, make_zero_grad_optimizer, ZeroOutTrainingContext

from torch.optim.lr_scheduler import StepLR, ExponentialLR, CyclicLR, CosineAnnealingLR, ReduceLROnPlateau
import torch.optim as optim

import paths
import utils

from typing import Callable, Generator


class ModelTraining(AdaptiveTrainingContext):
    def __init__(self):
        super().__init__(utils.load_data, patience=20, epochs=-1)

    def make_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=1e-5)

    def make_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=1e-5)


class ModelTraining100(AdaptiveTrainingContext):
    def __init__(self):
        super().__init__(utils.load_data100, patience=20, epochs=-1)

    def make_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=1e-5)

    def make_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=1e-5)


class ModelTraining100ZeroOut(ZeroOutTrainingContext):
    def __init__(self):
        super().__init__(utils.load_data100, patience=20, epochs=-1)

    def make_optimizer(self, model):
        return make_zero_grad_optimizer(optim.Adam, model, self.zero_out_level, model.parameters(), lr=1e-5)

    def make_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=1e-5)


CONFIGS = {
    "resnetadapt": {
        'resnet20.3_levels.cifar10': lambda: AdaptiveModelTrainer(
            resnetadapt.ResnetConfig(), ModelTraining()),
        'resnet20.3_levels.cifar100': lambda: AdaptiveModelTrainer(
            resnetadapt.ResnetConfig()
            .set_num_classes(100), ModelTraining100()),

        'resnet20.6_levels.cifar10': lambda: AdaptiveModelTrainer(
            resnetadapt.ResnetConfig()
            .set_small_channels((6, 8, 10, 12, 14, 16))
            .set_mid_channels((12, 16, 20, 24, 28, 32))
            .set_large_channels((24, 32, 40, 48, 56, 64)), ModelTraining()),
        'resnet20.6_levels.cifar100': lambda: AdaptiveModelTrainer(
            resnetadapt.ResnetConfig()
            .set_small_channels((6, 8, 10, 12, 14, 16))
            .set_mid_channels((12, 16, 20, 24, 28, 32))
            .set_large_channels((24, 32, 40, 48, 56, 64))
            .set_num_classes(100), ModelTraining100()),

        'resnet56.3_levels.cifar10': lambda: AdaptiveModelTrainer(
            resnetadapt.ResnetConfig()
            .set_num_blocks((9, 9, 9)), ModelTraining()),
        'resnet56.3_levels.cifar100': lambda: AdaptiveModelTrainer(
            resnetadapt.ResnetConfig()
            .set_num_blocks((9, 9, 9))
            .set_num_classes(100), ModelTraining100()),

        'resnet56.6_levels.cifar10': lambda: AdaptiveModelTrainer(
            resnetadapt.ResnetConfig()
            .set_num_blocks((9, 9, 9))
            .set_small_channels((6, 8, 10, 12, 14, 16))
            .set_mid_channels((12, 16, 20, 24, 28, 32))
            .set_large_channels((24, 32, 40, 48, 56, 64)), ModelTraining()),
        'resnet56.6_levels.cifar100': lambda: AdaptiveModelTrainer(
            resnetadapt.ResnetConfig()
            .set_num_blocks((9, 9, 9))
            .set_small_channels((6, 8, 10, 12, 14, 16))
            .set_mid_channels((12, 16, 20, 24, 28, 32))
            .set_large_channels((24, 32, 40, 48, 56, 64))
            .set_num_classes(100), ModelTraining100()),
    },
    "vggadapt": {
        'vgg11.3_levels.cifar10': lambda: AdaptiveModelTrainer(
            vggadapt.VGGConfig(), ModelTraining()),
        'vgg11.3_levels.cifar100': lambda: AdaptiveModelTrainer(
            vggadapt.VGGConfig()
            .set_num_classes(100), ModelTraining100()),

        'vgg11.6_levels.cifar10': lambda: AdaptiveModelTrainer(
            vggadapt.VGGConfig()
            .set_small_channels((24, 32, 40, 48, 56, 64))
            .set_mid_channels((48, 64, 80, 96, 112, 128))
            .set_large_channels((96, 128, 160, 192, 224, 256))
            .set_max_channels((192, 256, 320, 384, 448, 512)), ModelTraining()),
        'vgg11.6_levels.cifar100': lambda: AdaptiveModelTrainer(
            vggadapt.VGGConfig()
            .set_num_classes(100)
            .set_small_channels((24, 32, 40, 48, 56, 64))
            .set_mid_channels((48, 64, 80, 96, 112, 128))
            .set_large_channels((96, 128, 160, 192, 224, 256))
            .set_max_channels((192, 256, 320, 384, 448, 512)), ModelTraining100()),

        'vgg19.3_levels.cifar10': lambda: AdaptiveModelTrainer(
            vggadapt.VGGConfig()
            .set_version(19), ModelTraining()),
        'vgg19.3_levels.cifar100': lambda: AdaptiveModelTrainer(
            vggadapt.VGGConfig()
            .set_num_classes(100), ModelTraining100()),

        'vgg19.6_levels.cifar10': lambda: AdaptiveModelTrainer(
            vggadapt.VGGConfig()
            .set_version(19)
            .set_small_channels((24, 32, 40, 48, 56, 64))
            .set_mid_channels((48, 64, 80, 96, 112, 128))
            .set_large_channels((96, 128, 160, 192, 224, 256))
            .set_max_channels((192, 256, 320, 384, 448, 512)), ModelTraining()),
        'vgg19.6_levels.cifar100': lambda: AdaptiveModelTrainer(
            vggadapt.VGGConfig()
            .set_version(19)
            .set_num_classes(100)
            .set_small_channels((24, 32, 40, 48, 56, 64))
            .set_mid_channels((48, 64, 80, 96, 112, 128))
            .set_large_channels((96, 128, 160, 192, 224, 256))
            .set_max_channels((192, 256, 320, 384, 448, 512)), ModelTraining100()),
    },
    "incremental": {
        "incr.resnet20.3_levels.cifar100": lambda: AdaptiveModelTrainer(
            resnetadapt.ResnetConfig()
            .set_num_classes(100), ModelTraining100().set_incremental_training(True)),
        'incr.vgg11.3_levels.cifar100': lambda: AdaptiveModelTrainer(
            vggadapt.VGGConfig()
            .set_num_classes(100), ModelTraining100().set_incremental_training(True)),
        'incr.vgg11.6_levels.cifar100': lambda: AdaptiveModelTrainer(
            vggadapt.VGGConfig()
            .set_num_classes(100)
            .set_small_channels((24, 32, 40, 48, 56, 64))
            .set_mid_channels((48, 64, 80, 96, 112, 128))
            .set_large_channels((96, 128, 160, 192, 224, 256))
            .set_max_channels((192, 256, 320, 384, 448, 512)), ModelTraining100().set_incremental_training(True)),
    },
    "upscale": {
        "upscale.vgg11.cifar100": lambda: AdaptiveModelTrainer(
            vggadapt.VGGConfig()
            .set_small_channels((64, 96, 128))
            .set_mid_channels((128, 192, 256))
            .set_large_channels((256, 384, 512))
            .set_max_channels((512, 768, 1024))
            .set_num_classes(100)
            .set_prebuilt_level(0), ModelTraining100().set_incremental_training(True)),
    }, "zeroout": {
        'vgg11.3_levels.cifar100': lambda: ZeroOutTrainer(
            vggadapt.VGGConfig()
            .set_num_classes(100), ModelTraining100ZeroOut()),
        'resnet20.6_levels.cifar100': lambda: ZeroOutTrainer(
            resnetadapt.ResnetConfig()
            .set_small_channels((6, 8, 10, 12, 14, 16))
            .set_mid_channels((12, 16, 20, 24, 28, 32))
            .set_large_channels((24, 32, 40, 48, 56, 64))
            .set_num_classes(100), ModelTraining100ZeroOut()),
        'resnet20.3_levels.cifar100': lambda: ZeroOutTrainer(
            resnetadapt.ResnetConfig()
            .set_num_classes(100), ModelTraining100ZeroOut()),
    }
}


def resolve_from_str(config) -> Callable[[], BaseTrainer]:
    config = config.split(',')
    SUBPART = CONFIGS
    for i in config:
        if i == 'all':
            continue
        try:
            i = int(i)
        except ValueError:
            pass
        SUBPART = SUBPART[i]
    return SUBPART


def iter_over_conf(conf, basestr) -> Generator[str, None, None]:
    if isinstance(conf, dict):
        for key, val in conf.items():
            for s in iter_over_conf(val, basestr + f",{key}"):
                yield s
    elif isinstance(conf, list):
        for idx, val in enumerate(conf):
            for s in iter_over_conf(val, basestr + f",{idx}"):
                yield s
    else:
        yield basestr


def print_all_conf_paths(conf, basestr, file=sys.stdout) -> None:
    for s in iter_over_conf(conf, basestr):
        print(s, file=file)


if __name__ == "__main__":
    command, conf = sys.argv[1:]
    # command = "run"
    # conf = "zeroout,vgg11.3_levels.cifar100"
    res = resolve_from_str(conf)
    if command == "list":
        print_all_conf_paths(res, conf)
    elif command == "run":
        res().run_training(conf)
