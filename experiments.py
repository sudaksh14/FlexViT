#!/usr/bin/python3
from networks import resnetadapt, vggadapt, resnet, vgg
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
    "resnetadapt": {
        'resnet20.3_levels.cifar10': resnetadapt.ResnetConfig()
        .set_training_context(ModelTraining),
        'resnet20.3_levels.cifar100': resnetadapt.ResnetConfig()
        .set_training_context(ModelTraining100)
        .set_num_classes(100),
        'resnet20.6_levels.cifar10': resnetadapt.ResnetConfig()
        .set_small_channels((6, 8, 10, 12, 14, 16))
        .set_mid_channels((12, 16, 20, 24, 28, 32))
        .set_large_channels((24, 32, 40, 48, 56, 64))
        .set_training_context(ModelTraining),
        'resnet20.6_levels.cifar100': resnetadapt.ResnetConfig()
        .set_small_channels((6, 8, 10, 12, 14, 16))
        .set_mid_channels((12, 16, 20, 24, 28, 32))
        .set_large_channels((24, 32, 40, 48, 56, 64))
        .set_training_context(ModelTraining100)
        .set_num_classes(100),
        'resnet56.3_levels.cifar10': resnetadapt.ResnetConfig()
        .set_num_blocks((9, 9, 9))
        .set_training_context(ModelTraining),
        'resnet56.3_levels.cifar100': resnetadapt.ResnetConfig()
        .set_num_blocks((9, 9, 9))
        .set_training_context(ModelTraining100)
        .set_num_classes(100),
        'resnet56.6_levels.cifar10': resnetadapt.ResnetConfig()
        .set_num_blocks((9, 9, 9))
        .set_small_channels((6, 8, 10, 12, 14, 16))
        .set_mid_channels((12, 16, 20, 24, 28, 32))
        .set_large_channels((24, 32, 40, 48, 56, 64))
        .set_training_context(ModelTraining),
        'resnet56.6_levels.cifar100': resnetadapt.ResnetConfig()
        .set_num_blocks((9, 9, 9))
        .set_small_channels((6, 8, 10, 12, 14, 16))
        .set_mid_channels((12, 16, 20, 24, 28, 32))
        .set_large_channels((24, 32, 40, 48, 56, 64))
        .set_training_context(ModelTraining100)
        .set_num_classes(100),
    },
    "vggadapt": {
        'vgg11.3_levels.cifar10': vggadapt.VGGConfig()
        .set_training_context(ModelTraining),
        'vgg11.3_levels.cifar100': vggadapt.VGGConfig()
        .set_training_context(ModelTraining100)
        .set_num_classes(100),

        'vgg11.6_levels.cifar10': vggadapt.VGGConfig()
        .set_training_context(ModelTraining)
        .set_small_channels((24, 32, 40, 48, 56, 64))
        .set_mid_channels((48, 64, 80, 96, 112, 128))
        .set_large_channels((96, 128, 160, 192, 224, 256))
        .set_max_channels((192, 256, 320, 384, 448, 512)),
        'vgg11.6_levels.cifar100': vggadapt.VGGConfig()
        .set_training_context(ModelTraining100)
        .set_num_classes(100)
        .set_small_channels((24, 32, 40, 48, 56, 64))
        .set_mid_channels((48, 64, 80, 96, 112, 128))
        .set_large_channels((96, 128, 160, 192, 224, 256))
        .set_max_channels((192, 256, 320, 384, 448, 512)),

        'vgg19.3_levels.cifar10': vggadapt.VGGConfig()
        .set_version(19)
        .set_training_context(ModelTraining),
        'vgg19.3_levels.cifar100': vggadapt.VGGConfig()
        .set_training_context(ModelTraining100)
        .set_num_classes(100),

        'vgg19.6_levels.cifar10': vggadapt.VGGConfig()
        .set_version(19)
        .set_training_context(ModelTraining)
        .set_small_channels((24, 32, 40, 48, 56, 64))
        .set_mid_channels((48, 64, 80, 96, 112, 128))
        .set_large_channels((96, 128, 160, 192, 224, 256))
        .set_max_channels((192, 256, 320, 384, 448, 512)),
        'vgg19.6_levels.cifar100': vggadapt.VGGConfig()
        .set_version(19)
        .set_training_context(ModelTraining100)
        .set_num_classes(100)
        .set_small_channels((24, 32, 40, 48, 56, 64))
        .set_mid_channels((48, 64, 80, 96, 112, 128))
        .set_large_channels((96, 128, 160, 192, 224, 256))
        .set_max_channels((192, 256, 320, 384, 448, 512)),
    },
    "baseline_model": {
        "resnet20_cifar10": resnet.ResnetConfig()
        .set_training_context(ModelTraining),
        "resnet56_cifar10": resnet.ResnetConfig()
        .set_training_context(ModelTraining)
        .set_num_blocks((9, 9, 9)),
        "vgg11_cifar10": vgg.VGGConfig()
        .set_training_context(ModelTraining)
        .set_version(19),
        "vgg19_cifar10": vgg.VGGConfig()
        .set_training_context(ModelTraining),
        "resnet20_cifar100": resnet.ResnetConfig()
        .set_training_context(ModelTraining100),
        "resnet56_cifar100": resnet.ResnetConfig()
        .set_training_context(ModelTraining100)
        .set_num_blocks((9, 9, 9)),
        "vgg11_cifar100": vgg.VGGConfig()
        .set_training_context(ModelTraining100),
        "vgg19_cifar100": vgg.VGGConfig()
        .set_training_context(ModelTraining100)
        .set_version(19),
    }
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
    elif isinstance(conf, list):
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
        res.run_training(conf)
