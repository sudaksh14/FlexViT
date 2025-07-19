#!/usr/bin/python3
import hardware
from torchvision.datasets import CIFAR10, CIFAR100
from functools import partial
from typing import Callable, Generator
import utils
import paths
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CyclicLR, CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from networks import resnetadapt, vggadapt, resnet, vgg, vit, vitadapt
import sys

from training import *


class ModelTraining(AdaptiveTrainingContext):
    def __init__(self, *args, **kwargs):
        super().__init__(partial(utils.load_data, CIFAR10,
                                 *args, **kwargs), patience=50, epochs=-1)

    def make_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=1e-5)

    def make_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=self.epochs)


class ModelTraining100(AdaptiveTrainingContext):
    def __init__(self, *args, **kwargs):
        super().__init__(partial(utils.load_data, CIFAR100,
                                 *args, **kwargs), patience=50, epochs=-1)

    def make_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=1e-5)

    def make_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=self.epochs)


class ViTTraining(AdaptiveTrainingContext):
    def __init__(self, *args, **kwargs):
        super().__init__(partial(utils.load_data, CIFAR10,
                                 resize=(224, 224)), patience=20, epochs=-1)

    def make_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=1e-5)

    def make_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=self.epochs)


class ViTTraining100(AdaptiveTrainingContext):
    def __init__(self, *args, **kwargs):
        super().__init__(partial(utils.load_data, CIFAR100,
                                 resize=(224, 224)), patience=20, epochs=-1)

    def make_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=1e-5)

    def make_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=self.epochs)


class VitTrainingImagenet(AdaptiveTrainingContext):
    def __init__(self, *args, **kwargs):
        super().__init__(utils.load_imagenet, patience=50, epochs=300,
                         label_smoothing=0.11, gradient_clip_val=1.0)

    def make_optimizer(self, model):
        return optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.3)

    def make_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=0.0)


class VitTrainingImagenetWarmup(AdaptiveTrainingContext):
    warmup_epochs: int = 30

    def __init__(self, *args, **kwargs):
        super().__init__(utils.load_imagenet, patience=50, epochs=300,
                         label_smoothing=0.11, gradient_clip_val=1.0)

    def make_optimizer(self, model):
        return optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.3)

    def make_scheduler(self, optimizer):
        return SequentialLR(optimizer, [
            LinearLR(optimizer, start_factor=0.033,
                     total_iters=self.warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=self.epochs -
                              self.warmup_epochs, eta_min=0.0)
        ], milestones=[self.warmup_epochs])


@dataclasses.dataclass
class TrainerBuilder:
    training_method: type[BaseTrainer]
    model_config: ModelConfig
    training_context: TrainingContext

    def __init__(self, training_method: type[BaseTrainer], model_config: ModelConfig, training_context: TrainingContext):
        self.training_method = training_method
        self.model_config = model_config
        self.training_context = training_context

    def run_training(self, conf: str):
        trainer = self.training_method(
            self.model_config, self.training_context)
        return trainer.run_training(conf)

    def __call__(self, conf: str):
        return self.run_training(conf)


CONFIGS = {
    "resnetadapt": {
        'resnet20.3_levels.cifar10': TrainerBuilder(
            AdaptiveModelTrainer,
            resnetadapt.ResnetConfig(),
            ModelTraining()),
        'resnet20.3_levels.cifar100': TrainerBuilder(
            AdaptiveModelTrainer,
            resnetadapt.ResnetConfig()
            .set_num_classes(100),
            ModelTraining100()),

        'resnet20.6_levels.cifar10': TrainerBuilder(
            AdaptiveModelTrainer,
            resnetadapt.ResnetConfig()
            .set_small_channels((6, 8, 10, 12, 14, 16))
            .set_mid_channels((12, 16, 20, 24, 28, 32))
            .set_large_channels((24, 32, 40, 48, 56, 64)),
            ModelTraining()),
        'resnet20.6_levels.cifar100': TrainerBuilder(
            AdaptiveModelTrainer,
            resnetadapt.ResnetConfig()
            .set_small_channels((6, 8, 10, 12, 14, 16))
            .set_mid_channels((12, 16, 20, 24, 28, 32))
            .set_large_channels((24, 32, 40, 48, 56, 64))
            .set_num_classes(100),
            ModelTraining100()),

        'resnet56.3_levels.cifar10': TrainerBuilder(
            AdaptiveModelTrainer,
            resnetadapt.ResnetConfig()
            .set_num_blocks((9, 9, 9)),
            ModelTraining()),
        'resnet56.3_levels.cifar100': TrainerBuilder(
            AdaptiveModelTrainer,
            resnetadapt.ResnetConfig()
            .set_num_blocks((9, 9, 9))
            .set_num_classes(100),
            ModelTraining100()),

        'resnet56.6_levels.cifar10': TrainerBuilder(
            AdaptiveModelTrainer,
            resnetadapt.ResnetConfig()
            .set_num_blocks((9, 9, 9))
            .set_small_channels((6, 8, 10, 12, 14, 16))
            .set_mid_channels((12, 16, 20, 24, 28, 32))
            .set_large_channels((24, 32, 40, 48, 56, 64)),
            ModelTraining()),
        'resnet56.6_levels.cifar100': TrainerBuilder(
            AdaptiveModelTrainer,
            resnetadapt.ResnetConfig()
            .set_num_blocks((9, 9, 9))
            .set_small_channels((6, 8, 10, 12, 14, 16))
            .set_mid_channels((12, 16, 20, 24, 28, 32))
            .set_large_channels((24, 32, 40, 48, 56, 64))
            .set_num_classes(100),
            ModelTraining100()),
    },
    "vggadapt": {
        'vgg11.3_levels.cifar10': TrainerBuilder(
            AdaptiveModelTrainer,
            vggadapt.VGGConfig(),
            ModelTraining()),
        'vgg11.3_levels.cifar100': TrainerBuilder(
            AdaptiveModelTrainer,
            vggadapt.VGGConfig()
            .set_num_classes(100),
            ModelTraining100()),

        'vgg11.6_levels.cifar10': TrainerBuilder(
            AdaptiveModelTrainer,
            vggadapt.VGGConfig()
            .set_small_channels((24, 32, 40, 48, 56, 64))
            .set_mid_channels((48, 64, 80, 96, 112, 128))
            .set_large_channels((96, 128, 160, 192, 224, 256))
            .set_max_channels((192, 256, 320, 384, 448, 512)),
            ModelTraining()),
        'vgg11.6_levels.cifar100': TrainerBuilder(
            AdaptiveModelTrainer,
            vggadapt.VGGConfig()
            .set_num_classes(100)
            .set_small_channels((24, 32, 40, 48, 56, 64))
            .set_mid_channels((48, 64, 80, 96, 112, 128))
            .set_large_channels((96, 128, 160, 192, 224, 256))
            .set_max_channels((192, 256, 320, 384, 448, 512)),
            ModelTraining100()),

        'vgg19.3_levels.cifar10': TrainerBuilder(
            AdaptiveModelTrainer,
            vggadapt.VGGConfig()
            .set_version(19), ModelTraining()),
        'vgg19.3_levels.cifar100': TrainerBuilder(
            AdaptiveModelTrainer,
            vggadapt.VGGConfig()
            .set_num_classes(100),
            ModelTraining100()),

        'vgg19.6_levels.cifar10': TrainerBuilder(
            AdaptiveModelTrainer,
            vggadapt.VGGConfig()
            .set_version(19)
            .set_small_channels((24, 32, 40, 48, 56, 64))
            .set_mid_channels((48, 64, 80, 96, 112, 128))
            .set_large_channels((96, 128, 160, 192, 224, 256))
            .set_max_channels((192, 256, 320, 384, 448, 512)),
            ModelTraining()),
        'vgg19.6_levels.cifar100': TrainerBuilder(
            AdaptiveModelTrainer,
            vggadapt.VGGConfig()
            .set_version(19)
            .set_num_classes(100)
            .set_small_channels((24, 32, 40, 48, 56, 64))
            .set_mid_channels((48, 64, 80, 96, 112, 128))
            .set_large_channels((96, 128, 160, 192, 224, 256))
            .set_max_channels((192, 256, 320, 384, 448, 512)),
            ModelTraining100()),
    },
    "incremental": {
        "incr.resnet20.3_levels.cifar100": TrainerBuilder(
            AdaptiveModelTrainer,
            resnetadapt.ResnetConfig()
            .set_num_classes(100),
            ModelTraining100().set_incremental_training(True)),
        'incr.vgg11.3_levels.cifar100': TrainerBuilder(
            AdaptiveModelTrainer,
            vggadapt.VGGConfig()
            .set_num_classes(100),
            ModelTraining100().set_incremental_training(True)),
        'incr.vgg11.6_levels.cifar100': TrainerBuilder(
            AdaptiveModelTrainer,
            vggadapt.VGGConfig()
            .set_num_classes(100)
            .set_small_channels((24, 32, 40, 48, 56, 64))
            .set_mid_channels((48, 64, 80, 96, 112, 128))
            .set_large_channels((96, 128, 160, 192, 224, 256))
            .set_max_channels((192, 256, 320, 384, 448, 512)),
            ModelTraining100().set_incremental_training(True)),
    },
    "upscale": {
        "upscale.vgg11.cifar100": TrainerBuilder(
            AdaptiveModelTrainer,
            vggadapt.VGGConfig()
            .set_small_channels((64, 96, 128))
            .set_mid_channels((128, 192, 256))
            .set_large_channels((256, 384, 512))
            .set_max_channels((512, 768, 1024))
            .set_num_classes(100)
            .set_prebuilt_level(0),
            ModelTraining100().set_incremental_training(True)),
    }, "vitprebuild": {
        "cifar10": TrainerBuilder(
            SimpleTrainer,
            vit.ViTConfig()
            .set_num_classes(10), ViTTraining()),
        "cifar100": TrainerBuilder(
            SimpleTrainer,
            vit.ViTConfig()
            .set_num_classes(100), ViTTraining100())
    }, "vitadapt": {
        "cifar10": TrainerBuilder(
            AdaptiveModelTrainer,
            vitadapt.ViTConfig().set_num_classes(10),
            ViTTraining().set_load_from(vit.ViTConfig().set_num_classes(10))
        ),
        "cifar10.5levels": TrainerBuilder(
            AdaptiveModelTrainer,
            vitadapt.ViTConfig()
            .set_num_classes(10)
            .set_num_heads((12, 12, 12, 12, 12))
            .set_hidden_dims(
                (32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12)
            )
            .set_mlp_dims(
                (32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48)
            ),
            ViTTraining().set_load_from(vit.ViTConfig().set_num_classes(10))
        ),
        "cifar100": TrainerBuilder(
            AdaptiveModelTrainer,
            vitadapt.ViTConfig().set_num_classes(100),
            ViTTraining100().set_load_from(vit.ViTConfig().set_num_classes(100))
        ),
        "imagenet": TrainerBuilder(
            AdaptiveModelTrainer,
            vitadapt.ViTConfig().set_num_classes(1000)
            .set_num_heads((12, 12, 12, 12, 12))
            .set_hidden_dims(
                (32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12)
            )
            .set_mlp_dims(
                (32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48)
            ),
            VitTrainingImagenet()
        )
    }
}

DEFAULT_HARDWARE_CONFIG = hardware.HardwareConfig()
HARDWARE = {
    "vitprebuild": hardware.HardwareConfig().set_gpu_count(2),
    "vitadapt": hardware.HardwareConfig().set_gpu_count(4).set_time('72:00:00'),
}


def resolve_from_str(config, start=CONFIGS, return_on_index_error=False) -> Callable[[], BaseTrainer]:
    config = config.split(',')
    subpart = start
    for i in config:
        if i == 'all':
            continue
        try:
            i = int(i)
        except ValueError:
            pass
        try:
            subpart = subpart[i]
        except (KeyError, TypeError) as e:
            if not return_on_index_error:
                raise e
            return subpart
    return subpart


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


def print_all_conf_commands(conf, basestr, file=sys.stdout) -> None:
    for s in iter_over_conf(conf, basestr):
        hconf = resolve_from_str(s, HARDWARE, return_on_index_error=True)
        if hconf is HARDWARE:
            hconf = DEFAULT_HARDWARE_CONFIG
        print(f"{hconf.format_as_slurm_args()} experiment_job.sh {s}", file=file)


if __name__ == "__main__":
    command, conf = sys.argv[1:]
    # command = "run"
    # conf = "zeroout,vgg11.3_levels.cifar100"
    res = resolve_from_str(conf)
    if command == "list":
        print_all_conf_paths(res, conf)
    elif command == "run":
        hw = resolve_from_str(
            conf, HARDWARE, return_on_index_error=True)
        if isinstance(hw, hardware.HardwareConfig):
            hardware.CurrentDevice.set_hardware(hw)
        res(conf)
    elif command == "listcommand":
        print_all_conf_commands(res, conf)
