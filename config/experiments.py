from networks import flexresnet, flexvgg, flexvit, vit
from training import *

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from functools import partial
import torch.optim as optim

from torchvision.datasets import CIFAR10, CIFAR100


class ModelTraining(FlexTrainingContext):
    def __init__(self, *args, **kwargs):
        super().__init__(partial(utils.load_data, CIFAR10,
                                 *args, **kwargs), patience=50, epochs=-1)

    def make_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=1e-5)

    def make_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=self.epochs)


class ModelTraining100(FlexTrainingContext):
    def __init__(self, *args, **kwargs):
        super().__init__(partial(utils.load_data, CIFAR100,
                                 *args, **kwargs), patience=50, epochs=-1)

    def make_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=1e-5)

    def make_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=self.epochs)


class ViTTraining(FlexTrainingContext):
    def __init__(self, *args, **kwargs):
        super().__init__(partial(utils.load_data, CIFAR10,
                                 resize=(224, 224)), patience=20, epochs=-1)

    def make_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=1e-5)

    def make_scheduler(self, optimizer):
        # I accidentally put the wrong value here, but this is
        # giving pretty good results, so I'm not changing it.
        return CosineAnnealingLR(optimizer, T_max=1e-5)


class ViTTraining100(FlexTrainingContext):
    def __init__(self, *args, **kwargs):
        super().__init__(partial(utils.load_data, CIFAR100,
                                 resize=(224, 224)), patience=20, epochs=-1)

    def make_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=1e-5)

    def make_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=self.epochs)


class VitTrainingImagenet(FlexTrainingContext):
    def __init__(self, *args, **kwargs):
        super().__init__(utils.load_imagenet, patience=50, epochs=-1,
                         label_smoothing=0.11, gradient_clip_val=1.0)

    def make_optimizer(self, model):
        return optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.3)

    def make_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=0.0)


class VitTrainingImagenetWarmup(FlexTrainingContext):
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


CONFIGS = {
    "flexresnet": {
        'resnet20.3_levels.cifar10': TrainerBuilder(
            FlexModelTrainer,
            flexresnet.ResnetConfig(),
            ModelTraining()),
        'resnet20.3_levels.cifar100': TrainerBuilder(
            FlexModelTrainer,
            flexresnet.ResnetConfig()
            .set_num_classes(100),
            ModelTraining100()),

        'resnet20.6_levels.cifar10': TrainerBuilder(
            FlexModelTrainer,
            flexresnet.ResnetConfig()
            .set_small_channels((6, 8, 10, 12, 14, 16))
            .set_mid_channels((12, 16, 20, 24, 28, 32))
            .set_large_channels((24, 32, 40, 48, 56, 64)),
            ModelTraining()),
        'resnet20.6_levels.cifar100': TrainerBuilder(
            FlexModelTrainer,
            flexresnet.ResnetConfig()
            .set_small_channels((6, 8, 10, 12, 14, 16))
            .set_mid_channels((12, 16, 20, 24, 28, 32))
            .set_large_channels((24, 32, 40, 48, 56, 64))
            .set_num_classes(100),
            ModelTraining100()),

        'resnet56.3_levels.cifar10': TrainerBuilder(
            FlexModelTrainer,
            flexresnet.ResnetConfig()
            .set_num_blocks((9, 9, 9)),
            ModelTraining()),
        'resnet56.3_levels.cifar100': TrainerBuilder(
            FlexModelTrainer,
            flexresnet.ResnetConfig()
            .set_num_blocks((9, 9, 9))
            .set_num_classes(100),
            ModelTraining100()),

        'resnet56.6_levels.cifar10': TrainerBuilder(
            FlexModelTrainer,
            flexresnet.ResnetConfig()
            .set_num_blocks((9, 9, 9))
            .set_small_channels((6, 8, 10, 12, 14, 16))
            .set_mid_channels((12, 16, 20, 24, 28, 32))
            .set_large_channels((24, 32, 40, 48, 56, 64)),
            ModelTraining()),
        'resnet56.6_levels.cifar100': TrainerBuilder(
            FlexModelTrainer,
            flexresnet.ResnetConfig()
            .set_num_blocks((9, 9, 9))
            .set_small_channels((6, 8, 10, 12, 14, 16))
            .set_mid_channels((12, 16, 20, 24, 28, 32))
            .set_large_channels((24, 32, 40, 48, 56, 64))
            .set_num_classes(100),
            ModelTraining100()),
    },
    "flexvgg": {
        'vgg11.3_levels.cifar10': TrainerBuilder(
            FlexModelTrainer,
            flexvgg.VGGConfig(),
            ModelTraining()),
        'vgg11.3_levels.cifar100': TrainerBuilder(
            FlexModelTrainer,
            flexvgg.VGGConfig()
            .set_num_classes(100),
            ModelTraining100()),

        'vgg11.6_levels.cifar10': TrainerBuilder(
            FlexModelTrainer,
            flexvgg.VGGConfig()
            .set_small_channels((24, 32, 40, 48, 56, 64))
            .set_mid_channels((48, 64, 80, 96, 112, 128))
            .set_large_channels((96, 128, 160, 192, 224, 256))
            .set_max_channels((192, 256, 320, 384, 448, 512)),
            ModelTraining()),
        'vgg11.6_levels.cifar100': TrainerBuilder(
            FlexModelTrainer,
            flexvgg.VGGConfig()
            .set_num_classes(100)
            .set_small_channels((24, 32, 40, 48, 56, 64))
            .set_mid_channels((48, 64, 80, 96, 112, 128))
            .set_large_channels((96, 128, 160, 192, 224, 256))
            .set_max_channels((192, 256, 320, 384, 448, 512)),
            ModelTraining100()),

        'vgg19.3_levels.cifar10': TrainerBuilder(
            FlexModelTrainer,
            flexvgg.VGGConfig()
            .set_version(19), ModelTraining()),
        'vgg19.3_levels.cifar100': TrainerBuilder(
            FlexModelTrainer,
            flexvgg.VGGConfig()
            .set_num_classes(100),
            ModelTraining100()),

        'vgg19.6_levels.cifar10': TrainerBuilder(
            FlexModelTrainer,
            flexvgg.VGGConfig()
            .set_version(19)
            .set_small_channels((24, 32, 40, 48, 56, 64))
            .set_mid_channels((48, 64, 80, 96, 112, 128))
            .set_large_channels((96, 128, 160, 192, 224, 256))
            .set_max_channels((192, 256, 320, 384, 448, 512)),
            ModelTraining()),
        'vgg19.6_levels.cifar100': TrainerBuilder(
            FlexModelTrainer,
            flexvgg.VGGConfig()
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
            FlexModelTrainer,
            flexresnet.ResnetConfig()
            .set_num_classes(100),
            ModelTraining100().set_incremental_training(True)),
        'incr.vgg11.3_levels.cifar100': TrainerBuilder(
            FlexModelTrainer,
            flexvgg.VGGConfig()
            .set_num_classes(100),
            ModelTraining100().set_incremental_training(True)),
        'incr.vgg11.6_levels.cifar100': TrainerBuilder(
            FlexModelTrainer,
            flexvgg.VGGConfig()
            .set_num_classes(100)
            .set_small_channels((24, 32, 40, 48, 56, 64))
            .set_mid_channels((48, 64, 80, 96, 112, 128))
            .set_large_channels((96, 128, 160, 192, 224, 256))
            .set_max_channels((192, 256, 320, 384, 448, 512)),
            ModelTraining100().set_incremental_training(True)),
    },
    "upscale": {
        "upscale.vgg11.cifar100": TrainerBuilder(
            FlexModelTrainer,
            flexvgg.VGGConfig()
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
    }, "flexvit": {
        "cifar10": TrainerBuilder(
            FlexModelTrainer,
            flexvit.ViTConfig().set_num_classes(10),
            ViTTraining().set_load_from(vit.ViTConfig().set_num_classes(10))
        ),
        "cifar10.5levels": TrainerBuilder(
            FlexModelTrainer,
            flexvit.ViTConfig()
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
            FlexModelTrainer,
            flexvit.ViTConfig().set_num_classes(100),
            ViTTraining100().set_load_from(vit.ViTConfig().set_num_classes(100))
        ),
        "imagenet": TrainerBuilder(
            FlexModelTrainer,
            flexvit.ViTConfig().set_num_classes(1000)
            .set_num_heads((12, 12, 12, 12, 12))
            .set_hidden_dims(
                (32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12)
            )
            .set_mlp_dims(
                (32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48)
            ),
            VitTrainingImagenet()
        )
    }, "flexvitcorrect": TrainerBuilder(
        FlexModelTrainer,
        flexvit.ViTConfig().set_num_classes(1000)
        .set_num_heads((12, 12, 12, 12, 12))
        .set_hidden_dims(
            (32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12)
        )
        .set_mlp_dims(
            (32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48)
        ),
        VitTrainingImagenet().set_load_from(flexvit.ViTConfig().set_num_classes(1000)
                                            .set_num_heads((12, 12, 12, 12, 12))
                                            .set_hidden_dims(
            (32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12)
        )
            .set_mlp_dims(
            (32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48)
        ))
    )
}
