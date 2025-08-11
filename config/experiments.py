import math

from networks import flexresnet, flexvgg, flexvit, vit
from training import *

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, ReduceLROnPlateau
from functools import partial
import torch.optim as optim

from torchvision.datasets import CIFAR10, CIFAR100


class ModelTraining(FlexTrainingContext):
    def __init__(self, *args, **kwargs):
        super().__init__(partial(utils.load_data, CIFAR10),
                         patience=50, epochs=-1, *args, **kwargs)

    def make_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=1e-5)

    def make_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=300)


class ModelTraining100(FlexTrainingContext):
    def __init__(self, *args, **kwargs):
        super().__init__(partial(utils.load_data, CIFAR100),
                         patience=50, epochs=-1, *args, **kwargs)

    def make_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=1e-5)

    def make_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=300)


class ViTTraining(FlexTrainingContext):
    def __init__(self, *args, **kwargs):
        super().__init__(partial(utils.load_data, CIFAR10,
                                 resize=(224, 224)), patience=20, epochs=300, *args, **kwargs)

    def make_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=1e-5)

    def make_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=300)


class ViTTraining100(FlexTrainingContext):
    def __init__(self, *args, **kwargs):
        super().__init__(partial(utils.load_data, CIFAR100,
                                 resize=(224, 224)), patience=20, epochs=300, *args, **kwargs)

    def make_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=1e-5)

    def make_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=self.epochs)


class VitTrainingImagenet(FlexTrainingContext):
    def __init__(self, *args, **kwargs):
        super().__init__(utils.load_imagenet, patience=20, epochs=100,
                         label_smoothing=0.11, gradient_clip_val=1.0, *args, **kwargs)

    def make_optimizer(self, model):
        return optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.3)

    def make_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer=optimizer, T_max=100, eta_min=1e-8)


class VitTrainingImagenetWarmup(FlexTrainingContext):
    warmup_epochs: int = 30

    def __init__(self, *args, **kwargs):
        super().__init__(utils.load_imagenet, patience=50, epochs=300,
                         label_smoothing=0.11, gradient_clip_val=1.0, *args, **kwargs)

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
            flexresnet.ResnetConfig(
                num_classes=100),
            ModelTraining100()),

        'resnet20.6_levels.cifar10': TrainerBuilder(
            FlexModelTrainer,
            flexresnet.ResnetConfig(
                small_channels=(6, 8, 10, 12, 14, 16),
                mid_channels=(12, 16, 20, 24, 28, 32),
                large_channels=(24, 32, 40, 48, 56, 64),
            ),
            ModelTraining()),
        'resnet20.6_levels.cifar100': TrainerBuilder(
            FlexModelTrainer,
            flexresnet.ResnetConfig(
                small_channels=(6, 8, 10, 12, 14, 16),
                mid_channels=(12, 16, 20, 24, 28, 32),
                large_channels=(24, 32, 40, 48, 56, 64),
                num_classes=100),
            ModelTraining100()),

        'resnet56.3_levels.cifar10': TrainerBuilder(
            FlexModelTrainer,
            flexresnet.ResnetConfig(
                num_blocks=(9, 9, 9)),
            ModelTraining()),
        'resnet56.3_levels.cifar100': TrainerBuilder(
            FlexModelTrainer,
            flexresnet.ResnetConfig(
                num_blocks=(9, 9, 9),
                num_classes=100),
            ModelTraining100()),

        'resnet56.6_levels.cifar10': TrainerBuilder(
            FlexModelTrainer,
            flexresnet.ResnetConfig(
                num_blocks=(9, 9, 9),
                small_channels=(6, 8, 10, 12, 14, 16),
                mid_channels=(12, 16, 20, 24, 28, 32),
                large_channels=(24, 32, 40, 48, 56, 64)),
            ModelTraining()),
        'resnet56.6_levels.cifar100': TrainerBuilder(
            FlexModelTrainer,
            flexresnet.ResnetConfig(
                num_blocks=(9, 9, 9),
                small_channels=(6, 8, 10, 12, 14, 16),
                mid_channels=(12, 16, 20, 24, 28, 32),
                large_channels=(24, 32, 40, 48, 56, 64),
                num_classes=100),
            ModelTraining100()),
    },
    "flexvgg": {
        'vgg11.3_levels.cifar10': TrainerBuilder(
            FlexModelTrainer,
            flexvgg.VGGConfig(),
            ModelTraining()),
        'vgg11.3_levels.cifar100': TrainerBuilder(
            FlexModelTrainer,
            flexvgg.VGGConfig(
                num_classes=100),
            ModelTraining100()),

        'vgg11.6_levels.cifar10': TrainerBuilder(
            FlexModelTrainer,
            flexvgg.VGGConfig(
                small_channels=(24, 32, 40, 48, 56, 64),
                mid_channels=(48, 64, 80, 96, 112, 128),
                large_channels=(96, 128, 160, 192, 224, 256),
                max_channels=(192, 256, 320, 384, 448, 512)),
            ModelTraining()),
        'vgg11.6_levels.cifar100': TrainerBuilder(
            FlexModelTrainer,
            flexvgg.VGGConfig(
                num_classes=100,
                small_channels=(24, 32, 40, 48, 56, 64),
                mid_channels=(48, 64, 80, 96, 112, 128),
                large_channels=(96, 128, 160, 192, 224, 256),
                max_channels=(192, 256, 320, 384, 448, 512)),
            ModelTraining100()),

        'vgg19.3_levels.cifar10': TrainerBuilder(
            FlexModelTrainer,
            flexvgg.VGGConfig(
                version=19),
            ModelTraining()),
        'vgg19.3_levels.cifar100': TrainerBuilder(
            FlexModelTrainer,
            flexvgg.VGGConfig(
                num_classes=100),
            ModelTraining100()),

        'vgg19.6_levels.cifar10': TrainerBuilder(
            FlexModelTrainer,
            flexvgg.VGGConfig(
                version=19,
                small_channels=(24, 32, 40, 48, 56, 64),
                mid_channels=(48, 64, 80, 96, 112, 128),
                large_channels=(96, 128, 160, 192, 224, 256),
                max_channels=(192, 256, 320, 384, 448, 512)),
            ModelTraining()),
        'vgg19.6_levels.cifar100': TrainerBuilder(
            FlexModelTrainer,
            flexvgg.VGGConfig(
                version=19,
                num_classes=100,
                small_channels=(24, 32, 40, 48, 56, 64),
                mid_channels=(48, 64, 80, 96, 112, 128),
                large_channels=(96, 128, 160, 192, 224, 256),
                max_channels=(192, 256, 320, 384, 448, 512)),
            ModelTraining100()),
    }, "vitprebuild": {
        "cifar10": TrainerBuilder(
            SimpleTrainer,
            vit.ViTConfig(
                num_classes=10),
            ViTTraining()),
        "cifar100": TrainerBuilder(
            SimpleTrainer,
            vit.ViTConfig(
                num_classes=100),
            ViTTraining100())
    }, "flexvit": {
        "cifar10": TrainerBuilder(
            FlexModelTrainer,
            flexvit.ViTConfig(
                num_classes=10),
            ViTTraining(
                load_from=vit.ViTConfig(
                    num_classes=10))
        ),
        "cifar10.5levels": TrainerBuilder(
            FlexModelTrainer,
            flexvit.ViTConfig(
                num_classes=10,
                num_heads=(12, 12, 12, 12, 12),
                hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
                mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48)),
            ViTTraining(
                load_from=vit.ViTConfig(
                    num_classes=10))
        ),
        "cifar100": TrainerBuilder(
            FlexModelTrainer,
            flexvit.ViTConfig(
                num_classes=100),
            ViTTraining100(
                load_from=vit.ViTConfig(
                    num_classes=100))
        ),
        "imagenet": TrainerBuilder(
            FlexModelTrainer,
            flexvit.ViTConfig(
                num_classes=1000,
                num_heads=(12, 12, 12, 12, 12),
                hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
                mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48)),
            VitTrainingImagenet()),
        # "imagenet": TrainerBuilder(
        #     FlexModelTrainer,
        #     flexvit.ViTConfig(
        #         num_classes=1000,
        #         num_heads=(12, 12, 12, 12, 12, 12, 12, 12, 12, 12),
        #         hidden_dims=(16 * 12, 21 * 12, 26 * 12, 31 * 12, 36 * 12, 41 * 12, 46 * 12, 51 * 12, 56 * 12, 64 * 12),
        #         mlp_dims=(16 * 48, 21 * 48, 26 * 48, 31 * 48, 36 * 48, 41 * 48, 46 * 48, 51 * 48, 56*48, 64 * 48)),
        #     VitTrainingImagenet()
        # ),
    }, "flexvitcorrect": TrainerBuilder(
        FlexModelTrainer,
        flexvit.ViTConfig(
            num_classes=1000,
            num_heads=(12, 12, 12, 12, 12),
            hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
            mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48)),
        VitTrainingImagenet(
            load_from=flexvit.ViTConfig(
                num_classes=1000,
                num_heads=(12, 12, 12, 12, 12),
                hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
                mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48)))
    )
}
