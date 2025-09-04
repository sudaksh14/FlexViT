import math

from networks import flexresnet, flexvgg, flexvit, vit
from training import *
from networks.vit import ViTPrebuilt

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, ReduceLROnPlateau
from functools import partial
import torch.optim as optim

from torchvision.datasets import CIFAR10, CIFAR100
import scala.training
import scala.dataset

import timm.optim
import timm
from timm.optim import create_optimizer


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
                                 resize=(224, 224)), patience=20, epochs=1, *args, **kwargs)

    def make_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=1e-5)

    def make_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=self.epochs)


class VitTrainingImagenet(FlexTrainingContext):
    def __init__(self, *args, **kwargs):
        super().__init__(utils.load_imagenet, patience=20, epochs=150,
                         label_smoothing=0.11, gradient_clip_val=1.0, *args, **kwargs)

    def make_optimizer(self, model):
        return optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.3)

    def make_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer=optimizer, T_max=150, eta_min=1e-8)


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
                load_from='vitprebuild,cifar10')
        ),
        "cifar10.5levels": TrainerBuilder(
            FlexModelTrainer,
            flexvit.ViTConfig(
                num_classes=10,
                num_heads=(12, 12, 12, 12, 12),
                hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
                mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48)),
            ViTTraining(
                load_from='vitprebuild,cifar10')
        ),
        "cifar100": TrainerBuilder(
            FlexModelTrainer,
            flexvit.ViTConfig(
                num_classes=100),
            ViTTraining100(loader_function=partial(
                scala.dataset.load_cifar100))
        ),
        "imagenet": TrainerBuilder(
            FlexModelTrainer,
            flexvit.ViTConfig(
                num_classes=1000,
                num_heads=(12, 12, 12, 12, 12),
                hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
                mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48)),
            VitTrainingImagenet()
        ),
        "imagenet_non_uniform_heads": TrainerBuilder(
            FlexModelTrainer,
            flexvit.ViTConfig(
                num_classes=1000,
                num_heads=(4, 6, 8, 10, 12),
                hidden_dims=(64 * 4, 64 * 6, 64 * 8, 64 * 10, 64 * 12),
                mlp_dims=(64 * 16, 64 * 24, 64 * 32, 64 * 40, 64 * 48)),
            VitTrainingImagenet())
    }, "flexvitcorrect": TrainerBuilder(
        FlexModelTrainer,
        flexvit.ViTConfig(
            num_classes=1000,
            num_heads=(12, 12, 12, 12, 12),
            hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
            mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48)),
        VitTrainingImagenet(
            load_from='flexvit,imagenet')
    ), 'scala_test': TrainerBuilder(
        scala.training.ScalaDistillTrainer,
        flexvgg.VGGConfig(
            num_classes=100,
            small_channels=(24, 32, 40, 48, 56, 64),
            mid_channels=(48, 64, 80, 96, 112, 128),
            large_channels=(96, 128, 160, 192, 224, 256),
            max_channels=(192, 256, 320, 384, 448, 512)),
        scala.training.ScalaDistillContext(
            loader_function=partial(
                scala.dataset.load_imagenet,
                data_set='CIFAR',
                datapath=paths.DATA_PATH,
                input_size=32),
            teacher_loader=flexvgg.VGGConfig(
                num_classes=100,
                small_channels=(24, 32, 40, 48, 56, 64),
                mid_channels=(48, 64, 80, 96, 112, 128),
                large_channels=(96, 128, 160, 192, 224, 256),
                max_channels=(192, 256, 320, 384, 448, 512)).make_model,
            make_optimizer=lambda m: optim.AdamW(
                m.parameters(), lr=1e-5, weight_decay=0.3),
            make_scheduler=lambda opt: CosineAnnealingLR(
                optimizer=opt, T_max=150, eta_min=1e-8)
        )
    ), 'flexvit_distill': TrainerBuilder(
        scala.training.ScalaDistillTrainer,
        flexvit.ViTConfig(
            num_classes=1000,
            num_heads=(12, 12, 12, 12, 12),
            hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
            mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48)),
        scala.training.ScalaDistillContext(
            # loader_function=partial(utils.load_dummy_data, batch_size=256),
            loader_function=partial(scala.dataset.load_imagenet, batch_size=512),
            teacher_loader=flexvit.ViTConfig(
                num_classes=1000,
                num_heads=(12, 12, 12, 12, 12),
                hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
                mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48)).make_model,
            make_optimizer=lambda m: optim.AdamW(
                m.parameters(), lr=1e-5, weight_decay=0.3),
            make_scheduler=lambda opt: CosineAnnealingLR(
                optimizer=opt, T_max=150, eta_min=1e-8),
            mixup_fn=utils.mixup_fn,
            patience=20, epochs=150,
            label_smoothing=0.11, gradient_clip_val=1.0)
    ),
    'flexvit_distill_v3': TrainerBuilder(
        scala.training.ScalaDistillTrainer,
        flexvit.ViTConfig(
            prebuilt=ViTPrebuilt.Deit_v3_pretrain_21k,
            num_classes=1000,
            num_heads=(12, 12, 12, 12, 12),
            hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
            mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48)),
        scala.training.ScalaDistillContext(
            # loader_function=partial(utils.load_dummy_data, batch_size=256),
            loader_function=partial(scala.dataset.load_imagenet, batch_size=512),
            teacher_loader=flexvit.ViTConfig(
                prebuilt=ViTPrebuilt.Deit_v3_pretrain_21k,
                num_classes=1000,
                num_heads=(12, 12, 12, 12, 12),
                hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
                mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48)).make_model,
            make_optimizer=lambda m: optim.AdamW(
                m.parameters(), lr=1e-5, weight_decay=0.3),
            make_scheduler=lambda opt: CosineAnnealingLR(
                optimizer=opt, T_max=150, eta_min=1e-8),
            mixup_fn=utils.mixup_fn,
            patience=20, epochs=150,
            label_smoothing=0.11, gradient_clip_val=1.0)
    ) 
}
