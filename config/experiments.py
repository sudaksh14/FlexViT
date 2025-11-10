import math
from xml.parsers.expat import model

from networks import flexresnet, flexvgg, flexvit, vit, flexdeit_v3
from training import *
from networks.vit import ViTPrebuilt

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, ReduceLROnPlateau
from functools import partial
import torch.optim as optim
import timm
from torchvision.datasets import CIFAR10, CIFAR100
import distillation.training
import distillation.dataset
from timm.optim import Lamb
from timm.scheduler import CosineLRScheduler


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
        return CosineAnnealingLR(optimizer, T_max=self.epochs)


class ViTTraining100(FlexTrainingContext):
    def __init__(self, *args, **kwargs):
        # super().__init__(partial(utils.load_data, CIFAR100,
        #                          resize=(224, 224)), patience=20, epochs=1, *args, **kwargs)
        super().__init__(distillation.dataset.load_cifar100, patience=20, epochs=150, *args, **kwargs)

    def make_optimizer(self, model):
        return torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    def make_scheduler(self, optimizer):
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=10)
        cosine = CosineAnnealingLR(optimizer, T_max=self.epochs-10)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[10])


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

def load_teacher():
    """
    Simple loader for a teacher model with a given architecture.

    Returns
    -------
    model : torch.nn.Module
    Teacher model loaded with the state dict and moved to device in eval mode.
    """
    # Create a timm ViT without distillation
    model = timm.create_model('deit3_base_patch16_224.fb_in22k_ft_in1k', pretrained=True, num_classes=1000)
    model.eval().to(utils.get_device())
    return model

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
                num_classes=100,
                num_heads=(12, 12, 12, 12, 12),
                hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
                mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48)),
            ViTTraining100()
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
        distillation.training.ScalaDistillTrainer,
        flexvgg.VGGConfig(
            num_classes=100,
            small_channels=(24, 32, 40, 48, 56, 64),
            mid_channels=(48, 64, 80, 96, 112, 128),
            large_channels=(96, 128, 160, 192, 224, 256),
            max_channels=(192, 256, 320, 384, 448, 512)),
        distillation.training.ScalaDistillContext(
            loader_function=partial(
                distillation.dataset.load_imagenet,
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
        distillation.training.ScalaDistillTrainer,
        flexvit.ViTConfig(
            num_classes=1000,
            num_heads=(12, 12, 12, 12, 12),
            hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
            mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48)),
        distillation.training.ScalaDistillContext(
            # loader_function=partial(utils.load_dummy_data, batch_size=256),
            loader_function=partial(distillation.dataset.load_imagenet, batch_size=512),
            teacher_loader=load_teacher,
            make_optimizer=lambda m: optim.AdamW(
                m.parameters(), lr=5e-4, weight_decay=0.05),
            make_scheduler=lambda opt: CosineAnnealingLR(
                optimizer=opt, T_max=150, eta_min=1e-5),
            mixup_fn=utils.mixup_fn,
            patience=20, epochs=150,
            label_smoothing=0.11, gradient_clip_val=1.0)
    ), 'flexdeit_v3': TrainerBuilder(
        distillation.training.ScalaDistillTrainer,
        flexdeit_v3.ViTConfig_v3(
            num_classes=1000,
            num_heads=(12, 12, 12, 12, 12),
            hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
            mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48)),
        distillation.training.ScalaDistillContext(
            loader_function=partial(utils.load_dummy_data, batch_size=256),
            # loader_function=partial(scala.dataset.load_imagenet, batch_size=128),
            teacher_loader=load_teacher,
            make_optimizer=lambda m: Lamb(m.parameters(),
                lr=5e-4, weight_decay=0.05, betas=(0.9, 0.999)),
            make_scheduler=lambda opt: CosineLRScheduler(optimizer=opt,
                t_initial=100, lr_min=1e-6, warmup_lr_init=1e-6,
                warmup_t=5, cycle_limit=1),
            mixup_fn=utils.mixup_fn,
            patience=20, epochs=1,
            label_smoothing=0.11, gradient_clip_val=1.0)
    ) 
}
