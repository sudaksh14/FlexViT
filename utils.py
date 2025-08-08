from typing import Union, Any
import shutil
import os
import io

from torchvision.transforms import (
    Compose, RandomHorizontalFlip, RandomRotation,
    ColorJitter, ToTensor, Normalize, Resize, CenterCrop, ConvertImageDtype, RandAugment
)
from timm.data import Mixup
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Subset
from torch import nn
import torch
import tqdm

from flex_modules.module import Module
from networks.modules import ClassTokenLayer, PosEmbeddingLayer, LinearHead
import config.paths as paths

# Some of this code is from https://github.com/poojamangal15/Adaptive-Neural-Networks


def get_device() -> 'str':
    return torch.device("mps" if torch.backends.mps.is_available() else
                        "cuda" if torch.cuda.is_available() else "cpu")


def make_str_filename_safe(s: str):
    prefix_char = 'x'
    forbidden_chars = [
        ('x', 'xx'),
        ('/', 'xa'),
        ('<', 'xb'),
        ('>', 'xc'),
        (':', 'xd'),
        ('"', 'xe'),
        ('/', 'xf'),
        ('\\', 'xg'),
        ('|', 'xh'),
        ('?', 'xi'),
        ('*', 'xj'),
        ('(', 'xk'),
        (')', 'xl'),
        ('.', 'xm'),
        (',', 'xn'),
        ('\'', 'xo')
    ]

    description = s
    description = description.replace(
        prefix_char, f"{prefix_char}{prefix_char}")
    for forbidden, replacement in forbidden_chars:
        description = description.replace(forbidden, replacement)
    return description


class SelfDescripting:
    def setv(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def get_description(self) -> str:
        res = f"{self.__class__.__name__}"
        for name, val in self.__dict__.items():
            if name[:2] == "__":
                continue
            try:
                descr = val.get_description()
                res += f"_({descr})"
            except AttributeError:
                res += f"_{val}"
        return res

    def get_filename_safe_description(self) -> str:
        return make_str_filename_safe(self.get_description())

    def get_flat_dict(self) -> str:
        res = {}
        for name, val in self.__dict__.items():
            if name[:2] == "__":
                continue
            try:
                flatdict = val.get_flat_dict()
                for dname, dval in flatdict.items():
                    res[f"{name}.{dname}"] = dval
            except AttributeError:
                res[f"{name}"] = val
        return res


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: str) -> torch.Tensor:
    """
    Evaluates the model on the given dataloader and returns accuracy and F1 score.

    from https://github.com/poojamangal15/Adaptive-Neural-Networks
    """
    all_preds = []
    all_labels = []
    # Move model to the correct device and ensure correct data type
    model = model.to(device).to(torch.float32)
    model.eval()

    with torch.no_grad():
        for images, labels in tqdm.tqdm(dataloader):
            # Ensure images are on the same device and data type
            images = images.to(device).to(torch.float32)
            labels = labels.to(device)

            outputs = model(images)  # Perform forward pass
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy


def count_parameters(model: nn.Module) -> int:
    """
    Counts the number of trainable parameters in the model.

    from https://github.com/poojamangal15/Adaptive-Neural-Networks
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_in_mb(model: nn.Module) -> int:
    """
    Gets the models file size.

    adapted from https://github.com/poojamangal15/Adaptive-Neural-Networks
    """
    f = io.BytesIO()
    torch.save(model.state_dict(), f)
    return len(f.getvalue())


def try_make_dir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


IMAGENET_TRANSFORMS = [
    Resize(256),
    CenterCrop(224),
    RandomHorizontalFlip(p=0.5),
    RandAugment(num_ops=2, magnitude=9, interpolation=InterpolationMode.BILINEAR),
    ColorJitter(0.4, 0.4, 0.4, 0.1),
    ToTensor(),
    ConvertImageDtype(torch.float),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
]


def load_imagenet(data_dir=paths.IMAGENET_PATH, tmp_dir=paths.TMPDIR, batch_size=128):
    train_transform = Compose(IMAGENET_TRANSFORMS)
    test_transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        ConvertImageDtype(torch.float),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    train_dataset = ImageFolder(data_dir / "train", transform=train_transform)
    test_dataset = ImageFolder(data_dir / "val", transform=test_transform)

    # train_dataset = Subset(train_dataset, indices=torch.randperm(len(train_dataset))[:5000])
    # test_dataset = Subset(test_dataset, indices=torch.randperm(len(test_dataset))[:1000])

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    val_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=16)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=16)

    print(f"Using Batch Size: {batch_size} for dataloaders")
    return train_dataloader, val_dataloader, test_dataloader

# ----- Mixup + CutMix -----
mixup_fn = Mixup(
    mixup_alpha=0.2,
    cutmix_alpha=1.0,
    cutmix_minmax=None,
    prob=1.0,
    switch_prob=0.5,  # probability to switch between mixup and cutmix
    mode='batch',
    label_smoothing=0.11,
    num_classes=1000
)

def load_data(dataset, data_dir=paths.DATA_PATH, tmp_dir=paths.TMPDIR, resize=None, batch_size=64):
    """
    Loads data for CIFAR10 or CIFAR100

    Inspired by code from https://github.com/poojamangal15/Adaptive-Neural-Networks
    """
    normalizers = {
        CIFAR10: {'mean': [0.485, 0.456, 0.406],
                  'std': [0.229, 0.224, 0.225], },
        CIFAR100: {'mean': [0.5070, 0.4865, 0.4409],
                   'std': [0.2673, 0.2564, 0.2761], }
    }

    train_transform = [
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=15),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ToTensor(),
        Normalize(**normalizers[dataset])
    ]

    test_transform = [
        ToTensor(),
        Normalize(**normalizers[dataset])
    ]

    if resize:
        test_transform.insert(0, Resize(resize))
        train_transform.insert(0, Resize(resize))

    test_transform = Compose(test_transform)
    train_transform = Compose(train_transform)

    if tmp_dir:
        try_make_dir(data_dir)
        try_make_dir(tmp_dir)
        shutil.copytree(data_dir, tmp_dir, dirs_exist_ok=True)
    train_dataset = dataset(
        root=data_dir if tmp_dir is None else tmp_dir,
        train=True, download=True, transform=train_transform)
    test_dataset = dataset(
        root=data_dir if tmp_dir is None else tmp_dir,
        train=False, download=True, transform=test_transform)
    if tmp_dir is not None:
        shutil.copytree(tmp_dir, data_dir, dirs_exist_ok=True)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    val_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=8)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=8)

    return train_dataloader, val_dataloader, test_dataloader


def flexible_model_copy(src: Union[nn.Module, dict[str, Any]], dest: nn.Module):
    if not isinstance(src, nn.Module):
        dest.load_state_dict(src)
        return

    if isinstance(src, Module):
        src.copy_to_base(dest)
        return

    if isinstance(dest, Module):
        dest.load_from_base(src)
        return

    dest.load_state_dict(src.state_dict())


def torch_serialize(obj, *args, **kwargs):
    with io.BytesIO() as f:
        torch.save(obj, f, *args, **kwargs)
        return f.getvalue()


def torch_deserialize(data: bytes, *args, **kwargs):
    with io.BytesIO(data) as f:
        return torch.load(f, *args, **kwargs)


def save_model(model_config, model):
    with open(paths.TRAINED_MODELS / f"{model_config.get_filename_safe_description()}.pt", "wb") as f:
        torch.save(model.state_dict(), f)

def save_statedict(name, model):
    with open(paths.TRAINED_MODELS / f"{name}.pt", "wb") as f:
        torch.save(model.state_dict(), f)
    print("model state dict saved")

def load_model(model_config):
    model = model_config.make_model()
    with open(paths.TRAINED_MODELS / f"{model_config.get_filename_safe_description()}.pt", "rb") as f:
        sdict = torch.load(f)
        model.load_state_dict(sdict)
    return model
