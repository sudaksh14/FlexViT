from flex_modules.module import Module
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score
import os

from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

from torchvision.transforms import (
    Compose, RandomHorizontalFlip, RandomRotation,
    ColorJitter, ToTensor, Normalize, Resize, CenterCrop, ConvertImageDtype
)

from torch.utils.data import DataLoader
import shutil

import config.paths as paths

import tqdm

from networks.vit_modules import ClassTokenLayer, PosEmbeddingLayer


def get_device() -> 'str':
    return torch.device("mps" if torch.backends.mps.is_available() else
                        "cuda" if torch.cuda.is_available() else "cpu")


def fluent_setters(cls):
    variables = list(filter(lambda s: s[:2] != "__", cls.__dict__))
    for attr in variables:
        def make_setter(attr_name):
            def setter(self, value):
                setattr(self, attr_name, value)
                return self
            return setter
        setattr(cls, f'set_{attr}', make_setter(attr))
    return cls


class SelfDescripting:
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
        prefix_char = 'x'
        forbidden_chars = [
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
        ]

        description = self.get_description()
        description = description.replace(
            prefix_char, f"{prefix_char}{prefix_char}")
        for forbidden, replacement in forbidden_chars:
            description = description.replace(forbidden, replacement)
        return description

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
    """Counts the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_in_mb(model: nn.Module) -> int:
    torch.save(model.state_dict(), "temp.p")
    size_mb = os.path.getsize("temp.p") / (1024 * 1024)
    os.remove("temp.p")
    return size_mb


def try_make_dir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


IMAGENET_TRANSFORMS = [
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    ConvertImageDtype(torch.float),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
]


def load_imagenet(data_dir=paths.IMAGENET_PATH, tmp_dir=paths.TMPDIR, batch_size=128):
    transform = Compose(IMAGENET_TRANSFORMS)

    train_dataset = ImageFolder(data_dir / "train", transform=transform)
    test_dataset = ImageFolder(data_dir / "val", transform=transform)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    val_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=16)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=16)

    print("made dataloaders")
    return train_dataloader, val_dataloader, test_dataloader


def load_data(dataset, data_dir=paths.DATA_PATH, tmp_dir=paths.TMPDIR, resize=None, batch_size=64):
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
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=8)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=8)

    return train_dataloader, val_dataloader, test_dataloader


def flexible_model_copy(src: nn.Module, dest: nn.Module, verbose=0) -> None:
    MODULE_TYPES = (
        torch.nn.Conv2d,
        torch.nn.Linear,
        torch.nn.BatchNorm2d,
        torch.nn.MultiheadAttention,
        torch.nn.LayerNorm,
        ClassTokenLayer,
        PosEmbeddingLayer
    )

    def find_instance_type(obj, *types):
        for t in types:
            if isinstance(obj, t):
                return t
        return None

    dest_iter = iter(dest.named_modules())

    last_copied_from = None
    last_copied_to: nn.Module = None

    for src_name, src_module in src.named_modules():
        src_is_flexible = isinstance(src_module, Module)
        if src_is_flexible:
            src_instance_type = src_module.base_type()
        else:
            src_instance_type = find_instance_type(src_module, *MODULE_TYPES)
            if src_instance_type is None:
                if verbose >= 2:
                    print(f"Skip copying layer {src_name}")
                continue

        if last_copied_from is not None and src_module in last_copied_from.modules():
            if verbose >= 2:
                print(f"Skip copying layer {src_name}")
            continue

        while True:
            dest_name, dest_module = next(dest_iter)
            dest_is_flexible = isinstance(dest_module, Module)

            if dest_is_flexible:
                if src_instance_type != dest_module.base_type():
                    if verbose >= 2:
                        print(f"Cannot copy {src_name} to {dest_name}")
                    continue
            else:
                if not isinstance(dest_module, src_instance_type):
                    if verbose >= 2:
                        print(f"Cannot copy {src_name} to {dest_name}")
                    continue

            if last_copied_to is not None and dest_module in last_copied_to.modules():
                continue

            if verbose >= 1:
                print(f"copy from {src_name} to {dest_name}")
            if src_is_flexible:
                if dest_is_flexible:
                    dest_module.load_from_base(src_module.make_base_copy())
                else:
                    src_module.copy_to_base(dest_module)
            else:
                if dest_is_flexible:
                    dest_module.load_from_base(src_module)
                else:
                    dest_module.load_state_dict(src_module.state_dict())
            last_copied_to = dest_module
            break

        last_copied_from = src_module


def save_model(model: nn.Module, model_description: str, prefix: str = '') -> None:
    with open(paths.TRAINED_MODELS / f"{model_description}.pth", "wb") as file:
        torch.save(model, file)


def load_model(model_description: str, prefix: str = '') -> nn.Module:
    with open(paths.TRAINED_MODELS / f"{model_description}.pth", "rb") as file:
        return torch.load(file, weights_only=False)
