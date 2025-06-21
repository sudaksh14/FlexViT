import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score
import os

from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets import ImageFolder

from torchvision.transforms import (
    Compose, RandomResizedCrop, RandomHorizontalFlip, RandomRotation,
    ColorJitter, ToTensor, Normalize, Resize, CenterCrop
)

from torch.utils.data import random_split, DataLoader
import shutil

from dataclasses import dataclass

import paths

import adapt_modules as am


def get_device():
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

    def get_filename_safe_description(self):
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


def evaluate_model(model, dataloader, device):
    """
    Evaluates the model on the given dataloader and returns accuracy and F1 score.
    """
    all_preds = []
    all_labels = []
    # Move model to the correct device and ensure correct data type
    model = model.to(device).to(torch.float32)
    model.eval()

    with torch.no_grad():
        for images, labels in dataloader:
            # Ensure images are on the same device and data type
            images = images.to(device).to(torch.float32)
            labels = labels.to(device)

            outputs = model(images)  # Perform forward pass
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy


def count_parameters(model):
    """Counts the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_in_mb(model):
    torch.save(model.state_dict(), "temp.p")
    size_mb = os.path.getsize("temp.p") / (1024 * 1024)
    os.remove("temp.p")
    return size_mb


def dummy_data(data_dir=paths.DATA_PATH):
    val_split = 0.2
    batch_size = 64

    train_transform = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=15),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CIFAR10(root=data_dir, train=True,
                            download=True, transform=train_transform)
    test_dataset = CIFAR10(root=data_dir, train=False,
                           download=True, transform=test_transform)

    # Split the train dataset into train/val
    # train_size = int((1 - val_split) * len(train_dataset))
    # val_size = len(train_dataset) - train_size

    train_dataset, val_dataset, _ = random_split(
        train_dataset, [100 * 8, 8 * 8, len(train_dataset) - 108 * 8])

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=4)

    return train_dataloader, val_dataloader, test_dataloader


def load_data(data_dir=paths.DATA_PATH, tmp_dir=paths.TMPDIR, batch_size=64, val_split=0.2):
    # Data transformations for training
    train_transform = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=15),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Data transformations for validation/test
    test_transform = Compose([
        # Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

    if tmp_dir is not None:
        try:
            os.makedirs(tmp_dir)
        except FileExistsError:
            pass
        try:
            os.makedirs(data_dir)
        except FileExistsError:
            pass
        shutil.copytree(data_dir, tmp_dir, dirs_exist_ok=True)
    train_dataset = CIFAR10(root=data_dir if tmp_dir is None else tmp_dir, train=True,
                            download=True, transform=train_transform)
    test_dataset = CIFAR10(root=data_dir if tmp_dir is None else tmp_dir, train=False,
                           download=True, transform=test_transform)
    if tmp_dir is not None:
        shutil.copytree(tmp_dir, data_dir, dirs_exist_ok=True)

    # Split the train dataset into train/val
    # train_size = int((1 - val_split) * len(train_dataset))
    # val_size = len(train_dataset) - train_size
    # train_dataset, val_dataset = random_split(
    #     train_dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=4)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=4)

    return train_dataloader, val_dataloader, test_dataloader


def load_data100(data_dir=paths.DATA_PATH, tmp_dir=paths.TMPDIR, batch_size=64, val_split=0.2):
    # Data transformations for training
    train_transform = Compose([
        # RandomHorizontalFlip(p=0.5),
        # RandomRotation(degrees=15),
        # ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ToTensor(),
        Normalize(mean=[0.5070, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761])
    ])

    # Data transformations for validation/test
    test_transform = Compose([
        # Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.5070, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761])
    ])

    if tmp_dir is not None:
        try:
            os.makedirs(tmp_dir)
        except FileExistsError:
            pass
        try:
            os.makedirs(data_dir)
        except FileExistsError:
            pass
        shutil.copytree(data_dir, tmp_dir, dirs_exist_ok=True)
    train_dataset = CIFAR100(root=data_dir if tmp_dir is None else tmp_dir, train=True,
                             download=True, transform=train_transform)
    test_dataset = CIFAR100(root=data_dir if tmp_dir is None else tmp_dir, train=False,
                            download=True, transform=test_transform)
    if tmp_dir is not None:
        shutil.copytree(tmp_dir, data_dir, dirs_exist_ok=True)

    # Split the train dataset into train/val
    # train_size = int((1 - val_split) * len(train_dataset))
    # val_size = len(train_dataset) - train_size
    # train_dataset, val_dataset = random_split(
    #     train_dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=4)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=4)

    return train_dataloader, val_dataloader, test_dataloader


def load_imagenette(data_dir="./data/imagenette2",
                    batch_size=128,
                    val_split=0.1,
                    num_workers=8):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_tf = transforms.Compose([
        transforms.Resize(40),          # small augmentation
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        normalize,
    ])

    full_train = datasets.ImageFolder(f"{data_dir}/train", transform=train_tf)
    n_val = int(len(full_train) * val_split)
    train_ds, val_ds = random_split(full_train, [len(full_train)-n_val, n_val])
    test_ds = datasets.ImageFolder(f"{data_dir}/val", transform=val_tf)

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)

    return loader(train_ds, True), loader(val_ds, False), loader(test_ds, False)


def flexible_model_copy(src: nn.Module, dest: nn.Module, verbose=0):
    MODULE_TYPES = (
        torch.nn.Conv2d,
        torch.nn.Linear,
        torch.nn.BatchNorm2d,
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
        src_is_adaptable = isinstance(src_module, am.Module)
        if src_is_adaptable:
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
            dest_is_adaptable = isinstance(dest_module, am.Module)

            if dest_is_adaptable:
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
            if src_is_adaptable:
                if dest_is_adaptable:
                    dest_module.load_from_base(src_module.make_base_copy())
                else:
                    src_module.copy_to_base(dest_module)
            else:
                if dest_is_adaptable:
                    dest_module.load_from_base(src_module)
                else:
                    dest_module.load_state_dict(src_module.state_dict())
            last_copied_to = dest_module
            break

        last_copied_from = src_module


def save_model(model, model_description, prefix=''):
    with open(paths.TRAINED_MODELS / f"{model_description}.pth", "wb") as file:
        torch.save(model, file)


def load_model(model_description, prefix=''):
    with open(paths.TRAINED_MODELS / f"{model_description}.pth", "rb") as file:
        return torch.load(file, weights_only=False)
