from torch import nn


class Module(nn.Module):
    def set_level_use(self, level: int):
        raise NotImplemented()

    def current_level(self) -> int:
        raise NotImplemented()

    def max_level(self) -> int:
        raise NotImplemented()

    def base_type(self):
        raise NotImplemented()

    def copy_to_base(self, dest: nn.Module):
        raise NotImplemented()

    def load_from_base(self, src: nn.Module):
        raise NotImplemented()
