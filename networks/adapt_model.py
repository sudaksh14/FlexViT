from torch import nn
import adapt_modules as am


class AdaptModel(nn.Module):
    def set_level_use(self, level) -> None:
        self.level = level
        for _, module in self.named_modules():
            if isinstance(module, am.Module):
                module.set_level_use(level)

    def current_level(self) -> int:
        raise NotImplemented()

    def max_level(self) -> int:
        raise NotImplemented()
