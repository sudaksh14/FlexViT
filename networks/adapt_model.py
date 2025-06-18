from torch import nn

class AdaptModel(nn.Module):
    def set_level_use(self, level):
        raise NotImplemented()
    
    def current_level(self) -> int:
        raise NotImplemented()

    def max_level(self) -> int:
        raise NotImplemented()