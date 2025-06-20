from networks.adapt_model import AdaptModel
import utils

class ModelConfig(utils.SelfDescripting):
    def make_model(self) -> AdaptModel:
        raise NotImplemented()