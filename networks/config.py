from networks.adapt_model import AdaptModel
import utils

class ModelConfig(utils.SelfDescripting):
    def make_model(self) -> AdaptModel:
        raise NotImplemented()
    
    def no_prebuilt(self) -> 'ModelConfig':
        raise NotImplemented()

class AdaptConfig(ModelConfig):
    def create_base_config(self, level) -> ModelConfig:
        raise NotImplemented()