from networks.flex_model import AdaptModel
import utils

class ModelConfig(utils.SelfDescripting):
    def make_model(self) -> AdaptModel:
        raise NotImplementedError()
    
    def no_prebuilt(self) -> 'ModelConfig':
        raise NotImplementedError()

class AdaptConfig(ModelConfig):
    def create_base_config(self, level) -> ModelConfig:
        raise NotImplementedError()