from networks.flex_model import FlexModel
import utils


class ModelConfig(utils.SelfDescripting):
    def make_model(self) -> FlexModel:
        raise NotImplementedError()

    def no_prebuilt(self) -> 'ModelConfig':
        raise NotImplementedError()


class FlexModelConfig(ModelConfig):
    def create_base_config(self, level) -> ModelConfig:
        raise NotImplementedError()

    def max_level(self) -> int:
        raise NotImplementedError()
