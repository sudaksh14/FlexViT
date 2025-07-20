from networks.flex_model import FlexModel
import utils


class ModelConfig(utils.SelfDescripting):
    """
    ModelConfig defines an interface followed by configuration classes of all models. 
    """

    def make_model(self) -> FlexModel:
        """
        Constructs a model according to this config
        """
        raise NotImplementedError()

    def no_prebuilt(self) -> 'ModelConfig':
        """
        Turns of the loading of pretrained weights
        """
        raise NotImplementedError()


class FlexModelConfig(ModelConfig):
    """
    FlexModelConfig extends the ModelConfig interface.
    FlexModelConfig is followed by configuration classes of all flexible models. 
    """

    def create_base_config(self, level) -> ModelConfig:
        """
        Creates a regular model config for the non flexible counterpart of this model.
        """
        raise NotImplementedError()

    def max_level(self) -> int:
        """
        Queries the maximum level the flexible model will have.
        """
        raise NotImplementedError()
