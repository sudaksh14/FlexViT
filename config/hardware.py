import dataclasses
import utils


@utils.fluent_setters
@dataclasses.dataclass
class HardwareConfig(utils.SelfDescripting):
    """
    This class stores the hardware configuration of the node(s) running an experiment.
    """
    partition: str = 'gpu_h100'
    gpu_count: int = 1
    time: str = '24:00:00'

    # do not change this as training code is currently not set up to handle multiple nodes
    node_count: int = 1

    def format_as_slurm_args(self) -> str:
        """
        This formats the configuration into arguments as passed to sbatch or srun.
        This is used to reserve nodes for experiments with their associated hardware requirements.
        """
        return f"-N {self.node_count} -p {self.partition} -t {self.time} --gpus-per-node={self.gpu_count} --mem=0"


class CurrentDevice:
    hardware: HardwareConfig = None

    @staticmethod
    def set_hardware(hw: HardwareConfig):
        CurrentDevice.hardware = hw

    @staticmethod
    def get_hardware() -> HardwareConfig:
        return CurrentDevice.hardware


DEFAULT_HARDWARE_CONFIG = HardwareConfig()
HARDWARE = {
    "vitprebuild": HardwareConfig().set_gpu_count(4),
    "flexvit": {
        "cifar10.5levels": HardwareConfig().set_gpu_count(4).set_time('12:00:00'),
        "imagenet": HardwareConfig().set_gpu_count(4).set_time('96:00:00')
    },
    "flexvitcorrect": HardwareConfig().set_gpu_count(4).set_time('24:00:00'),
}
