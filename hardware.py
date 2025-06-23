import utils
import dataclasses


@utils.fluent_setters
@dataclasses.dataclass
class HardwareConfig(utils.SelfDescripting):
    partition = 'gpu_h100'
    gpu_count = 1
    time = '24:00:00'
    node_count = 1

    def format_as_slurm_args(self) -> str:
        return f"-N {self.node_count} -p {self.partition} -t {self.time} --gpus-per-node={self.gpu_count}"
