import os
from pathlib import Path


ON_CLUSTER = 'SLURM_JOBID' in os.environ

USER = os.environ.get('USER', default='user')
HOSTNAME = os.environ.get('HOSTNAME', default='localhost')
HOME = os.environ.get('HOME', default='.')
PWD = os.environ.get('PWD', default='.')
TMPDIR = os.environ.get('TMPDIR', default='./tmp')
SLURM_JOBID = int(os.environ.get('SLURM_JOBID', default='0'))


def make_path(path: str) -> Path:
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    return Path(path)


def make_file(path: str) -> Path:
    make_path(os.path.dirname(path))
    return Path(path)


PROJECT_DIR = make_path(HOME if ON_CLUSTER else './.tmp')
TMPDIR = make_path(TMPDIR if ON_CLUSTER else './.tmp/temp')

JOBDIR = make_path(f"{PROJECT_DIR}/jobs/job_{SLURM_JOBID}")

DATA_PATH = make_path(f"{PROJECT_DIR}/data")
LOG_PATH = make_path(JOBDIR / 'logs')

TRAINED_MODELS = make_path(PROJECT_DIR / 'pretrained')

FIGURES = make_path(PROJECT_DIR / 'figures')

try:
    snellius_imagenet = "/scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder"
    IMAGENET_PATH = make_path(snellius_imagenet) if ON_CLUSTER else (
        DATA_PATH / 'imagenet')
except PermissionError:
    # Not on cluster
    IMAGENET_PATH = None
