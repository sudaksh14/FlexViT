from pathlib import Path
import os


def make_path(path: str) -> Path:
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    return Path(path)


def make_file(path: str) -> Path:
    make_path(os.path.dirname(path))
    return Path(path)


ON_CLUSTER = 'SLURM_JOBID' in os.environ

HOME = os.environ.get('HOME', default='.')
TMPDIR = os.environ.get('TMPDIR', default='./tmp')
SLURM_JOBID = int(os.environ.get('SLURM_JOBID', default='0'))

PROJECT_DIR = make_path(HOME if ON_CLUSTER else './.tmp')
TMPDIR = make_path(TMPDIR if ON_CLUSTER else (PROJECT_DIR / 'temp'))

JOBDIR = make_path(f"{PROJECT_DIR}/jobs/job_{SLURM_JOBID}")

DATA_PATH = make_path(f"{PROJECT_DIR}/data")
LOG_PATH = make_path(JOBDIR / 'logs')

TRAINED_MODELS = make_path(PROJECT_DIR / 'pretrained')

FIGURES = make_path(PROJECT_DIR / 'figures')

IMAGENET_PATH = make_path(
    "/scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder"
    if ON_CLUSTER else (
        DATA_PATH / 'imagenet'))
