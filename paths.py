import os
from pathlib import Path


ON_CLUSTER = 'SLURM_JOBID' in os.environ

USER = os.environ.get('USER', default='user')
HOSTNAME = os.environ.get('HOSTNAME', default='localhost')
HOME = os.environ.get('HOME', default='.')
PWD = os.environ.get('PWD', default='.')
TMPDIR = os.environ.get('TMPDIR', default='./tmp')
SLURM_SUBMIT_DIR = os.environ.get('SLURM_SUBMIT_DIR', default='.')
SLURM_JOBID = int(os.environ.get('SLURM_JOBID', default='0'))
SLURM_JOB_NAME = os.environ.get('SLURM_JOB_NAME', default='job')

SLURM_NODELIST = os.environ.get('SLURM_NODELIST', default=None)
SLURM_ARRAY_TASK_ID = os.environ.get('SLURM_ARRAY_TASK_ID', default=None)
SLURM_NTASKS = os.environ.get('SLURM_NTASKS', default=None)
SLURM_NTASKS_PER_NODE = os.environ.get('SLURM_NTASKS_PER_NODE', default=None)


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
