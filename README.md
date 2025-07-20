# FlexViT

## Replicating thesis results with SLURM
First make sure you have the entire repository on a compute cluster with SLURM installed.

### Configuration
To run this on slurm you will probably first have to adjust some configurations. First of all you can find hardware configurations in the file `config/hardware.py`, where you can change the partition, amount of gpu's you want to allocate, and reservation time. Don't change the number of nodes, as support for multiple nodes is not added.

### WandB
For logging results this project uses the platform [Weights & Biases](https://wandb.ai). So you will need this installed and you need to be logged in via the CLI. By default results will be logged to project "FlexViT", but you can change this in `config/wandb.py`.

### Paths
All paths used by FlexViT can be found and changed in `config/paths.py`. Take special care in making sure `IMAGENET_PATH` points to where you have an unpacked image folder of ImageNet-1k. If you running this on a large shared compute cluster there is probably a shared copy of it already there, otherwise you will have to download it yourself.

### Running experiments
First start the imagenet and cifar10 prebuilt experiment.

    ./runall.sh flexvit,imagenet
    ./runall.sh vitprebuild,cifar10

After `vitprebuild,cifar10` is finished you can start the FlexViT CIFAR10 experiment.

    ./runall.sh flexvit,cifar10.5levels

After these are all finished run `plot.py` on a node with a gpu

    srun -N 1 -p PARTITION_HERE -t 1:00:00 --gpus-per-node=1 --pty plot.py


