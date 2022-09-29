# Automatic Data Augmentation via invariance Constrained Learning

This is the implementation of the algorithm and experiments described on the paper.

It extends the Trivial Augment codebase to accomodate our method. we have made the following changes:

* Implemented our primal-dual algorithm in a new train script.

* Modified dataloaders to keep track and log sampled transformations.

* Added experiment logging to weights and biases.

## Installation

We require a working PyTorch version with GPU support,

as the TrivialAugment codebase only supports setups with at least one CUDA device. Note that we added some dependencies with respect to TrivialAugment.

Other requirements can be installed by running:

```

pip install -r requirements.txt

```

## Running experiments

As in trivial augment hyperparameters are loaded from yaml configuration files.
To run a particular experiment call:
```
python -m TrivialAugment.train_pd_batch -c conf/{CONFIG_FILE}.yaml
```
We have added the following parameters to config file for our algorithm:

- MH:
  - steps(int): Number of Metropolis hastings steps used to sample the worst case distribution ($\lambda^{\star}_c$)
- PD:
  lr: Dual Learning Rate
  margin: ($\epsilon$)

## Paper Experiments

### Config Files
Configuration files with all hyperparameters can be found on the folder `config`.
To generate variants corresponding to an experiment (e.g. different seeds or margins) call:
```
python create_variants_of_set_config.py {Configuration file path} 1
```

Experiments are grouped in the following folders:

 - Ablations
	 - Margins: Constraint Level ablation.
	 - Steps: Number of steps Ablation.
	 - Uniform: Uniform augmentation constraint ablation.
- Results: Corresponds to the table in the body of the paper.

### Bash Scripts
Bash scripts for launching experiments  on the paper can be found on the folder `scripts`.  It follows the same structure as config files.

Due to our particular hardware setup, in multiple GPU environments we only use one GPU, which we specify through the environment variable LOCAL_RANK, which can be set by running:

```
export LOCAL_RANK
```
Then you can simply run the corresponding script.
