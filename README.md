# Conditional Action Trees

This repository complements the "Conditional Actions Tree" paper.

arxiv link here soon!

![M_level_2](images/M_2.gif)
![M_level_2](images/Flat_4.gif)
![M_level_2](images/Ma_4.gif)

# Discord Community For Support

For any support questions please join the [Griddly Discord Community](https://discord.gg/xuR8Dsv)

## Install Griddly

These experiments use several custom griddly environments.

```
pip install griddly
```



## Install Dependencies for this experiment

First navigate to this directory then:

```
pip install -r requirements.txt
```

## :warning: Rllib < 1.4.0 :warning: 

The current 1.3.0 release of rllib has some bugs that are fixed in the latest RLLib master branch which can be found here:

pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-2.0.0.dev0-cp38-cp38-manylinux2014_x86_64.whl

### WandB Integration

To upload the results to your own WandB account, create a `.wandb_rc` file in your user directory that contains your WandB API key.
All results and videos will then be automatically uploaded.

## Running experiments

You can copy any of the following lines to run any of the experiments in the paper.

#### No Masking

```
rllib_baseline.py  --experiment-name="M" --yaml-file="clusters_po.yaml"
rllib_baseline.py  --experiment-name="MP" --yaml-file="clusters_po_with_push.yaml"
rllib_baseline.py  --experiment-name="MPS" --yaml-file="clusters_po_with_push_separate_colors.yaml"
rllib_baseline.py  --experiment-name="Ma" --yaml-file="clusters_po_with_push_units.yaml"
rllib_baseline.py  --experiment-name="MSa" --yaml-file="clusters_po_with_push_separate_colors_units.yaml"
```

#### Depth-2

```
rllib_baseline_flat.py  --experiment-name="M" --yaml-file="clusters_po.yaml"
rllib_baseline_flat.py  --experiment-name="MP" --yaml-file="clusters_po_with_push.yaml"
rllib_baseline_flat.py  --experiment-name="MPS" --yaml-file="clusters_po_with_push_separate_colors.yaml"
rllib_baseline_flat.py  --experiment-name="Ma" --yaml-file="clusters_po_with_push_units.yaml"
rllib_baseline_flat.py  --experiment-name="MSa" --yaml-file="clusters_po_with_push_separate_colors_units.yaml" 
```

#### CAT_CL + CAT_CD

Both runs in these experiments run consecutively using ray's `grid_search` method

```
rllib_conditional_actions.py  --experiment-name="M" --yaml-file="clusters_po.yaml"
rllib_conditional_actions.py  --experiment-name="MP" --yaml-file="clusters_po_with_push.yaml"
rllib_conditional_actions.py  --experiment-name="MPS" --yaml-file="clusters_po_with_push_separate_colors.yaml"
rllib_conditional_actions.py  --experiment-name="Ma" --yaml-file="clusters_po_with_push_units.yaml"
rllib_conditional_actions.py  --experiment-name="MSa" --yaml-file="clusters_po_with_push_separate_colors_units.yaml"
```


## Griddly + RLLib 

The experiments are performed using several custom RLLib classes:

### [ConditionalActionImpalaTrainer](https://github.com/Bam4d/Griddly/blob/develop/python/griddly/util/rllib/torch/conditional_actions/conditional_action_policy_trainer.py#L119)

Contains the code for setting up the mixin and the modified vtrace policy

### [ConditionalActionMixin](https://github.com/Bam4d/Griddly/blob/develop/python/griddly/util/rllib/torch/conditional_actions/conditional_action_mixin.py)

Overrides the typical policy rollout method to use the Conditional Action Trees when sampling actions

### [ConditionalActionVTraceTorchPolicy](https://github.com/Bam4d/Griddly/blob/develop/python/griddly/util/rllib/torch/conditional_actions/conditional_action_policy_trainer.py#L104)

Applies constructed masks to the vtrace policy

### [TorchConditionalMaskingExploration](https://github.com/Bam4d/Griddly/blob/develop/python/griddly/util/rllib/torch/conditional_actions/conditional_action_exploration.py)

Contains the tree traversal and mask creation code

## Environments

The 5 environments that are used for the paper are contained in this repository with filenames similar to `clusters_po....yaml`

They are all based on the `Clusters` environment which has full documentation [here](https://griddly.readthedocs.io/en/latest/games/Clusters/index.html)

## WandB Results

View all of the experiments, training results and videos [here](https://wandb.ai/chrisbam4d/conditional_action_trees)

