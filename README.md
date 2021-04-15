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

### :warning: Rllib < 1.3.0 :warning: 

The current master of rllib has some bugs that are fixed on our own RLLib branch which can be found here:

```
git clone git@github.com:Bam4d/ray.git
```


## Environments

The 5 environments that are used for the paper are contained in this repository with filenames similar to `clusters_po....yaml`

They are all based on the `Clusters` environment which has full documentation [here](https://griddly.readthedocs.io/en/latest/games/Clusters/index.html)

## Wandb

View all of the experiments, training results and videos [here](https://wandb.ai/chrisbam4d/conditional_action_trees)
