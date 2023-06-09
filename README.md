# Goal-Conditioned Exploration Framework

This repo is established for goal-conditioned exploration  in multi-goal robotic environments. We aim to provide a modular and readable framework.

The fundamental algorithm is [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) (HER) , and we have extended the HER algorithm from the following aspects:
- Exploration Goal Selection: how to set exploration goal at the beginning of an episode

- Transition Selection: how to select more valueable transitions to replay

- Intrinsic Reward: use priors to accelerate learning

Everyone is welcome to suggest and contribute.



## Requirements
This repo only support Ubuntu system, before running the code, install the OpenMPI
```
sudo apt-get update && sudo apt-get install libopenmpi-dev 
```
Other packages
- python==3.6

- gym==0.15.4

- mpi4py==3.1.3

- torch==1.8.1

- mujoco-py==2.1.2.14

- wandb==0.13.10


## Main Module

### Arguements
All parameters of different algorithms are in [arguements.py](./arguments.py), please check the parameters before running algorithms.

Then you can run [train.py](./train.py) for training. 

### Goal Teacher - Exploration Goal Selection
Many algorithms set intrinsic goals to help robot to exploration in hard-exploraion environments. This module contains the goal selection algorithms.
- Supported Algorithms

    ✅ [HGG](https://arxiv.org/abs/1906.04279) 
     ([rl_modules/teachers/HGG](./rl_modules/teachers/HGG/)) 

    ✅ [VDS](https://arxiv.org/abs/2006.09641) 
    ([rl_modules/teachers/VDS](./rl_modules/teachers/VDS/))

    ✅ [AIM](https://arxiv.org/abs/2105.13345)
     ([rl_modules/teachers/AIM](./rl_modules/teachers/AIM/))

The AGE(Active Goal Exploration) module contains different goal sampling stratagies, please see [ageteacher.py](https://github.com/poisonwine/Goal-Conditioned-Exploration/blob/master/rl_modules/teachers/AGE/ageteacher.py) for detailed information.

- Supported goal sampling stratagy

    ✅ [MEGA](https://arxiv.org/abs/2007.02832)
    ([rl_modules/teahcers/AGE](./rl_modules/teachers/AGE/))

    ✅ [RIG](https://arxiv.org/abs/1807.04742) 
    ([rl_modules/teachers/AGE](./rl_modules/teachers/AGE/))

    ✅ [MinQ](https://arxiv.org/abs/1907.08225) 
    ([rl_modules/teachers/AGE](./rl_modules/teachers/AGE/))

    ✅ [LP](https://openreview.net/forum?id=Vk9RH9aL1Yv)
    ([rl_modules/teachers/AGE](./rl_modules/teachers/AGE/))

    ✅ [Diverse](https://arxiv.org/abs/1903.03698)
     ([rl_modules/teachers/AGE](./rl_modules/teachers/AGE/))

    ✅ [DEST](./rl_modules/teachers/AGE/) (Ours)

Some examples of running commands:
```python
# HGG
mpirun -np 6 python -u train.py --env_name FetchPushMiddleGap-v1 --agent DDPG --n_epochs 100 --seed 5   --alg HGG --goal_teacher --teacher_method HGG  

# VDS
mpirun -np 6 python -u train.py --env_name FetchPushMiddleGap-v1  --agent DDPG --n_epochs 100 --seed 5   --alg VDS --goal_teacher --teacher_method VDS  

# AIM 
mpirun -np 6 python -u train.py --env_name FetchPushMiddleGap-v1  --agent DDPG --n_epochs 100 --seed 5   --alg AIM --goal_teacher --teacher_method AIM  

#MEGA/MinQ/RIG/Diverse
mpirun -np 6 python -u train.py --env_name FetchPushMiddleGap-v1  --agent DDPG --n_epochs 100 --seed 5   --alg MEGA/MinQ/RIG/Diverse --goal_teacher --teacher_method AGE  --sample_stratage MEGA/MinQ/RIG/Diverse

# DEST
mpirun -np 6 python -u train.py --env_name FetchPushMiddleGap-v1  --agent DDPG --n_epochs 100 --seed 5  --explore_alpha 0.5 --alg DEST --goal_teacher --teacher_method AGE --sample_stratage MEGA_MinV --goal_shift   --state_discover_method mine --state_discover --reward_teacher --reward_method mine --age_lambda 0.2  

```

### Reward Teacher - Intrinsic Reward 
Use intrinsic reward to score goals or help robot learning.
 - Supported Algorithms

    ✅ [MINE](https://arxiv.org/abs/2103.08107)  (rl_modules/teachers/MINE)

    ✅ [RND](https://arxiv.org/abs/1810.12894) (rl_modules/teachers/RND)

    ✅ [ICM](https://arxiv.org/abs/1705.05363)(rl_modules/teahcers/ICM)

    ✅ [AIM](https://arxiv.org/abs/2105.13345) (rl_modules/teachers/AIM)

Some examples of running commands:
```python
# MINE/AIM/ICM
mpirun -np 6 python -u train.py --env_name FetchPushMiddleGap-v1  --agent DDPG --n_epochs 100 --seed 5  --alg MINE  --reward_teacher --reward_method mine/aim/icm --intrinisic_r
```


### Transition Selection
Different transition selection algorithms,including

✅ [CHER](https://dl.acm.org/doi/10.5555/3454287.3455418)
✅ [MEP](https://arxiv.org/abs/1905.08786v1)
✅ [EB-HER](https://arxiv.org/abs/1810.01363)
✅ [PER](https://arxiv.org/abs/1511.05952)
✅ [LABER](https://arxiv.org/abs/2110.01528)

Running command
```python 
# CHER
mpirun -np 6 python -u train.py --env_name FetchPushMiddleGap-v1  --agent DDPG --n_epochs 100 --seed 5  --alg CHER  --use_cher True

# PER
mpirun -np 6 python -u train.py --env_name FetchPushMiddleGap-v1  --agent DDPG --n_epochs 100 --seed 5  --alg PER  --use_per True

# LABER
mpirun -np 6 python -u train.py --env_name FetchPushMiddleGap-v1  --agent DDPG --n_epochs 100 --seed 5  --alg LABER  --use_laber True

# MEP/EB-HER
mpirun -np 6 python -u train.py --env_name FetchPushMiddleGap-v1  --agent DDPG --n_epochs 100 --seed 5  --alg MEP/EB-HER  --episode_priority True --traj_rank_method entropy/energy
```

### Reinforcement Learning Algorithm
We implement three common reinforment learning  algorithms in robot learning, including DDPG、 TD3 and SAC. Besides, we implement three types of critic network architecture, including [monolithic](https://github.com/poisonwine/Goal-Conditioned-Exploration/blob/master/rl_modules/models.py)、[BVN](https://arxiv.org/abs/2204.13695) and [MRN](https://arxiv.org/abs/2208.08133).

Examples of Running commands
```python 
# DDPG/SAC/TD3
mpirun -np 6 python -u train.py --env_name FetchPushMiddleGap-v1  --agent DDPG/SAC/TD3 --n_epochs 100 --seed 5  --alg HER

# different critic type 
mpirun -np 6 python -u train.py --env_name FetchPushMiddleGap-v1  --agent DDPG/SAC/TD3 --n_epochs 100 --seed 5  --alg HER --critic_type monolithic/BVN/MRN
```
### Envs
Our repo contains plenty of  goal-conditioned robot envs, please see [myenvs](https://github.com/poisonwine/Goal-Conditioned-Exploration/tree/master/myenvs/__init__.py). 

There are six most difficult envs where  the desired goals and block initial position have a huge gap, as the following figure shows.
![hard_envs](./fetchenv_hard.png)


## Results
The training results will be saved as the following structure
```
—— saved_models
—————— alg
———————— seed-5
———————————— progress_5.csv
———————————— models
———————— seed-6
———————————— progress_6.csv
———————————— models
```

When you want to plot results, see [utils/plot.py](./utils/plot.py). The defalut results structure is as follows:
```
—— results
—————— alg1
———————— progress_5.csv
———————— progress_6.csv
—————— alg2
———————— progress_5.csv
———————— progress_6.csv
```

## Some ToDos

- Model-based Planning
    - [MHER](https://arxiv.org/abs/2107.00306)
    - [PEG](https://openreview.net/forum?id=6qeBuZSo7Pr)
    - [MapGo](https://arxiv.org/abs/2105.06350)
    
- Other curiosity and curriculum learning methods






## Acknowledgement:
We borrowed some code from the following repositories:
- [Pytorch DDPG-HER Implementation](https://github.com/TianhongDai/hindsight-experience-replay)

- [CURROT](https://github.com/psclklnk/currot/tree/main)

- [Exploration-baseline](https://github.com/yuanmingqi/rl-exploration-baselines)

- [Modular-HER](https://github.com/YangRui2015/Modular_HER)
