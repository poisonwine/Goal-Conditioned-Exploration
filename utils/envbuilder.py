from collections import defaultdict
import numpy as np
import myenvs
import gym
import re
from arguments import get_args
import envs.fetch as fetch_env
from myenvs.robosuite.UR5e import UR5eLift
from myenvs.maze.ant_maze import AntMazeEnv

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

_my_game_envs = defaultdict(set)
for env in myenvs.registry.all():
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _my_game_envs[env_type].add(env.id)

def get_env_type(env_name):
    env_id = env_name
    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    elif env_id in _my_game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        # my own env has higher priority
        for g, e in _my_game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id

def make_env(env_name):
    env_type, env_id = get_env_type(env_name)
    args = get_args()
    args.env_name= env_name
    # print(env_type,_my_game_envs.keys())
    if env_name == 'UR5PickAndPlace-v1':
        env = UR5eLift(robot_name='UR5e')
        eval_env = UR5eLift(robot_name='UR5e')
    elif env_name == 'AntMaze-v1':
        env = AntMazeEnv(variant='AntMaze-SR', eval=False)
        eval_env = AntMazeEnv(variant='AntMaze-SR', eval=True)
    elif env_type in _my_game_envs.keys() and (not args.goal_teacher):
        env = myenvs.make(env_id)
        env._max_episode_steps = env.spec.max_episode_steps
        eval_env = myenvs.make(env_id)
        eval_env._max_episode_steps = eval_env.spec.max_episode_steps
    elif args.env_name[:5] =='Fetch' and args.goal_teacher:
        env = fetch_env.make_env(args)
        eval_env = fetch_env.make_env(args)
    else:
        env = gym.make(env_id)
        env._max_episode_steps = env.spec.max_episode_steps
        eval_env = gym.make(env_id)
        eval_env._max_episode_steps = eval_env.spec.max_episode_steps
    return env, eval_env

# args = get_args()
# args.env_name = 'FetchThrowRubberBall-v0'
# env = make_env('FetchThrowRubberBall-v0')
# for i in range(50):
#     env.reset()
#     for j in range(50):
#         obs,r,done,info = env.step(np.random.randn(4))
#         print(info)
#         env.render()
