import numpy as np
import gym
import os, sys
from arguments import get_args, TD3_config, SAC_config,DDPG_config
from mpi4py import MPI
from rl_modules.algorithm.DDPG import ddpg_agent
from rl_modules.algorithm.TD3 import td3_agent
from rl_modules.algorithm.SAC import sac_agent
from rl_modules.replay_buffer import ReplayBuffer, TrajectoryPriorityBuffer, TransitionPriorityBuffer, goal_replay_buffer, goal_buffer
from her_modules.her import RandomSampler, HERSampler, PrioritizedSampler, PrioritizedHERSampler
from mpi_utils.normalizer import normalizer
from rl_modules.rollouts import Rollouts
from rl_modules.evaluator import eval_agent
from rl_modules.goal_sampler import GoalSampler
from rl_modules.density import KernalDensityEstimator, FlowDensity, RNDDensity
from her_modules.goal_curriculum import GoalAugmentor
from rl_modules.curriculum_agent import Curricum_Agent
import random
import torch
from tensorboardX import SummaryWriter
from utils.envbuilder import make_env
import json
from rl_modules.teachers.VDS.vds import VDSteacher
from rl_modules.teachers.HGG.hgg import HGGteacher
from rl_modules.teachers.AGE.ageteacher import AGETeacher
from rl_modules.teachers.AIM.aim import AIMTeacher
from rl_modules.teachers.MINE.mine import MineTeacher
from rl_modules.teachers.ICM.icm import ICMTeacher
import wandb

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""



def get_env_params(env, args):
    obs = env.reset()
    # close the environment
    params = {'obs': int(obs['observation'].shape[0]),
              'goal': int(obs['desired_goal'].shape[0]),
              'action': int(env.action_space.shape[0]),
              'action_max': int(env.action_space.high[0])}
    # print(params)
    params['max_timesteps'] = int(env._max_episode_steps)
    params['env_name'] = args.env_name
    params['replay_strategy'] = args.replay_strategy
    params['seed'] = args.seed
    # save_path = os.path.join('saved_models', args.env_name, args.alg)
    # os.makedirs(save_path, exist_ok=True)
    # with open(os.path.join(save_path, 'params_'+ str(args.seed) + '.json'), 'w') as f:
    #     f.write(json.dumps(params))
    params['env_steps'] = 0
    return params

def config_sampler(args, env_params, reward_func):


    if args.her:
        if args.use_per:
            sampler = PrioritizedHERSampler(T=env_params['max_timesteps'], args=args, reward_func=reward_func, alpha=args.alpha,
                                            beta=args.beta,replay_strategy=args.replay_strategy, replay_k=args.replay_k)
        else:
        
            sampler = HERSampler(replay_strategy=args.replay_strategy, replay_k=args.replay_k, args=args, reward_func=reward_func)
    else:
        if args.use_per:
            sampler = PrioritizedSampler(T=env_params['max_timesteps'], args=args, reward_func=reward_func,alpha=args.alpha, beta=args.beta)
        else:
            sampler = RandomSampler(args=args, reward_func=reward_func)

    return sampler


def config_replay_buffer(env_params, sampler, args):
    if args.episode_priority and not args.use_per:
        replaybuffer = TrajectoryPriorityBuffer(env_params=env_params, sampler=sampler,args=args, priority=args.traj_rank_method)

    elif args.use_per:
        replaybuffer = TransitionPriorityBuffer(env_params=env_params, sampler=sampler,args=args, priority=args.traj_rank_method)

    else:
        replaybuffer = ReplayBuffer(env_params=env_params, args=args, sampler=sampler)

    return replaybuffer

def config_train_agent(args, env, env_params,replaybuffer, o_norm, g_norm, reward_teacher):
    if args.agent == 'DDPG':
        ddpg_config = DDPG_config()
        agent = ddpg_agent(env, env_params, ddpg_config, args, replaybuffer, o_norm, g_norm, reward_teacher)
    elif args.agent == 'TD3':
        td3_configs = TD3_config()
        agent = td3_agent(env, env_params, td3_configs, args, replaybuffer, o_norm, g_norm, reward_teacher)
    elif args.agent == 'SAC':
        sac_configs = SAC_config()
        agent = sac_agent(env, env_params, sac_configs, args, replaybuffer, o_norm, g_norm, reward_teacher)
    return agent

def config_density_model(args):
    pass






def launch(args):
    # create the ddpg_agent
    env, eval_env = make_env(args.env_name)
    # set random seeds for reproduce
  
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    eval_env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # get the environment parameters
    env_params = get_env_params(env, args)
    rank = MPI.COMM_WORLD.Get_rank()
    
    comment = str(rank)+'-' + args.agent + '-' + args.env_name + '-' + args.alg + '-seed' + str(args.seed) + '-' + 'goal_explore-' + str(args.goal_teacher)
    logger = SummaryWriter(comment=comment)
    run_name = args.env_name +'-' + args.alg+'-'+ args.agent +'-seed-'+ str(args.seed) 
    if MPI.COMM_WORLD.Get_rank() == 0:
        wandb.init(project='GoalExplore', name=run_name, group=args.env_name, config=args)
    o_norm = normalizer(size=env_params['obs'], default_clip_range=args.clip_range)
    g_norm = normalizer(size=env_params['goal'], default_clip_range=args.clip_range)

    Sampler = config_sampler(args=args, env_params=env_params, reward_func=env.compute_reward)
    replaybuffer = config_replay_buffer(env_params=env_params, sampler=Sampler,args=args)
    candidate_ags_buffer = goal_buffer(name='candidate_ags', buffer_size=8000, sample_dim=env_params['goal'])
    candidate_dgs_buffer = goal_buffer(name='candidate_dgs', buffer_size=8000, sample_dim=env_params['goal'])

    if args.reward_teacher:
        if args.reward_method == 'aim':
            reward_teacher = AIMTeacher(args=args, env_param=env_params, g_norm=g_norm, buffer=replaybuffer, ag_buffer=candidate_ags_buffer)
        elif args.reward_method == 'mine':
            reward_teacher = MineTeacher(args=args, env_params=env_params, buffer=replaybuffer, o_norm=o_norm, g_norm=g_norm)
        elif args.reward_method == 'icm':
            reward_teacher = ICMTeacher(args=args, env_params=env_params, o_norm=o_norm, g_norm=g_norm, buffer=replaybuffer)

    else:
        reward_teacher = None

    agent = config_train_agent(args, env, env_params, replaybuffer, o_norm, g_norm, reward_teacher)
    evalagent = eval_agent(eval_env, agent, args, env_params, o_norm, g_norm)
    ## config for density_estimator
    if args.density_method == 'kernel':
        ag_density_estimator = KernalDensityEstimator(name='achieved_goal',
                                                    args=args,
                                                    logger=logger,
                                                    buffer=candidate_ags_buffer,
                                                    bandwidth=args.density_h)



        dg_density_estimator = KernalDensityEstimator(name='desired_goal',
                                                    args=args,
                                                    logger=logger,
                                                    buffer=candidate_dgs_buffer,
                                                    bandwidth=args.density_h)
    elif args.density_method == 'flow':
        ag_density_estimator = FlowDensity(name='achieved_goal', logger=logger, buffer=candidate_ags_buffer)
        dg_density_estimator = FlowDensity(name='desired_goal', logger=logger, buffer=candidate_dgs_buffer)
    elif args.density_method == 'rnd':
        ag_density_estimator = RNDDensity(name='achieved_goal', logger=logger, buffer=candidate_ags_buffer, lr=1e-3)
        dg_density_estimator = RNDDensity(name='desired_goal', logger=logger, buffer=candidate_dgs_buffer, lr=1e-3)

    
    if args.goal_teacher:
        if args.teacher_method == 'VDS':
            goal_teacher = VDSteacher(args=args,
                                env_params=env_params,
                                env=env,
                                buffer=replaybuffer,
                                ag_buffer=candidate_ags_buffer,
                                dg_buffer=candidate_dgs_buffer,
                                policy=agent,
                                o_norm=o_norm,
                                g_norm=g_norm,
                                )
        elif args.teacher_method == 'HGG':
            goal_teacher = HGGteacher(args=args,
                                    env_params=env_params,
                                    env=env,
                                    policy=agent,
                                    o_norm=o_norm,
                                    g_norm=g_norm,
                                    ag_buffer=candidate_ags_buffer)

        elif args.teacher_method == 'AGE':
           
            goal_teacher = AGETeacher(args=args, env=env,env_params=env_params, density_estimator=ag_density_estimator, policy=agent,
                                    o_norm=o_norm , g_norm=g_norm, buffer=replaybuffer,dg_buffer=candidate_dgs_buffer, ag_buffer=candidate_ags_buffer,
                                    state_discover_module=reward_teacher)
                                    
        elif args.teacher_method == 'AIM':
            goal_teacher = AIMTeacher(args=args, env_param=env_params, g_norm=g_norm, buffer=replaybuffer, ag_buffer=candidate_ags_buffer)
    
        elif args.teacher_method == 'Random':
            goal_teacher = None

    else:
        goal_teacher = None

    goal_sampler = GoalSampler(args=args,
                               env=env,
                               ag_estimator=ag_density_estimator,
                               dg_estimator=dg_density_estimator,
                               ag_buffer=candidate_ags_buffer,
                               dg_buffer=candidate_dgs_buffer,
                               replay_buffer=replaybuffer,
                               policy=agent,
                               teacher=goal_teacher,
                               o_norm=o_norm,
                               g_norm=g_norm,
                               evaluator=evalagent)


    GoalAgmentor = GoalAugmentor(env_name=args.env_name, error_distance=args.error_dis, batch_size=args.argued_goal_size)

    rollout_worker = Rollouts(env, env_params, o_norm, g_norm, agent, args, goal_sampler=goal_sampler)

    C_agent = Curricum_Agent(args=args,
                             policy=agent,
                             eval_agent=evalagent,
                             env=env,
                             env_params=env_params,
                             buffer=replaybuffer,
                             rollout_worker=rollout_worker,
                             her_sampler=Sampler,
                             o_norm=o_norm,
                             g_norm=g_norm,
                             ag_density_estimator=ag_density_estimator,
                             dg_density_estimator=dg_density_estimator,
                             ag_buffer=candidate_ags_buffer,
                             dg_buffer=candidate_dgs_buffer,
                             goal_sampler=goal_sampler,
                             goal_argumentor=GoalAgmentor,
                             goal_teacher=goal_teacher,
                             reward_teacher=reward_teacher,
                             logger=logger,
                             )

    C_agent.learn()

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    # run_name = args.alg + '-seed-'
    save_root = os.path.join(args.save_dir, args.env_name, args.alg, 'seed-'+str(args.seed))
    os.makedirs(save_root, exist_ok=True)
    with open(os.path.join(save_root, 'params.json'),'w') as f:
        json.dump(args.__dict__, f, indent=4)
    print('goal_teacher',args.goal_teacher, 'reward_teacher',args.reward_teacher)
    launch(args)

