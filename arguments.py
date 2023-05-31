import argparse
import click
import numpy as np
"""
Here are the param for the training

"""
def DDPG_config():
    configs = {'policy_decay':2,
               'max_grad_norm': None}
    return configs


def TD3_config():
    configs = {
        'gamma': 0.99,
        'epsilon_init': 1.0,
        'epsilon_min': 0.2,
        'noisy_range': 0.5,
        'policy_delay': 2,
        'policy_noise': 0.2,
        'min_a': -1,
        'max_a': 1,
        'OU_noise':True,
        'decay':2,
        'max_grad_norm': 1.0
    }
    return configs

def SAC_config():
    configs = {
        'auto_entropy_tune': False,
        'alpha': 0.001,
        'min_log_sigma':-20.0,
        'max_log_sigma': 2.0,
        'alpha_lr': 0.001,
        'policy_decay':2,
        'max_grad_norm': 1.0
    }
    return configs




def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env_name', type=str, default='FetchPickAndPlace-v1', help='the environment name')
    parser.add_argument('--agent', type=str,default='DDPG',choices=['DDPG','SAC','TD3'],help='policy agent')
    parser.add_argument('--alg', type=str, default='MINE', help='which algorithm to use')
    parser.add_argument('--critic_type', type=str, default='MRN', choices=['monolithic', 'BVN', 'MRN'], help='critic type')
    parser.add_argument('--n_epochs', type=int, default=200, help='the number of epochs to train the agent')
    parser.add_argument('--n_cycles', type=int, default=50, help='the times to collect samples per epoch')
    parser.add_argument('--num_rollouts_per_mpi', type=int, default=5, help='the rollouts per mpi')
    parser.add_argument('--n_batches', type=int, default=40, help='the times to update the network')
    parser.add_argument('--save_interval', type=int, default=5, help='the interval that save the models')
    parser.add_argument('--use_pretrain_policy', type=bool, default=False)
    

    parser.add_argument('--seed', type=int, default=100, help='random seed')
    parser.add_argument('--num_workers', type=int, default=1, help='the number of cpus to collect samples')
    parser.add_argument('--replay_strategy', type=str, default='future', help='the HER strategy', choices=(['future', 'final', 'episode','random']))
    parser.add_argument('--clip_return', type=float, default=50, help='if clip the returns')
    parser.add_argument('--save_dir', type=str, default='/data1/ydy/RL/HER_v3/saved_models', help='the path to save the models')
    parser.add_argument('--noise_eps', type=float, default=0.2, help='noise eps')
    parser.add_argument('--random_eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--buffer_size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay_k', type=int, default=4, help='ratio to be replace')
    parser.add_argument('--clip_obs', type=float, default=200, help='the clip ratio')
    parser.add_argument('--batch_size', type=int, default=256, help='the sample batch size')
    parser.add_argument('--beh_batch_size', type=int, default=256, help='selected hindsight goal')
    parser.add_argument('--actual_batch_size', type=int, default=0, help='actual desired goal')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--action_l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--lr_actor', type=float, default=0.001, help='the learning rate of the actor')
    parser.add_argument('--lr_critic', type=float, default=0.001, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--n_test_rollouts', type=int, default=10, help='the number of tests')
    parser.add_argument('--clip_range', type=float, default=5, help='the clip range')
    parser.add_argument('--demo_length', type=int, default=200, help='the demo length')
    parser.add_argument('--warm_up_steps', type=int, default=2500, help='warm up steps')
    parser.add_argument('--cuda', type=bool, default=False, help='if use gpu do the acceleration')

    parser.add_argument('--her', type=bool, default=True, help='whether to use her sampler')

    parser.add_argument('--buffer_priority', type=bool, default=False,
                        help='when buffer is full, prioritize to delete unsuccessful trajectory')
    parser.add_argument('--episode_priority', type=bool, default=False, help='whether to prioritize episode')
    parser.add_argument('--traj_rank_method', type=str, default='entropy', choices=(['entropy', 'sim_p', 'energy']), help='rank method')


    parser.add_argument('--density_method', type=str, default='kernel',choices=['kernel','flow','rnd'], help='whether to use density estimator to select hindsight goal')
    # parser.add_argument('--goal_explore', type=bool, default=False, help='whether to use candidate goal to explore')
    parser.add_argument('--explore_goal_num', type=int, default=200, help="explore goal number")


    # config for cher
    parser.add_argument('--use_cher', type=bool, default=False, help='whether to use cher sampler')
    # config for per
    parser.add_argument('--use_per', type=bool, default=False, help='whether to use td-error based PER')
    parser.add_argument('--alpha',type=float, default=0.6, help='PER default alpha')
    parser.add_argument('--beta', type=float, default=0.4, help='PER default alpha')



    # config for laber: Large Batch Experience Replay, https://arxiv.org/abs/2110.01528
    parser.add_argument('--use_laber', type=bool, default=False, help='whether to use LaBER to update agent')
    parser.add_argument('--m_factor', type=int, default=4, help='laber parameter')

    ##config for ger
    parser.add_argument('--use_ger', type=bool, default=False, help='whether to use ger')
    parser.add_argument('--n_GER', type=int, default=1, help='whether to use GER')
    parser.add_argument('--error_dis', type=float, default=0.05, help=' threshold ')
    parser.add_argument('--goal_arguement', type=bool, default=True, help='whether to use goal arguement')
    parser.add_argument('--argued_goal_size', type=int, default=256, help='arguemented goal size')

    ## config for mep
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature value for prioritization')
    parser.add_argument('--fit_interval', type=int, default=20, help='fit_interval')
    # parser.add_argument('--prioritization', type=click.Choice(['none', 'entropy', 'tderror']), default='entropy', help='the prioritization strategy to be used.')

    ## config for energy-based HER
    parser.add_argument('--w_potential', type=float, default=1.0, help='weight for potential energy')
    parser.add_argument('--w_linear', type=float, default=1.0, help='weight for linear energy')
    parser.add_argument('--w_rotational', type=float, default=1.0, help='weight for rotation energy')
    parser.add_argument('--clip_energy', type=float, default=999, help='clip_energy')

    # config for goal sampler teacher
    parser.add_argument('--goal_teacher',action='store_true', help='whether to use goal teacher sampler')
    parser.add_argument('--teacher_method', type=str, default='AIM', choices=['random', 'VDS', 'HGG','AGE','AIM'])
    # config for reward teacher
    parser.add_argument('--reward_teacher',action='store_true', help='whether to use reward teacher')
    parser.add_argument('--reward_method', type=str, default='mine',choices=['aim','icm','mine'])
    parser.add_argument('--intrinisic_r', action='store_true', help='whether to use intrinisic reward to replace sparse reward')
    # parser.add_argument('--goal_teacher',type=bool, default=False, help='whether to use goal teacher sampler')
    parser.add_argument('--mine_iter', type=int, default=40)
    parser.add_argument('--mine_batch', type=int, default=256)
    parser.add_argument('--reward_scale', type=float, default=100)

    parser.add_argument('--fit_rate', type=int, default=100, help='after how many episodes update the target V critic ')
   
    args, _ = parser.parse_known_args()
    #config for HGG
    parser.add_argument('--goal', help='method of goal generation', type=str, default='vanilla',choices=['vanilla', 'fixobj', 'interval'])
    parser.add_argument('--hgg_c', help='weight of initial distribution in flow learner', type=np.float32, default=3.0)
    parser.add_argument('--hgg_L', help='Lipschitz constant', type=np.float32, default=5.0)
    parser.add_argument('--hgg_pool_size', help='size of achieved trajectories pool', type=np.int32, default=1000)
    parser.add_argument('--pool_rule', help='rule of collecting achieved states', type=str, default='full',choices=['full', 'final'])
    parser.add_argument('--episodes', help='number of episodes per cycle', type=np.int32, default= int(args.n_cycles * args.num_rollouts_per_mpi))

    # config for VDS
    parser.add_argument('--vds_hidden_dim',type=int, default=128, help='VDS Q function hidden layer')
    parser.add_argument('--vds_layer_num', type=int, default=3, help='the number of VDS Q function mlp layers')
    parser.add_argument('--Q_num', type=int, default=4, help='the number of Q function')
    parser.add_argument('--vds_batches', type=int, default=40, help='the update number per epoch')
    parser.add_argument('--vds_batch_size', type=int, default=256, help='the update number per epoch')
    parser.add_argument('--vds_lr', type=float, default=1e-4, help='the Q ensemble learning rate')
    parser.add_argument('--n_candidate', type=int, default=200, help=' the number of goal candidates sampled to pass the value ensemble')
  
    parser.add_argument('--vds_gamma', type=float, default=0.95, help='the Q ensemble function gamma')

    # config for adversarial intricnsic motivation(AIMteacher)
    parser.add_argument('--aim_discriminator_steps',type=int, default=20, help='the step when update discriminator')
    parser.add_argument('--aim_reward_norm_offset',type=float, default=0.1, help='the reward offset')
    parser.add_argument('--aim_batch_size',type=int, default=256, help='batch size when update discriminator')
    parser.add_argument('--lambda_coef',type=float, default=10.0, help='lambda penalty')
    parser.add_argument('--aim_temperature',type=float, default=1.0, help='sample temperature')
    parser.add_argument('--sample_type',type=str, default='topk', help='sample temperature')

    # config for active goal exploration
    parser.add_argument('--q_cuttoff',action='store_true', help='q cuttoff')
    parser.add_argument('--goal_shift',action='store_true', help='g shifting')
    parser.add_argument('--shift_delta', type=float, default=0.5)
    parser.add_argument('--sample_stratage',type=str,default='MEGA',choices=['MEGA','Diverse','MinQ','MinV', 'RIG','VAD','LP','MEGA_VAD', 'MEGA_LP','MEGA_MinV'])
    parser.add_argument('--state_discover',action='store_true', help='state discover')
    parser.add_argument('--state_discover_method', type=str, default='prior',choices=['prior','icm','mine'])
    parser.add_argument('--age_lambda', type=float, default=0.5)
    parser.add_argument('--explore_alpha', type=float, default=0.5)
    parser.add_argument('--curriculum_select',action='store_true', help='curriculum select')
    parser.add_argument('--density_h', type=float, default=0.1)
    
    # config for icm teacher
    parser.add_argument('--icm_iteration',type=int, default=40, help='the step when update icm module')
    parser.add_argument('--icm_beta',type=float, default=0.2, help='the step when update discriminator')
    parser.add_argument('--icm_reward_scale',type=float, default=0.2, help='the step when update discriminator')
   


    # config for model based planning
    parser.add_argument('--mb_learning',action='store_true', help='if use model based method')
    parser.add_argument('--mb_batch_size', type=int, default=256, help='the sample batch size')
    parser.add_argument('--mb_update_steps', type=int, default=40, help='update the  transition model')
    parser.add_argument('--update_time_per_batch', type=int, default=3, help='update times when sample a batch')
    parser.add_argument('--obs_noise',type=float, default=0.1, help='obs noise when update the forward model')
    
    # config for BiLinear value network
    args, _ = parser.parse_known_args()

    if args.critic_type == 'BVN':
        parser.add_argument('--bvn_hidden_dim', type=int, default=3, help='BVN hidden dim')
    elif args.critic_type == 'MRN':
        parser.add_argument('--mrn_hidden_dim', type=int, default=128, help='MRN hidden dimension')
        parser.add_argument('--mrn_emb_dim', type=int, default=16, help='MRN embedding dimension')


    parser.add_argument('--save_data', type=bool, default=True, help='whether to save achieved goal data')


    args = parser.parse_args()


    return args
