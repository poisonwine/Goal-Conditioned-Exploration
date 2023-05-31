import os
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as p3d
from mpl_toolkits.mplot3d import Axes3D
import argparse
import numpy as np
from rl_modules.density import KernalDensityEstimator
from tensorboardX import SummaryWriter
import json
import torch
from utils.envbuilder import make_env
from rl_modules.models import MRNCritic, V_critic, actor
from arguments import get_args
import matplotlib.patches as patches

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='saved_models')
parser.add_argument('--smooth', type=int, default=1)
parser.add_argument('--env_name', type=str, default='FetchSlide-v1')
parser.add_argument('--epoch', type=int, default=3)
parser.add_argument('--visualize_3D', type=bool, default=False)
parser.add_argument('--alg', type=str, default='her')
args = parser.parse_args()

def load_goal_results(filepath):
    if not os.path.exists(filepath):
        print('no such file')
        return None
    with open(filepath, 'r') as f:
        lines = [line for line in f]
    keys = lines[0].strip('\n').split(',')
    result = dict.fromkeys(keys)
    origin_dgs = []
    initial_ags = []
    hindsight_goals = []
    for line in lines[1:]:
        data = line.strip('\n').split(',')
        for i in range(len(data)):
            data[i] = float(data[i])
        origin_dgs.append(data[:3])
        initial_ags.append(data[3:6])
        hindsight_goals.append(data[6:])
    result['origin_dg'] = np.array(origin_dgs)
    result['initial_ag'] = np.array(initial_ags)
    result['hindsight_goal'] = np.array(hindsight_goals)
    return result

def load_failed_desired_goal(filepath):
    if not os.path.exists(filepath):
        print('no such file')
        return None
    with open(filepath, 'r') as f:
        lines = [line for line in f]
    desired_goals = []
    for line in lines:
        data = line.strip('\n').split(',')
        for i in range(len(data)):
            data[i] = float(data[i])
        desired_goals.append(data)
    desired_goals = np.array(desired_goals)
    return desired_goals

def load_failed_goal_trajectory(filepath):
    if not os.path.exists(filepath):
        print('no such file')
        return None
    with open(filepath, 'r') as f:
        lines = [line for line in f]
    fail_goal_trajs = []
    for line in lines:
        ag_traj =[]
        data = line.strip('\n').split(',')
        for i in range(len(data)):
            data[i] = float(data[i])
        for i in range(0, len(data), 3):
            ag_traj.append(data[i:i+3])
        fail_goal_trajs.append(np.array(ag_traj))
    return np.array(fail_goal_trajs)


def Hindsight_goal_visualize():

    filepath = os.path.join(args.dir, args.env_name, args.alg, 'goal_data', 'goaldata_'+str(args.epoch)+'.csv')
    save_path = os.path.join(args.dir, args.env_name, args.alg, 'visualize')
    os.makedirs(save_path, exist_ok=True)
    result = load_goal_results(filepath)
    dgs = result['origin_dg']
    initial_ags = result['initial_ag']
    hindsight_goals = result['hindsight_goal']

    plt.figure(figsize=(8,6))
    plt.subplot(111)
    plt.scatter(dgs[:2000, 0], dgs[:2000, 1], color='b', label='desired goals')
    plt.scatter(hindsight_goals[:,0], hindsight_goals[:,1], color='g', label='hindsight goals')
    plt.scatter(initial_ags[:2000, 0], initial_ags[:2000, 1],  color='r', label='initial position')
    plt.legend()
    plt.xlim(0, 1.8)
    plt.ylim(0, 1.5)
    plt.savefig(os.path.join(save_path, '2D_visual_'+str(args.epoch) + '.jpg'))
    plt.show()
    if args.visualize_3D:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.gca(projection='3d')
        ax.scatter(dgs[:2000, 0], dgs[:2000, 1], dgs[:2000, 2], color='b',label='desired goals')
        ax.scatter(hindsight_goals[:10000,0], hindsight_goals[:10000,1], hindsight_goals[:10000,2], color='g',label='hindsight goals')
        ax.scatter(initial_ags[:2000, 0], initial_ags[:2000, 1], initial_ags[:2000, 2], color='r',label='initial position')
        # 添加坐标轴(顺序是Z, Y, X)
        plt.legend()
        ax.view_init(elev=25, azim=30)
        ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
        ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
        ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
        ax.set_zbound(lower=0, upper=0.8)
        plt.savefig(os.path.join(save_path, '3D_visual_'+str(args.epoch) + '.jpg'))
    plt.show()

def visualize_failed_trajectory():
    save_path = os.path.join(args.dir, args.env_name, args.alg, 'data_analysis')
    fail_desired_goal_path = os.path.join(args.dir, args.env_name, args.alg, 'data_analysis', 'failed_desired_goal.csv')
    failed_ag_traj_path = os.path.join(args.dir, args.env_name, args.alg, 'data_analysis', 'failed_goal_trajectory.csv')
    failed_dg = load_failed_desired_goal(fail_desired_goal_path)
    failed_ag_traj = load_failed_goal_trajectory(failed_ag_traj_path)
    fig = plt.figure(figsize=(8, 6))
    plt.subplot(111)
    plt.scatter(failed_dg[:10, 0], failed_dg[:10, 1], color='r', label='desired goals')
    for i in range(10):
        for j in range(failed_ag_traj.shape[1]):
            plt.scatter(failed_ag_traj[i, j, 0], failed_ag_traj[i, j, 1], color='g')
    plt.legend()
    plt.xlim(0.5, 1.8)
    plt.ylim(0.4, 1.5)
    plt.savefig(os.path.join(save_path, 'failed_2D.jpg'))
    if args.visualize_3D:
        fig = plt.figure(figsize=(8,6))
        ax = fig.gca(projection='3d')
        ax.scatter(failed_dg[:2, 0], failed_dg[:2, 1], failed_dg[:2, 2], color='r',label='desired goals')
        for i in range(2):
            for j in range(failed_ag_traj.shape[1]):
                ax.scatter(failed_ag_traj[i, j, 0], failed_ag_traj[i, j, 1], failed_ag_traj[i, j, 2],color='g')
        plt.legend()
        ax.view_init(elev=25, azim=30)
        ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
        ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
        ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
        ax.set_zbound(lower=0, upper=0.8)
        plt.savefig(os.path.join(save_path, 'failed_3D.jpg'))
        plt.show()
    plt.show()

# Hindsight_goal_visualize()

def density_test():
    filepath = os.path.join(args.dir, args.env_name, args.alg, 'goal_data', 'goaldata_' + str(args.epoch) + '.csv')
    save_path = os.path.join(args.dir, args.env_name, args.alg, 'visualize')
    os.makedirs(save_path, exist_ok=True)
    result = load_goal_results(filepath)
    origin_goals = result['origin_dg']
    initial_ags = result['initial_ag']
    hindsight_goals = result['hindsight_goal']
    unique_goal = np.unique(origin_goals, axis=0)
    print(unique_goal.shape)
    logger = SummaryWriter(comment='density_test')
    Density_estimater = KernalDensityEstimator(name='goal', logger=logger, sample_dim=3, kernel='gaussian')
    Density_estimater.extend(origin_goals[:20000])
    Density_estimater.fit(n_kde_samples=5000)
    generated_goals= Density_estimater.fitted_kde.sample(100)
    #dgs = np.unique(dgs, axis=0)
    indx = np.random.randint(len(hindsight_goals), size=2000)
    select_goals = hindsight_goals[indx]
    p = Density_estimater.evaluate_elementwise_entropy(select_goals)

    plt.figure(figsize=(8, 6))
    plt.subplot(111)
    plt.scatter(hindsight_goals[:20000, 0], hindsight_goals[:20000, 1], color='g', label='hindsight goals')
    plt.scatter(select_goals[:, 0], select_goals[:,1], color='r', label='select hindsight goal')
    plt.scatter(generated_goals[:, 0], generated_goals[:, 1], color='b', label='generative goal')
    #plt.scatter(initial_ags[:20000, 0], initial_ags[:20000, 1], color='r', label='initial position')
    #plt.scatter(origin_goals[:20000,0], origin_goals[:20000, 1],color='b',label='dg')
    plt.legend()
    plt.xlim(0, 1.8)
    plt.ylim(0, 1.5)
    plt.show()
    x = []
    for i in range(p.shape[0]):
        x.append([p[i], select_goals[i,:]])
        #print(p[i], hindsight_goals[i, :])
    x.sort(key=(lambda x: x[0]))
    plt.figure(2)
    for i in range(1500,2000):
        plt.scatter(x[i][1][0], x[i][1][1])
        print(x[i])
    plt.xlim(0, 1.8)
    plt.ylim(0, 1.5)
    plt.show()
    if args.visualize_3D:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.gca(projection='3d')
        #ax.scatter(dgs[:2000, 0], dgs[:2000, 1], dgs[:2000, 2], color='b',label='desired goals')
        #ax.scatter(hindsight_goals[:10000,0], hindsight_goals[:10000,1], hindsight_goals[:10000,2], color='g',label='hindsight goals')
        ax.scatter(initial_ags[:2000, 0], initial_ags[:2000, 1], initial_ags[:2000, 2], color='r',label='initial position')
        for i in range(100):
            ax.scatter(x[i][1][0], x[i][1][1], x[i][1][2], color='b')
        # 添加坐标轴(顺序是Z, Y, X)
        plt.legend()
        ax.view_init(elev=25, azim=30)
        ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
        ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
        ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
        ax.set_zbound(lower=0, upper=0.8)
        plt.show()
# density_test()


def visualize_V_critic(env_name , num_grid_point, epoch):
    args = get_args()
    save_root = './figures'
    os.makedirs(save_root, exist_ok=True)
    # env_name = 'FetchPushWallObstacle-v1'
    root = './saved_models'
    alg = 'HER_MRN'
    model_path = os.path.join(root, env_name, alg, 'model_epoch_'+str(epoch)+'.pt')
    env = make_env(env_name=env_name)
    o_mean, o_std, g_mean, g_std, critic_state_dict, V_critic_state_dict, actor_state_dict= torch.load(model_path)
    observation = env.reset()
    env_params = {'obs':observation['observation'].shape[0],
                  'goal':observation['desired_goal'].shape[0],
                  'action':env.action_space.shape[0],
                  'action_max':env.action_space.high[0]}
    object_h = observation['desired_goal'][-1]

    # print(object_h)
    # mrn_critic = MRNCritic(env_params=env_params)
    v_critic = V_critic(env_params=env_params)
    # mrn_critic.load_state_dict(critic_state_dict)
    v_critic.load_state_dict(V_critic_state_dict)
    policy = actor(env_params=env_params) 
    policy.load_state_dict(actor_state_dict)
    policy.eval()
    if env_name == 'FetchPushWallObstacle-v1':
        x = np.linspace(1.05, 1.5, num_grid_point)
        y = np.linspace(0.4, 1.1, num_grid_point)
    else:
        return None
    grid_x, grid_y = np.meshgrid(x, y)
    goal_xy = np.concatenate([np.reshape(grid_x, [-1,1]), np.reshape(grid_y, [-1, 1])],axis=-1)
    # print(goal_xy.shape)
    h = np.repeat(np.array(object_h),repeats=goal_xy.shape[0]).reshape(-1, 1)+ np.random.normal(scale=0.01, size=(goal_xy.shape[0], 1))
    
    test_goals = np.concatenate([goal_xy, h], axis=-1)


    def plot_different_states_value(states, step, v_critic, epoch):
        initial_states = np.tile(states, (num_grid_point*num_grid_point, 1))
    
        initial_states_norm = np.clip((initial_states - o_mean) / o_std, -5, 5)
        test_goals_norm = np.clip((test_goals - g_mean) / g_std, -5, 5)
        obs = torch.tensor(np.concatenate([initial_states_norm, test_goals_norm], axis=-1), dtype=torch.float32)

        v_output = v_critic(obs).detach().numpy()
        v_min, v_max = v_output.min(), v_output.max()
        print('max value',v_max, 'min value', v_min)
        v_output = np.reshape(v_output, [num_grid_point, num_grid_point])
        
        fig, ax = plt.subplots()
        c = ax.pcolormesh(grid_x, grid_y, v_output, cmap='RdBu', vmin=v_min, vmax=v_max)

        # to do: plot obstacle
        if env_name == 'FetchPushWallObstacle-v1':
            import matplotlib.patches as patches
            rect2 = patches.Rectangle((1.22, 0.8), 0.04, 0.30, linewidth=3, fill=True, facecolor='g')
            rect3 = patches.Rectangle((1.22, 0.4), 0.04, 0.29, linewidth=3, fill=True, facecolor='g')
            ax.add_patch(rect2)
            ax.add_patch(rect3)
        
        ax.set_title('visualize of V critic')
        ax.axis([grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
        fig.colorbar(c, ax=ax)
        ax.axis('tight')
        plt.legend(loc='best')


        plt.savefig(os.path.join(save_root, './v_critic_visualize_'+'epoch_'+ str(epoch)+ '_step_' + str(step)+'.jpg'))
        plt.close()
    

    # env.reset()
    test_episode = 1
    for i in range(test_episode):
        observation = env.reset()
        obs = observation['observation']
        g = observation['desired_goal']
        # plot_different_states_value(states=obs)

        for i in range(0, 100, 10):
            model_path = os.path.join(root, env_name, alg, 'model_epoch_'+str(i)+'.pt')
            o_mean, o_std, g_mean, g_std, critic_state_dict, V_critic_state_dict, actor_state_dict= torch.load(model_path)
            critic = V_critic(env_params=env_params)
            critic.load_state_dict(V_critic_state_dict)
            plot_different_states_value(obs, step=0, v_critic=critic, epoch=i)
        
        for t in range(env._max_episode_steps):
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            
            with torch.no_grad():
                pi = policy(inputs)
            action = pi.detach().numpy().squeeze()

            # put actions into the environment
            observation_new, reward, _, info = env.step(action)
            obs = observation_new['observation']

            # plot_different_states_value(observation_new['observation'], step=t, v_critic=v_critic)

        print('the episode is: {}, is success: {}'.format(i, info['is_success']))


def visualize_mrn_critic(env_name , num_grid_point, epoch):
    args = get_args()
    env = make_env(env_name)
    save_root = './figures'
    path = '/data1/ydy/RL/HER_v3/saved_models/'+ env_name +'/MRN-MinQ/models/model_epoch_'+str(epoch)+'.pt'
    models = torch.load(path)
    print(models.keys())
    observation = env.reset()
    env_params = {'obs':observation['observation'].shape[0],
                  'goal':observation['desired_goal'].shape[0],
                  'action':env.action_space.shape[0],
                  'action_max':env.action_space.high[0]}
    mrn_critic = MRNCritic(env_params, emb_dim=args.mrn_emb_dim, hidden_dim=args.mrn_hidden_dim)
    mrn_critic.load_state_dict(models['critic'])
    policy = actor(env_params=env_params)
    policy.load_state_dict(models['actor'])
    policy.eval()
    if env_name == 'FetchPushWallObstacle-v1':
        x = np.linspace(1.05, 1.5, num_grid_point)
        y = np.linspace(0.4, 1.1, num_grid_point)
    elif env_name == 'FetchPushDoubleObstacle-v1':
        x = np.linspace(1.05, 1.55, num_grid_point)
        y = np.linspace(0.45, 1.05, num_grid_point)
    object_h = observation['desired_goal'][-1]
    grid_x, grid_y = np.meshgrid(x, y)
    goal_xy = np.concatenate([np.reshape(grid_x, [-1,1]), np.reshape(grid_y, [-1, 1])],axis=-1)
    # print(goal_xy.shape)
    h = np.repeat(np.array(object_h),repeats=goal_xy.shape[0]).reshape(-1, 1)+ np.random.normal(scale=0.01, size=(goal_xy.shape[0], 1))
    
    test_goals = np.concatenate([goal_xy, h], axis=-1)
    o_mean = models['o_norm_mean']
    o_std = models['o_norm_std']
    g_mean = models['g_norm_mean']
    g_std = models['g_norm_std']

    def plot_different_states_value(states, q_critic, epoch):
        initial_states = np.tile(states, (num_grid_point*num_grid_point, 1))
    
        initial_states_norm = np.clip((initial_states - o_mean) / o_std, -5, 5)
        test_goals_norm = np.clip((test_goals - g_mean) / g_std, -5, 5)
        obs = torch.tensor(np.concatenate([initial_states_norm, test_goals_norm], axis=-1), dtype=torch.float32)
        with torch.no_grad():
            action = policy(obs)
            q_output, dist_a, dist_s = q_critic.evaluate(obs, action)
        q_min, q_max = q_output.numpy().min(), q_output.numpy().max()
        dist_a_min, dist_a_max = dist_a.numpy().min(), dist_a.numpy().max()
        dist_s_min, dist_s_max = dist_s.numpy().min(), dist_s.numpy().max()

        print(q_min, q_max, dist_a_min, dist_a_max, dist_s_min, dist_s_max)
        q_output = np.reshape(q_output.numpy(), [num_grid_point, num_grid_point])
        dist_a = np.reshape(dist_a.numpy(), [num_grid_point, num_grid_point])
        dist_s = np.reshape(dist_s.numpy(), [num_grid_point, num_grid_point])

        def plot_different_conponent(values, name, epoch, v_min, v_max):
            print(name)
            fig, ax = plt.subplots()
            c = ax.pcolormesh(grid_x, grid_y, values, cmap='RdBu', vmin=v_min, vmax=v_max)

            # to do: plot obstacle
            if env_name == 'FetchPushWallObstacle-v1':
            
                rect2 = patches.Rectangle((1.22, 0.8), 0.04, 0.30, linewidth=3, fill=True, facecolor='g')
                rect3 = patches.Rectangle((1.22, 0.4), 0.04, 0.29, linewidth=3, fill=True, facecolor='g')
                ax.add_patch(rect2)
                ax.add_patch(rect3)
                plt.xlim(1, 1.6)
                plt.ylim(0.3, 1.15)
            elif env_name == 'FetchPushDoubleObstacle-v1':
                #  rect1 = patches.Rectangle((1.05, 0.45), 0.5, 0.6, linewidth=5, edgecolor="#7f7f7f", fill=False)
                rect2 = patches.Rectangle((1.145, 0.64), 0.22, 0.04, linewidth=3, fill=True, facecolor='g')
                rect3 = patches.Rectangle((1.235, 0.86), 0.22, 0.04, linewidth=3, fill=True, facecolor='b')
                # ax.add_patch(rect1)
                ax.add_patch(rect2)
                ax.add_patch(rect3)
                plt.xlim(1, 1.6)
                plt.ylim(0.4, 1.1)
            ax.set_title('visualize of'+name)
            ax.axis([grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
            fig.colorbar(c, ax=ax)
            ax.axis('tight')
            plt.legend(loc='best')
            plt.savefig(os.path.join(save_root, env_name+'_mrn_critic_visualize_'+name+'_epoch_'+ str(epoch)+'.jpg'))
        
        plot_different_conponent(q_output, name='Q values', epoch=epoch, v_min=q_min,v_max=q_max)
        plot_different_conponent(dist_a, name='dist a', epoch=epoch, v_min=q_min,v_max=q_max)
        plot_different_conponent(dist_s, name='dist s', epoch=epoch, v_min=q_min,v_max=q_max)
       

    observation = env.reset()
    obs = observation['observation']
    plot_different_states_value(obs, mrn_critic, epoch)
    


def visualize_aim_discriminator(env_name, num_grid_point, epoch):
    from rl_modules.teachers.AIM.discriminator import DiscriminatorEnsemble, Discriminator
    args = get_args()
    env = make_env(env_name)
    save_root = './figures'
    aim_path = '/data1/ydy/RL/HER_v3/saved_models/'+ env_name +'/AIM/models/aim_discriminator_epoch_'+str(epoch)+'.pt'

    model_path = '/data1/ydy/RL/HER_v3/saved_models/'+ env_name +'/AIM/seed-5/models/model_epoch_'+str(epoch)+'.pt'
    models = torch.load(model_path)
    observation = env.reset()
    env_params = {'obs':observation['observation'].shape[0],
                  'goal':observation['desired_goal'].shape[0],
                  'action':env.action_space.shape[0],
                  'action_max':env.action_space.high[0]}
    aim_discriminator = Discriminator(x_dim=env_params['goal'] * 2, reward_type='aim', lambda_coef = args.lambda_coef)
    weights = torch.load(aim_path)
    aim_discriminator.load_state_dict(weights)
    if env_name == 'FetchPushWallObstacle-v1':
        x = np.linspace(1.05, 1.5, num_grid_point)
        y = np.linspace(0.4, 1.1, num_grid_point)
    elif env_name == 'FetchPushDoubleObstacle-v1':
        x = np.linspace(1.05, 1.55, num_grid_point)
        y = np.linspace(0.45, 1.05, num_grid_point)
    o_mean = models['o_norm_mean']
    o_std = models['o_norm_std']
    g_mean = models['g_norm_mean']
    g_std = models['g_norm_std']
    object_h = observation['desired_goal'][-1]
    grid_x, grid_y = np.meshgrid(x, y)
    goal_xy = np.concatenate([np.reshape(grid_x, [-1,1]), np.reshape(grid_y, [-1, 1])],axis=-1)
    # print(goal_xy.shape)
    h = np.repeat(np.array(object_h),repeats=goal_xy.shape[0]).reshape(-1, 1)+ np.random.normal(scale=0.01, size=(goal_xy.shape[0], 1))
    
    test_goals = np.concatenate([goal_xy, h], axis=-1)
    test_goals_norm = np.clip((test_goals - g_mean) / g_std, -5, 5)
    desired_goal = np.array([1.1,0.75, 0.44])

    test_desired_goal = np.repeat(desired_goal.reshape(1,-1), repeats=test_goals.shape[0], axis=0)
    test_desired_goal_norm = np.clip((test_desired_goal - g_mean) / g_std, -5, 5)
    obs = torch.tensor(np.concatenate([test_goals_norm, test_desired_goal_norm], axis=-1), dtype=torch.float32)
    
    r = aim_discriminator.reward(obs)

    r_min, r_max = r.min(), r.max()
    r_reshape = np.reshape(r, [num_grid_point, num_grid_point])
    fig, ax = plt.subplots()
    c = ax.pcolormesh(grid_x, grid_y, r_reshape, cmap='RdBu', vmin=r_min, vmax=r_max)

            # to do: plot obstacle
    if env_name == 'FetchPushWallObstacle-v1':
    
        rect2 = patches.Rectangle((1.22, 0.8), 0.04, 0.30, linewidth=3, fill=True, facecolor='g')
        rect3 = patches.Rectangle((1.22, 0.4), 0.04, 0.29, linewidth=3, fill=True, facecolor='g')
        ax.add_patch(rect2)
        ax.add_patch(rect3)
        plt.xlim(1, 1.6)
        plt.ylim(0.3, 1.15)
    elif env_name == 'FetchPushDoubleObstacle-v1':
        #  rect1 = patches.Rectangle((1.05, 0.45), 0.5, 0.6, linewidth=5, edgecolor="#7f7f7f", fill=False)
        rect2 = patches.Rectangle((1.145, 0.64), 0.22, 0.04, linewidth=3, fill=True, facecolor='g')
        rect3 = patches.Rectangle((1.235, 0.86), 0.22, 0.04, linewidth=3, fill=True, facecolor='b')
        # ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)
        plt.xlim(1, 1.6)
        plt.ylim(0.4, 1.1)

    ax.set_title('visualize of aim discriminator')
    ax.axis([grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
    fig.colorbar(c, ax=ax)
    ax.axis('tight')
    plt.legend(loc='best')
    plt.savefig(os.path.join(save_root, env_name+'_aim-discriminator_'+'_epoch_'+ str(epoch)+'.jpg'))



def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    
    return inputs


def mine_test(env_name, epoch, seed):
    model_path =  '/data1/ydy/RL/HER_v3/saved_models/'+ env_name +'/MINE/'+'seed-'+ str(seed)+'/models/model_epoch_'+str(epoch)+'.pt'
    mine_net_path = '/data1/ydy/RL/HER_v3/saved_models/'+ env_name +'/MINE/'+'seed-'+ str(seed)+'/models/mine_net_epoch_'+str(epoch)+'.pt'

    model_weight_dict = torch.load(model_path)
    print(model_weight_dict.keys())
    from rl_modules.teachers.MINE.mine import MineNet
   
    args = get_args()
    env = make_env(env_name)
    observation = env.reset()
    env_params = {'obs':observation['observation'].shape[0],
                  'goal':observation['desired_goal'].shape[0],
                  'action':env.action_space.shape[0],
                  'action_max':env.action_space.high[0]}
    policy = actor(env_params)
    policy.load_state_dict(model_weight_dict['actor'])
    policy.eval()

    mine_net = MineNet(input_size=env_params['goal']*2)
    mine_net.load_state_dict(torch.load(mine_net_path))
    
    o_mean = model_weight_dict['o_norm_mean']
    o_std = model_weight_dict['o_norm_std']
    g_mean = model_weight_dict['g_norm_mean']
    g_std = model_weight_dict['g_norm_std']


    mine_result = []
    obs_traj = []
    test_episode = 1
    for t in range(test_episode):
        print('the {} th trajectory'.format(t))
        observation = env.reset()
        obs_traj = []
        mine_result = []
        obs = observation['observation']
        g = observation['desired_goal']
        obs_traj.append(obs)
        for j in range(50):
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            with torch.no_grad():
                pi = policy(inputs)
            action = pi.detach().numpy().squeeze()

            # put actions into the environment
            observation_new, reward, _, info = env.step(action)
            
            obs = observation_new['observation']
            obs_traj.append(obs)
        # calculate each step mutual info

        for i in range(len(obs_traj)-1):
            gripper_state, object_state = split_robot_state_from_observation(env_name=env_name, observation=obs_traj[i])
            gripper_state_next, object_state_next =  split_robot_state_from_observation(env_name=env_name, observation=obs_traj[i+1])
            joint = torch.from_numpy(np.hstack([gripper_state.reshape(1,-1), object_state.reshape(1,-1)]).astype(np.float32))
            marginal =torch.from_numpy(np.hstack([gripper_state_next.reshape(1,-1), object_state.reshape(1,-1)]).astype(np.float32))
            with torch.no_grad():
                T_joint = mine_net(joint)
                T_marginal = mine_net(marginal)
                T_marginal_exp = torch.exp(T_marginal)
                mine_estimate = torch.mean(T_joint) - torch.log(torch.mean(T_marginal_exp))
                
            distance = np.linalg.norm(gripper_state - object_state)
            mine_result.append([float(mine_estimate.cpu().numpy()), distance])
        plt.figure()
        plt.plot(np.array(mine_result)[:,0])
        plt.plot(np.array(mine_result)[:,1])
        plt.savefig('./mine.jpg')
        print(mine_result)

        

def split_robot_state_from_observation(env_name, observation, type='gripper_pos'):
    obs = np.asarray(observation)
    dimo = obs.shape[-1]
    if env_name.lower().startswith('fetch'):
        assert dimo == 25, "Observation dimension changed."
        grip_pos, object_pos, object_rel_pos, gripper_state, object_rot, object_velp, object_velr, grip_velp, gripper_vel =\
            np.hsplit(obs, np.array([3, 6, 9, 11, 14, 17, 20, 23]))
        if type =='gripper_pos_vel':
            robot_state =np.concatenate((grip_pos.copy(), grip_velp.copy()), axis=-1)
        elif type == 'gripper_pos':
            robot_state = grip_pos.copy()
        obs_achieved_goal = object_pos.copy()
        return robot_state, obs_achieved_goal
    elif env_name.lower().startswith('hand'):
        assert NotImplementedError
        return None


if __name__=='__main__':
    
    # visualize_V_critic(env_name = 'FetchPushWallObstacle-v1', num_grid_point=60, epoch=70)
    # visualize_mrn_critic(env_name ='FetchPushDoubleObstacle-v1', num_grid_point=60, epoch=0)
    # visualize_aim_discriminator(env_name='FetchPushDoubleObstacle-v1', num_grid_point=60, epoch=20)
    mine_test(env_name='FetchPush-v1',epoch=20,seed=5)
