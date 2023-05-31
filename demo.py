import torch
from rl_modules.models import actor
import gym
import csv
import numpy as np
import os
from utils.envbuilder import make_env
import argparse
from PIL import Image

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    
    return inputs

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--smooth', type=int, default=1)
    parser.add_argument('--env_name', type=str, default='FetchPushDoubleObstacle-v1')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--render', type=bool, default=True)
    parser.add_argument('--demo_length', type=int, default=1)
    parser.add_argument('--clip_obs', type=float, default=200, help='the clip ratio')
    parser.add_argument('--clip_range', type=float, default=5, help='the clip range')
    parser.add_argument('--alg', type=str, default='BVN')
    parser.add_argument('--save_data', type=bool, default=False, help='whether to save data')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    # load the model param
    model_path = os.path.join(args.save_dir, args.env_name, args.alg, 'models', 'model_best.pt')
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    # save_path = os.path.join(args.save_dir, args.env_name, args.alg, 'data_analysis')
    # os.makedirs(save_path, exist_ok=True)
    # create the environment
    env, _ = make_env(args.env_name)
    # get the env param
    observation = env.reset()
    
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0], 
                  'goal': observation['desired_goal'].shape[0], 
                  'action': env.action_space.shape[0], 
                  'action_max': env.action_space.high[0],
                  }
    
    # create the actor network
    actor_network = actor(env_params)
    actor_network.load_state_dict(model['actor'])
    actor_network.eval()
    o_mean = model['o_norm_mean']
    o_std =  model['o_norm_std']
    g_mean = model['g_norm_mean']
    g_std = model['g_norm_std']
    desired_goals = []
    ag_trajs = []
    failed_counter = 0
    for i in range(args.demo_length):
        observation = env.reset()
        ag_traj = []
        # start to do the demo
        obs = observation['observation']
        g = observation['desired_goal']

        for t in range(env._max_episode_steps):
            if args.render:
                if t in [0,5,10,20,30,40]:
                    img = env.render(mode='rgb_array')
                    image_pil=Image.fromarray(img)
                    image_pil.save(os.path.join('./figures',args.env_name+'-'+str(t)+'.png'))

            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            with torch.no_grad():
                pi = actor_network(inputs)
            action = pi.detach().numpy().squeeze()

            # put actions into the environment
            observation_new, reward, _, info = env.step(action)
            ag_traj += list(observation_new['achieved_goal'])
            #print(info['is_success'])
            obs = observation_new['observation']
        if info['is_success'] == 0:
            failed_counter += 1
            print('{:d}th failed trajectory'.format(failed_counter))
            desired_goals.append(g)
            ag_trajs.append(ag_traj)
        print('the episode is: {}, is success: {}'.format(i, info['is_success']))
    print('success_rate: {}%'.format((1-failed_counter/args.demo_length) * 100))
   