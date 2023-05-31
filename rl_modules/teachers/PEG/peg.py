# planning goals for exploration

import numpy as np
import torch
import torch.nn as nn
import os
from mpi4py import MPI
import csv
from rl_modules.rollouts import ModelBasedRollouts
from rl_modules.dynamics import ForwardDynamics
import wandb
from rl_modules.teachers.ICM.icm import ICMTeacher
from rl_modules.teachers.MINE.mine import MineTeacher


def obs_to_goal_func(obs, env_name):
    if env_name.lower().startswith('fetch'):
        if len(np.array(obs).shape)==1:
            return obs[3:6]
        elif len(np.array(obs).shape==2):
            return obs[:,3:6]
    else:
        print('env obs to goal function is not implement')
        return 0

class MBPlanTeacher(object):
    def __init__(self, args, env, env_params, policy, o_norm, g_norm, buffer, mb_buffer, state_discover_module) -> None:
        self.args = args
        self.policy = policy
        self.env = env
        self.env_params = env_params
        self.o_norm = o_norm
        self.g_norm = g_norm
        self.experience_buffer = buffer
        self.mb_buffer = mb_buffer
        self.obs_to_goal_func = obs_to_goal_func
        self.forward_model = ForwardDynamics(args, env, env_params, o_norm, buffer, hidden_dim=128)
        self.mbplanner = ModelBasedRollouts(self.args.env_name,  env, o_norm, g_norm, policy, self.forward_model, mb_buffer, obs_to_goal_func, env.compute_reward)
        self.save_root = os.path.join(self.args.save_dir, self.args.env_name, self.args.alg)
        self.model_path = os.path.join(self.save_root, 'models')
        self.save_frequency = self.args.save_interval
        self.epoch = 0
        if MPI.COMM_WORLD.Get_rank() == 0:
            os.makedirs(self.model_path, exist_ok=True)
        
        self.state_method = self.args.state_discover_method
        if self.state_method in ['icm','mine']:
            self.state_model = state_discover_module
        else:
            self.state_model = None

    def update(self):
        total_loss = []
        for _ in range(self.args.mb_update_steps):
            obs, actions, obs_next = self.sample_batch()

            loss = self.mbplanner.update_forward_model(obs=obs, actions=actions, next_obs=obs_next, update_times=self.args.update_time_per_batch, )

            total_loss.append(loss)

        if MPI.COMM_WORLD.Get_rank() == 0:
            result = {'forward model loss': np.mean(np.array(total_loss))}
            wandb.log(result)
            if self.epoch % self.save_frequency ==0:
                self.save_model()
        
        self.epoch += 1

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    def sample_batch(self):
        if self.args.use_per:
            transitions, idxes = self.experience_buffer.sample(self.args.mb_batch_size)
        else:
            transitions = self.experience_buffer.sample(self.args.mb_batch_size)
        sampled_batch_size = transitions['obs'].shape[0]
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        obs, actions, obs_next = transitions['obs'], transitions['actions'], transitions['obs_next']
        return obs, actions, obs_next
        
    def generate_model_based_transition(self, batchsize):
        


        pass


    def select_novel_states(self, select_num):
        init_state_num = select_num
        idxs = np.random.randint(self.experience_buffer.current_size, size=init_state_num)
        temp_states = self.experience_buffer.buffers['obs'][idxs][:,:-1,:].copy() # shape[num, T, obs]
        temp_next_states = self.experience_buffer.buffers['obs'][idxs][:,1:,:].copy()
        shape = temp_states.shape
        temp_states_reshape = temp_states.reshape(-1, shape[-1])
        temp_next_states_reshape = temp_next_states.reshape(-1, shape[-1])
        # print(temp_next_states_reshape.shape)
        select_states = []
        if self.args.state_discover:
            if self.state_method=='prior':
                gripper_pos, ags = self.split_robot_state_from_observation(env_name=self.args.env_name, observation=temp_states_reshape)
                dist = np.linalg.norm(gripper_pos-ags, axis=-1).reshape(shape[0], shape[1])
                idxs = np.argmin(dist, axis=-1)
               
            elif self.state_method=='mine':
                norm_obs = self.o_norm.normalize(temp_states_reshape)
                norm_next_obs = self.o_norm.normalize(temp_next_states_reshape)
                pred_diff = self.state_model.compute_reward(norm_obs, norm_next_obs, clip=False).reshape(shape[0], shape[1])
                # reshape(shape[0], shape[1])
                # print(pred_diff.shape)
                idxs = np.argmax(pred_diff, axis=-1)

            elif self.state_method =='icm':
                norm_obs = self.o_norm.normalize(temp_states_reshape)
                norm_next_obs = self.o_norm.normalize(temp_next_states_reshape)
                actions= self.experience_buffer.buffers['actions'][idxs].copy()
                actions = actions.reshape(shape[0]*shape[1], -1)
                pred_diff = self.state_model.compute_reward(norm_obs, actions, norm_next_obs, clip=False).reshape(shape[0], shape[1])
                idxs = np.argmax(pred_diff, axis=-1)

            for i in range(idxs.shape[0]):
                s = temp_states[i, idxs[i], :]
                select_states.append(s)

            
        else:
            select_states = temp_states[:, 0, :]  # s0
        select_states = np.array(select_states)
        return select_states

    def save_model(self):
        torch.save(self.forward_model, os.path.join(self.model_path, 'ForwardModel_'+'epoch_'+str(self.epoch)+'.pt'))
