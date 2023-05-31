import math
import torch
import numpy as np
from rl_modules.teachers.abstract_teacher import AbstractTeacher
import scipy.spatial.distance as scdis
import os
from mpi4py import MPI
import csv
from rl_modules.teachers.ICM.icm import ICMTeacher
from rl_modules.teachers.MINE.mine import MineTeacher


'''
this goal teacher achieves the following goal sample method:

MEGA: sample goal with lowest density,
      paper: Maximum Entropy Gain Exploration for Long Horizon Multi-goal Reinforcement Learning

Diverse: sample goal according to 1 / p, where p is the density, This is similar to using Skew-Fit alpha=-1.  
        paper: 'Skew-Fit: State-Covering Self-Supervised reinforcement learning'

RIG: sample achieved goal randomly, 
     paper:'Visual reinforcement learning with imagined goals'

MinQ: sample goals that have lowest Q values,
     paper:'Dynamical distance learning for semi-supervised and unsupervised skill discovery'

VLP: value-based learning progress

 

'''

class AGETeacher(AbstractTeacher):
    def __init__(self, args, env, env_params, density_estimator, policy,o_norm, g_norm, buffer, dg_buffer, ag_buffer, state_discover_module):
        self.density_estimator = density_estimator
        self.args = args
        self.env = env
        self.env_params = env_params
        self.policy = policy
        self.v_critic = policy.V_critic
        self.v_target = policy.V_critic_target

        self.o_norm = o_norm
        self.g_norm = g_norm
        self.experience_buffer = buffer
        self.dg_buffer = dg_buffer
        self.ag_buffer = ag_buffer
        self.dg_sample_num = 100
        self.max_v = 1.0 / (1-self.args.gamma)
        self.delta = self.args.shift_delta
        self.q_cutoff = self.args.q_cuttoff
        self.cutoff_thresh = -20.0
        self.sample_num = 250
        self.goal_shift = self.args.goal_shift
        self.sample_stratagy = self.args.sample_stratage
        self.save_root = os.path.join(self.args.save_dir, self.args.env_name, self.args.alg, 'seed-'+str(self.args.seed))
        self.goal_save_path = os.path.join(self.save_root,'candidate_goal')
        self.state_save_path = os.path.join(self.save_root, 'novel_state')
        self.save_fre =self.args.save_interval
        if MPI.COMM_WORLD.Get_rank() == 0:
            os.makedirs(self.goal_save_path, exist_ok=True)
            os.makedirs(self.state_save_path, exist_ok=True)
        self.epoch = 0

        self.state_method = self.args.state_discover_method
        if self.args.state_discover:
            if self.state_method in ['icm','mine']:
                self.state_model = state_discover_module
        else:
            self.state_model = None
        self.balance_lambda = self.args.age_lambda

    def update(self):
        self.density_estimator.fit()
        if self.args.state_discover:
            if self.state_model is not None: 
                self.state_model.update()

     

    def sample(self, batchsize):

        ags = self.ag_buffer.random_sample(batch_size=self.sample_num)
        assert ags.shape[0] > batchsize

        if self.goal_shift:
            ags = self.frontier_goal_shifting(ags)
        ags = np.unique(ags, axis=0) # to ensure the diversity of selected goals
        assert ags.shape[0] > batchsize
        

        q_values = self.compute_q_values(ags).reshape(-1, 1)
        write_content = np.concatenate((ags, q_values), axis=1)
            
        novel_state_based_pri = self.cal_novel_state_based_value(ags).reshape(-1,1)
        density_pri = self.cal_density_based_priority(ags, q_values).reshape(-1, 1)
        value_pri = self.cal_value_distance(ags, q_values).reshape(-1, 1)
        lp_pri = self.cal_learning_progress(ags, q_values).reshape(-1, 1)
        mega_vad_pri = (density_pri + value_pri) / 2
        mega_lp_pri = (density_pri + lp_pri) / 2
        mega_minv = self.balance_lambda * density_pri + (1-self.balance_lambda)*novel_state_based_pri
        # mega_minv = mega_minv / mega_minv.sum()
        if MPI.COMM_WORLD.Get_rank() == 0:
            write_content = np.concatenate((ags, q_values, density_pri, value_pri, lp_pri, mega_vad_pri, mega_lp_pri), axis=1)
            with open(os.path.join(self.goal_save_path, 'epoch_'+str(self.epoch)+'.csv'), mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(write_content.tolist())

        self.epoch += 1

        if self.sample_stratagy == 'Diverse':
            selected_idx = np.random.choice(ags.shape[0], size=batchsize, replace=True, p=density_pri.flatten())
        elif self.sample_stratagy == 'MEGA':
            selected_idx = np.argsort(density_pri.flatten())[-batchsize:]
        elif self.sample_stratagy == 'MinQ':
            selected_idx = np.argsort(q_values.flatten())[:batchsize]
        elif self.sample_stratagy == 'MinV':
            selected_idx = np.argsort(novel_state_based_pri.flatten())[-batchsize:]
        elif self.sample_stratagy == 'RIG':
            # print(ags.shape, batchsize)
            selected_idx = np.random.randint(ags.shape[0], size=batchsize)
        elif self.sample_stratagy == 'VAD':
            selected_idx = np.random.choice(ags.shape[0], size=batchsize, replace=True, p=value_pri.flatten())
        elif self.sample_stratagy == 'LP':
            selected_idx = np.random.choice(ags.shape[0], size=batchsize, replace=True, p=lp_pri.flatten())
        elif self.sample_stratagy == 'MEGA_VAD':
            selected_idx = np.random.choice(ags.shape[0], size=batchsize, replace=True, p=mega_vad_pri.flatten())
        elif self.sample_stratagy == 'MEGA_LP':
            selected_idx = np.random.choice(ags.shape[0], size=batchsize, replace=True, p=mega_lp_pri.flatten())
        elif self.sample_stratagy == 'MEGA_MinV':
            # selected_idx = np.random.choice(ags.shape[0], size=batchsize, replace=True, p=mega_minv.flatten())
            selected_idx = np.argsort(mega_minv.flatten())[-batchsize:]

        
        selected_goals = ags[selected_idx].copy()
        q_values = q_values[selected_idx].copy()
        value_pri = value_pri[selected_idx].copy()
        lp_pri = lp_pri[selected_idx].copy()
        density_pri = density_pri[selected_idx].copy()
        mega_vad_pri = mega_vad_pri[selected_idx].copy()
        mega_lp_pri = mega_lp_pri[selected_idx].copy()


        values = np.concatenate((q_values, density_pri, value_pri, lp_pri,mega_vad_pri, mega_lp_pri), axis=1)
        if self.q_cutoff:
            return selected_goals, values
        else:
            return selected_goals


    def frontier_goal_shifting(self, achieved_goals):
        # desired_goals = self.dg_buffer.random_sample(batch_size=achieved_goals.shape[0])
        # direction = desired_goals - achieved_goals
        # norm_direction = direction / np.linalg.norm(direction)
        # shifting_goals = achieved_goals + self.delta * norm_direction
        delta = self.env.distance_threshold
        shift_goals = achieved_goals.copy()
        shift_goals[:,:2] += np.random.normal(0, delta, size=(shift_goals.shape[0], 2))
        return np.concatenate([achieved_goals, shift_goals], axis=0)



    def cal_density_based_priority(self, goals, q_values):
        if self.density_estimator.fitted_model is None:
            self.density_estimator.fitted_model.fit()
        log_px = self.density_estimator.fitted_model.score_samples(goals) # logpx lower is better
        if self.q_cutoff:
            q_mean = np.mean(q_values)
            bad_q_inds = q_values.flatten() <= q_mean
            log_px[bad_q_inds] *= -1e-8
        abs_logpx = np.abs(log_px)
        select_p = abs_logpx / abs_logpx.sum()
        
        return select_p


    def cal_value_distance(self, goals, q_values):
        obs_dict = self.env.reset()
        obs = obs_dict['observation']
        input_obs = np.repeat(obs.reshape(1, -1), repeats=goals.shape[0], axis=0) 
        obs_norm, g_norm = self._preproc_og(input_obs, goals)
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        input_norm_obs = torch.tensor(inputs_norm, dtype=torch.float32)
        with torch.no_grad():
            v_current = self.v_critic(input_norm_obs).numpy().reshape(-1, 1)

        eval_desired_goal = self.dg_buffer.random_sample(self.dg_sample_num)
        init_obs = np.repeat(obs.reshape(1, -1), repeats=self.dg_sample_num, axis=0)

        init_norm_obs, repeat_dg_norm = self._preproc_og(init_obs, eval_desired_goal) 
        target_input_norm = np.concatenate([init_norm_obs, repeat_dg_norm], axis=1)
        target_input_norm = torch.tensor(target_input_norm, dtype=torch.float32)
        with torch.no_grad():
            v_target = self.v_critic(target_input_norm).numpy().reshape(-1, 1)
        
        v_diff = np.clip(np.sqrt(scdis.cdist(v_current, v_target).mean(axis=1)), 1e-5, self.max_v).flatten()

        if self.q_cutoff:
            q_mean = np.mean(q_values)
            bad_q_inds = q_values.flatten() <= q_mean
            v_diff[bad_q_inds] = self.max_v
        # the larger the value difference, the potiental is less
        v_diff = 1 / v_diff
        probs = v_diff / v_diff.sum()
      
        return probs
    

    def cal_learning_progress(self, goals, q_values):
        obs_dict = self.env.reset()
        obs = obs_dict['observation']
        input_obs = np.repeat(obs.reshape(1, -1), repeats=goals.shape[0], axis=0) 
        obs_norm, g_norm = self._preproc_og(input_obs, goals)
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        input_norm_obs = torch.tensor(inputs_norm, dtype=torch.float32)
        with torch.no_grad():
            lp_current = self.v_critic(input_norm_obs).numpy().flatten()
            lp_target = self.v_target(input_norm_obs).numpy().flatten()

        lp_diff = lp_current - lp_target
        lp_diff[lp_diff < 0] = 0
        if self.q_cutoff:
            q_mean = np.mean(q_values)
            bad_q_inds = q_values.flatten() <= q_mean
            lp_diff[bad_q_inds] = 0

        if np.sum(lp_diff) < 1e-3:
            probs = np.ones(goals.shape[0]) / goals.shape[0]
        else:
            probs = lp_diff / lp_diff.sum()
        return probs


    
    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        norm_o = self.o_norm.normalize(o)
        norm_g = self.g_norm.normalize(g)
        return norm_o, norm_g

    def compute_q_values(self, goals):
        # sample initial state
        init_state_num = 32
        idxs = np.random.randint(self.experience_buffer.current_size, size=init_state_num)
        temp_states = self.experience_buffer.buffers['obs'][idxs].copy()
        states = temp_states[:, 0, :]
        states = np.tile(states,(goals.shape[0], 1))
        goals = np.repeat(goals, repeats=init_state_num, axis=0)
        norm_states, norm_g = self._preproc_og(states, goals)
        sg_input = torch.tensor(np.concatenate((norm_states, norm_g), axis=-1), dtype=torch.float32)
        with torch.no_grad():
            if hasattr(self.policy, 'actor_target_network'):
                pi = self.policy.actor_target_network(sg_input)
                actions = torch.from_numpy(self.policy._select_actions(pi))
            else:
                pi = self.policy.actor_network(sg_input)
                actions = torch.from_numpy(self.policy._select_actions(pi))
            q_values = self.policy.critic_target_network(sg_input, actions)
            q_values = q_values.detach().numpy().reshape(init_state_num, -1) # shape : [init_num, num_goals]
            q_values = np.mean(q_values, axis=0)
        return q_values         
        
    def split_robot_state_from_observation(self, env_name, observation, type='gripper_pos'):
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
       
    def select_novel_states(self, select_num=32):
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

            if self.state_method=='prior':
                pri = np.ones((shape[0],1))
            elif self.state_method=='icm' or self.state_method=='mine':
                pri = []
                for i in range(idxs.shape[0]):
                    pri.append(pred_diff[i, idxs[i]])
                pri = np.array(pri).reshape(-1,1)
            
        else:
            select_states = temp_states[:, 0, :]  # s0
            pri = np.ones((shape[0],1))
        select_states = np.array(select_states)
        # gripper_object_pos = select_states[:,:6]
        # write_content =  np.concatenate((gripper_object_pos, pri),axis=1).tolist()
        # with open(os.path.join(self.state_save_path, self.state_method+'-novel_state-'+'epoch-'+str(self.epoch)+'.csv'),mode='a', newline='') as f:
        #     writer = csv.writer(f) 
        #     writer.writerows(write_content)

        return select_states

    

    def cal_novel_state_based_value(self, goals):

        init_state_num = 32
        novel_states = self.select_novel_states(select_num=init_state_num)
        states = np.tile(novel_states,(goals.shape[0], 1))
        goals = np.repeat(goals, repeats=init_state_num, axis=0)
        norm_states, norm_g = self._preproc_og(states, goals)
        sg_input = torch.tensor(np.concatenate((norm_states, norm_g), axis=-1), dtype=torch.float32)
        with torch.no_grad():
            state_values = self.v_critic(sg_input)
        reshape_state_values = state_values.detach().numpy().reshape(init_state_num, -1)
        state_values = np.mean(reshape_state_values, axis=0)
        abs_values = np.abs(state_values)
        pri = abs_values/ abs_values.sum()
        return pri





    def load(self, path):
        return super().load(path)

    def save(self, path):
        return super().save(path)