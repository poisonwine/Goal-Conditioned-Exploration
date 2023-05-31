import numpy as np
import torch
from rl_modules.teachers.AIM.discriminator import DiscriminatorEnsemble, Discriminator
import os
from mpi4py import MPI
import torch.nn.functional as F
import wandb

class AIMTeacher(object):
    def __init__(self, args, env_param, g_norm, buffer, ag_buffer, noise=0.05, g_normalize=True, reward_type='aim', use_ensemble=False, n_ensemble=5):
        self.args = args
        self.env_params = env_param
        self.buffer = buffer
        self.use_ensemble = use_ensemble
        self.reward_type = reward_type
        self.n_ensemble = n_ensemble
        self.batch_size = self.args.aim_batch_size
        self.ag_buffer = ag_buffer
        if use_ensemble:
            self.aim_discriminator = DiscriminatorEnsemble(n_ensemble=self.n_ensemble, x_dim=self.env_params['goal'] * 2, reward_type=reward_type, lambda_coef = self.args.lambda_coef)
        else:
            self.aim_discriminator = Discriminator(x_dim=self.env_params['goal'] * 2, reward_type=reward_type, lambda_coef = self.args.lambda_coef)
        self.g_normalizer = g_norm
        self.normalize = g_normalize
        self.noise_mean = noise
        self.save_root = os.path.join(self.args.save_dir, self.args.env_name, self.args.alg)
       
  
        self.wgan_loss = 0.
        self.graph_penalty = 0.
        self.min_aim_f_loss = 0.
        self.aim_rew_std = 1.
        self.aim_rew_mean = 0.
        self.save_root = os.path.join(self.args.save_dir, self.args.env_name, self.args.alg)
        # elf.save_root = os.path.join(self.args.save_dir, self.args.env_name, self.args.alg)
        self.goal_save_path = os.path.join(self.save_root, 'seed-'+str(self.args.seed),'candidate_goal')
      
        self.model_path = os.path.join(self.save_root, 'models')
        if MPI.COMM_WORLD.Get_rank() == 0:
            os.makedirs(self.model_path, exist_ok=True)
        self.epoch = 0
        self.save_frequency = self.args.save_interval
        self.n_candidate = self.args.n_candidate
        self.temperature = self.args.aim_temperature
        self.sample_type = self.args.sample_type

    def update(self):
        drewards = []
        for _ in range(self.args.aim_discriminator_steps):
            _, rsamples = self.update_aim_discriminator()                
            drewards.extend(rsamples)
        drewards = np.reshape(drewards, newshape=(-1,))
        self.aim_rew_std = np.std(drewards) + self.args.aim_reward_norm_offset # 0.1
        self.aim_rew_mean = np.max(drewards) + self.args.aim_reward_norm_offset # 0.1
        if MPI.COMM_WORLD.Get_rank() == 0:
            result = {'aim_reward_mean': self.aim_rew_mean,
                      'aim_reward_std': self.aim_rew_std}
            wandb.log(result)
            if self.epoch % self.save_frequency == 0:
                self.save_model()
        self.epoch += 1

    def update_aim_discriminator(self):
        if self.args.use_per:
            transitions, idxes = self.buffer.sample(self.args.batch_size)
        else:
            transitions = self.buffer.sample(self.args.batch_size)
        sampled_batch_size = transitions['obs'].shape[0]
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        obs_achiev_goal, obs_desired_goal = transitions['ag'], transitions['g']
        next_obs_achiev_goal, next_obs_desired_goal = transitions['ag_next'], transitions['g_next']
        if self.normalize:
            obs_achiev_goal, obs_desired_goal = self.g_normalizer.normalize(obs_achiev_goal), self.g_normalizer.normalize(obs_desired_goal)
            next_obs_achiev_goal, next_obs_desired_goal = self.g_normalizer.normalize(next_obs_achiev_goal), self.g_normalizer.normalize(next_obs_desired_goal)

        policy_states = torch.tensor(np.concatenate([obs_achiev_goal, obs_desired_goal], axis=-1), dtype=torch.float32) # s, s_g
        policy_next_states = torch.tensor(np.concatenate([next_obs_achiev_goal, next_obs_desired_goal], axis=-1), dtype=torch.float32) # s', s_g  
        target_states_desired = next_obs_desired_goal + np.random.normal(scale=self.noise_mean, size=next_obs_desired_goal.shape)     
        target_states = torch.tensor(np.concatenate([target_states_desired, next_obs_desired_goal], axis=-1),dtype=torch.float32)  # s_g, s_g

        self.aim_disc_loss, self.wgan_loss, self.graph_penalty, self.min_aim_f_loss = \
            self.aim_discriminator.optimize_discriminator(target_states, policy_states, policy_next_states)

        all_rewards = []
        all_rewards.append(self.aim_discriminator.reward(target_states))
        all_rewards.append(self.aim_discriminator.reward(policy_states))
        return self.aim_disc_loss, all_rewards

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    def compute_reward(self, achieved_goal, desired_goal):
        assert achieved_goal.shape == desired_goal.shape
        if self.normalize:
            achieved_goal, desired_goal = self.g_normalizer.normalize(achieved_goal), self.g_normalizer.normalize(desired_goal)
        obs = torch.tensor(np.concatenate([achieved_goal, desired_goal], axis=-1), dtype=torch.float32)
        aim_reward = self.aim_discriminator.reward(obs)
        aim_reward = (aim_reward - self.aim_rew_mean)/(self.aim_rew_std*2.)
        return aim_reward 

    
    
    def sample(self, batchsize):
        ags = self.ag_buffer.random_sample(self.n_candidate)
        ags += np.random.normal(scale=0.03, size=ags.shape)
        dg_index = np.random.randint(self.buffer.current_size, size=20)
        dgs = self.buffer.buffers['g'][dg_index][0,:]
        unique_dgs = np.unique(dgs, axis=0)
        output = []
        for i in range(unique_dgs.shape[0]):
            repeat_dgs = np.repeat(unique_dgs[i,:].reshape(1,-1), repeats=self.n_candidate, axis=0)
            aim_out = self.compute_reward(ags, repeat_dgs).reshape(1,-1)
            output.append(aim_out)
        
        aim_outputs = np.array(output).mean(axis=0).reshape(1,-1)
        aim_outs = torch.from_numpy(aim_outputs)
        aim_out_min = aim_outs.min()
        aim_out_max = aim_outs.max()
        if self.sample_type == 'softmin':
            logits = ((aim_outs - aim_out_min) / (aim_out_max - aim_out_min+1e-5) - 0.5) * 2.0

            prob = F.softmin(logits/self.temperature, dim = 0) #[bs]

            dist = torch.distributions.Categorical(probs=prob)
            sample_idx = dist.sample((batchsize, ))
            sample_idx = sample_idx.cpu().numpy().flatten()
        elif self.sample_type == 'topk':
            aim_outputs = aim_outputs.flatten()
            sample_idx = np.argsort(aim_outputs)[-batchsize:]
        
        select_goals = ags[sample_idx, :].copy()
        return select_goals



    def save_model(self):
        if self.use_ensemble:
            torch.save(self.aim_discriminator, os.path.join(self.model_path, 'aim_discriminator_'+'epoch_'+str(self.epoch)+'.pt'))
        else:
            torch.save(self.aim_discriminator.state_dict(),os.path.join(self.model_path, 'aim_discriminator_'+'epoch_'+str(self.epoch)+'.pt'))
    
    def load(self):
        path = os.path.join(self.model_path, 'aim_discriminator.pt')
        if os.path.exists(path):
            if self.use_ensemble:
                self.aim_discriminator = torch.load(path)
            else:
                self.aim_discriminator =  Discriminator(x_dim=self.env_params['goal'] * 2, reward_type=self.reward_type, lambda_coef = self.args.lambda_coef)
                self.aim_discriminator.load_state_dict(torch.load(path))
        else:
            pass
