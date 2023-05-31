import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import wandb
import os
from mpi4py import MPI


class ForwardInverseModel(nn.Module):
    def __init__(self, env_params, hidden_dim):
        super().__init__()
        self.env_params = env_params
        self.inverse_model = nn.Sequential(
            nn.Linear(env_params['obs'] *2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, env_params['action'])
        )

        self.forward_model = nn.Sequential(
            nn.Linear(env_params['obs'] + env_params['action'], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, env_params['obs'])
        )

    def forward(self, obs, action, next_obs, training=True):
        if training:
            # inverse prediction 
            im_input_tensor = torch.cat([obs, next_obs], dim=1)
            pred_action =  self.inverse_model(im_input_tensor)
            # forward_prediction
            fm_input_tensor = torch.cat([obs,action], dim=-1)
            pred_next_obs = self.forward_model(fm_input_tensor)
            return pred_action, pred_next_obs
        else:
            fm_input_tensor = torch.cat([obs, action],dim=-1)
            pred_next_obs = self.forward_model(fm_input_tensor)
            return pred_next_obs


class ICMTeacher(object):
    def __init__(self, args, env_params, o_norm, g_norm, buffer) -> None:
        self.args = args
        self.env_params = env_params
        self.lr = 1e-3
        self.o_normalizer = o_norm
        self.g_normalizer = g_norm
        self.buffer = buffer
        self.loss_beta = self.args.icm_beta # balance the forward and inverse loss
        self.reward_scale = self.args.icm_reward_scale
        self.inverse_forward_model = ForwardInverseModel(env_params=env_params, hidden_dim=128)
        self.im_loss = nn.MSELoss()
        self.fm_loss = nn.MSELoss()
        self.save_frequency = self.args.save_interval
        self.epoch = 0
        self.optimizer = torch.optim.Adam(lr=self.lr, params=self.inverse_forward_model.parameters())

        self.save_root = os.path.join(self.args.save_dir, self.args.env_name, self.args.alg)
        self.model_path = os.path.join(self.save_root, 'models')
        if MPI.COMM_WORLD.Get_rank() == 0:
            os.makedirs(self.model_path, exist_ok=True)

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        norm_o = self.o_normalizer.normalize(o)
        norm_g = self.g_normalizer.normalize(g)
        return norm_o, norm_g
     
    
    def sample_batch(self):
        if self.args.use_per:
            transitions, idxes = self.buffer.sample(self.args.batch_size)
        else:
            transitions = self.buffer.sample(self.args.batch_size)
        sampled_batch_size = transitions['obs'].shape[0]
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        obs = torch.tensor(transitions['obs'], dtype=torch.float32)
        actions = torch.tensor(transitions['actions'], dtype=torch.float32)
        next_obs = torch.tensor(transitions['obs_next'], dtype=torch.float32)
        return obs, actions, next_obs

    def update(self):
        forward_loss_sum = 0
        inverse_loss_sum = 0
        total_loss_sum = 0
        for iter in range(self.args.icm_iteration):
            obs, actions, next_obs = self.sample_batch()
            pred_actions, pred_next_obs = self.inverse_forward_model(obs, actions, next_obs)
            
            forward_loss = self.fm_loss(pred_next_obs, next_obs) * self.loss_beta
            
            inverse_loss = self.im_loss(pred_actions, actions)*(1-self.loss_beta)
            loss = forward_loss + inverse_loss
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            forward_loss_sum += forward_loss.item()
            inverse_loss_sum += inverse_loss.item()
            total_loss_sum += loss.item()
        if MPI.COMM_WORLD.Get_rank()==0:
            loss_result = {
                'forward_loss': forward_loss_sum / float(self.args.icm_iteration),
                'inverse_loss': inverse_loss_sum / float(self.args.icm_iteration),
                'total_loss':  total_loss_sum / float(self.args.icm_iteration),
            }
            wandb.log(loss_result)
            if self.epoch % self.save_frequency ==0:
                self.save_model()
        self.epoch += 1 

    def compute_reward(self, obs, actions, next_obs,clip=True):
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        action_tensor = torch.tensor(actions, dtype=torch.float32)
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32)
        with torch.no_grad():
            pred_next_state = self.inverse_forward_model(obs=obs_tensor, action=action_tensor, next_obs=next_obs_tensor, training=False)
            pred_next_state = pred_next_state.cpu().numpy()
        pred_state = np.clip(pred_next_state, -self.args.clip_obs, self.args.clip_obs)
        if clip:
            intri_r = np.clip(np.mean((pred_state-next_obs)**2, axis=1),0, 0.5).reshape(1,-1)
        else:
            intri_r = np.mean((pred_state-next_obs)**2, axis=1)
        return intri_r
        
    def save_model(self):
        torch.save(self.inverse_forward_model.state_dict(), os.path.join(self.model_path, 'icm_'+'epoch_'+str(self.epoch)+'.pt'))

            