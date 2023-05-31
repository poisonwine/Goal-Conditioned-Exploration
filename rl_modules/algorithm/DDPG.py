from rl_modules.models import actor, critic, BVNCritic, MRNCritic, V_critic
from mpi_utils.mpi_utils import sync_networks, sync_grads
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os


class ddpg_agent(object):
    def __init__(self, env, env_params, ddpg_config, args, buffer, o_norm, g_norm, reward_teacher):
        self.env = env
        self.args = args
        self.env_params = env_params
        self.agent_params = ddpg_config
        self.actor_network = actor(env_params)
        self.actor_target_network = actor(env_params)
        self.V_critic = V_critic(env_params=env_params)
        self.V_critic_target = V_critic(env_params=env_params)
        if self.args.critic_type=='BVN':
            self.critic_network = BVNCritic(env_params, self.args.bvn_hidden_dim)
            self.critic_target_network = BVNCritic(env_params, self.args.bvn_hidden_dim)
        elif self.args.critic_type=='MRN':
            self.critic_network = MRNCritic(env_params, emb_dim=self.args.mrn_emb_dim, hidden_dim=self.args.mrn_hidden_dim)
            self.critic_target_network = MRNCritic(env_params, emb_dim=self.args.mrn_emb_dim, hidden_dim=self.args.mrn_hidden_dim)
        elif self.args.critic_type =='monolithic':
            self.critic_network = critic(env_params)
            self.critic_target_network = critic(env_params)
        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        sync_networks(self.V_critic)
        # load the weights into the target networks
       
             
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        self.V_critic_target.load_state_dict(self.V_critic.state_dict())


        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        self.v_critic_optim = torch.optim.Adam(self.V_critic.parameters(), lr=self.args.lr_critic)
        self.buffer = buffer
        self.o_norm = o_norm
        self.g_norm = g_norm
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_decay = self.agent_params['policy_decay']
        self.max_grad_norm = self.agent_params['max_grad_norm']
        self.total_iter = 0
        self.reward_teacher = reward_teacher


    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g




    def _update_network(self):
        self.total_iter += 1
        # sample the episodes
        if self.args.use_per:
            transitions, idxes = self.buffer.sample(self.args.batch_size)
        else:
            transitions = self.buffer.sample(self.args.batch_size)
        sampled_batch_size = transitions['obs'].shape[0]
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)

        if self.args.intrinisic_r:
            if self.args.reward_method == 'aim':
                r = self.reward_teacher.compute_reward(transitions['ag_next'], transitions['g']).reshape(-1, 1)
                r_tensor = torch.tensor(r, dtype=torch.float32)
            elif self.args.reward_method=='mine':
                extra_r = self.reward_teacher.compute_reward(transitions['obs'], transitions['obs_next'])
                r_tensor= torch.tensor(extra_r + transitions['r'], dtype=torch.float32)
            elif self.args.reward_method =='icm':
                extra_r = self.reward_teacher.compute_reward(transitions['obs'], transitions['action'], transitions['obs_next'])
                r_tensor= torch.tensor(extra_r + transitions['r'], dtype=torch.float32)
        else:
            r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)

        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.to(self.device)
            inputs_next_norm_tensor = inputs_next_norm_tensor.to(self.device)
            actions_tensor = actions_tensor.to(self.device)
            r_tensor = r_tensor.to(self.device)
        # calculate the target Q value function and V function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            q = self.critic_network(inputs_norm_tensor, actions_tensor)
            td_errors = (q - target_q_value).abs().detach().cpu().data.numpy().flatten()
            if self.args.use_per:
                self.buffer.update_priorities(idxes, td_errors)

            if self.args.use_laber:
                probs = td_errors / td_errors.sum()
                indices = np.random.choice(sampled_batch_size, sampled_batch_size // self.args.m_factor, p=probs)
                td_errors_for_selected_indices = td_errors[indices]
                inputs_norm_tensor_laber = inputs_norm_tensor[indices]
                actions_tensor_laber = actions_tensor[indices]
                q_targets_laber = target_q_value[indices]
                # Compute the weights for SGD update
                # LaBER:
                loss_weights1 = (1.0 / td_errors_for_selected_indices) * td_errors.mean()
                loss_weights1 = torch.from_numpy(loss_weights1).unsqueeze(1)

            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)


            # update V network
            
            v_next = self.V_critic_target(inputs_next_norm_tensor)
            target_v = r_tensor + self.args.gamma* v_next
            target_v = torch.clamp(target_v, -clip_return, clip_return)
            target_v = target_v.detach()

        if self.args.use_laber:
            real_q_value = self.critic_network(inputs_norm_tensor_laber, actions_tensor_laber)
            critic_loss = (0.5 * F.mse_loss(real_q_value, q_targets_laber, reduction="none").cpu() * loss_weights1).mean()
        else:
        # the q loss
            real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
            critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the actor loss
        if self.args.use_laber:
            actions_real = self.actor_network(inputs_norm_tensor_laber)
            actor_loss = - self.critic_network(inputs_norm_tensor_laber, actions_real).mean()
        else:
            actions_real = self.actor_network(inputs_norm_tensor)
            actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()

        if self.args.use_pretrain_policy:
            with torch.no_grad():
                actions_pretrain = self.pretrain_actor(inputs_norm_tensor)
                q_pre = self.pretrain_qcritic(inputs_norm_tensor, actions_pretrain)
                real_q = real_q_value.detach()
                adv = real_q - q_pre
                weights = torch.clamp(adv.exp(), 0, 10.0)
        
            bc_loss = torch.mean(weights * torch.square(actions_real - actions_pretrain))

            actor_loss += bc_loss
            
        real_v = self.V_critic(inputs_norm_tensor)

        v_loss = (target_v- real_v).pow(2).mean()


        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.actor_network.parameters(), self.max_grad_norm)
        sync_grads(self.actor_network)
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic_network.parameters(), self.max_grad_norm)
        sync_grads(self.critic_network)
        self.critic_optim.step()


        # update V critic
        self.v_critic_optim.zero_grad()
        v_loss.backward()
        sync_grads(self.V_critic)
        self.v_critic_optim.step()

        
        if self.total_iter % self.policy_decay == 0:
            self.soft_update()
        
        if self.total_iter % (self.args.n_batches * self.args.fit_rate * 0.5):
            self.soft_update_v()


    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    def soft_update(self):
        self._soft_update_target_network(self.actor_target_network, self.actor_network)
        self._soft_update_target_network(self.critic_target_network, self.critic_network)
    
    def soft_update_v(self):
        self._soft_update_target_network(self.V_critic_target, self.V_critic)


    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action