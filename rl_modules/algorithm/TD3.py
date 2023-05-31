from mpi_utils.mpi_utils import sync_networks, sync_grads
import torch
import numpy as np
from rl_modules.models import actor, critic,BVNCritic, MRNCritic
import math
import torch.nn as nn
import torch.nn.functional as F


class td3_agent(object):
    def __init__(self, env, env_params, agent_params, args, buffer, o_norm, g_norm, reward_teacher):
        self.env = env
        self.args = args
        self.env_params = env_params
        self.agent_params = agent_params
        self.buffer = buffer

        self.o_norm = o_norm
        self.g_norm = g_norm
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.total_iter = 0  #

        # configure for td3 agent
        self.gamma = self.agent_params['gamma']
        self.epsilon_init = self.agent_params['epsilon_init']
        self.epsilon_min = self.agent_params['epsilon_min']
        self.decay = self.agent_params['decay']
        self.min_a = -self.env_params['action_max']
        self.max_a = self.env_params['action_max']
        self.policy_delay = self.agent_params['policy_delay']
        self.noisy_range = self.agent_params['noisy_range']
        self.policy_noise = self.agent_params['policy_noise']
        self.OU_noise = self.agent_params['OU_noise']
        self.max_grad_norm = self.agent_params['max_grad_norm']

        self.iter = 0
        self.epsilon = lambda x: self.epsilon_min + (self.epsilon_init - self.epsilon_min) * math.exp(- x / self.decay)


        ## network configuration
        self.actor_network = actor(env_params)
        self.actor_target_network = actor(env_params)

        if self.args.critic_type == 'BVN':
            self.critic_network = BVNCritic(env_params, self.args.bvn_hidden_dim)
            self.critic_target_network = BVNCritic(env_params, self.args.bvn_hidden_dim)
            self.critic_network2 = BVNCritic(env_params, self.args.bvn_hidden_dim)
            self.critic_target_network2 = BVNCritic(env_params, self.args.bvn_hidden_dim)

        elif self.args.critic_type == 'MRN':
            self.critic_network = MRNCritic(env_params, emb_dim=self.args.mrn_emb_dim, hidden_dim=self.args.mrn_hidden_dim)
            self.critic_target_network = MRNCritic(env_params, emb_dim=self.args.mrn_emb_dim, hidden_dim=self.args.mrn_hidden_dim)
            self.critic_network2 = MRNCritic(env_params, emb_dim=self.args.mrn_emb_dim, hidden_dim=self.args.mrn_hidden_dim)
            self.critic_target_network2 = MRNCritic(env_params, emb_dim=self.args.mrn_emb_dim, hidden_dim=self.args.mrn_hidden_dim)

        elif self.args.critic_type == 'monolithic':
            self.critic_network = critic(env_params)
            self.critic_target_network = critic(env_params)
            self.critic_network2 = critic(env_params)
            self.critic_target_network2 = critic(env_params)

        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        sync_networks(self.critic_network2)

        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        self.critic_target_network2.load_state_dict(self.critic_network2.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic1_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        self.critic2_optim = torch.optim.Adam(self.critic_network2.parameters(), lr=self.args.lr_critic)
        self.buffer = buffer

        self.o_norm = o_norm
        self.g_norm = g_norm
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.count = 0
        self.reward_teacher = reward_teacher

    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    def soft_update(self):
        self._soft_update_target_network(self.actor_target_network, self.actor_network)
        self._soft_update_target_network(self.critic_target_network, self.critic_network)
        self._soft_update_target_network(self.critic_target_network2, self.critic_network2)


    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g


    def _update_network(self):
        self.total_iter += 1
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
            r = self.reward_teacher.compute_reward(transitions['ag_next'], transitions['g']).reshape(-1, 1)
            r_tensor = torch.tensor(r, dtype=torch.float32)
        else:
            r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)
        # r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.to(self.device)
            inputs_next_norm_tensor = inputs_next_norm_tensor.to(self.device)
            actions_tensor = actions_tensor.to(self.device)
            r_tensor = r_tensor.to(self.device)

        with torch.no_grad():

            target_next_action = self.actor_target_network(inputs_next_norm_tensor)
            noise = (torch.rand_like(actions_tensor) * self.policy_noise).clamp(-self.noisy_range, self.noisy_range)
            target_next_action = target_next_action + noise
            target_next_action = torch.clamp(target_next_action, self.min_a, self.max_a)
            q_min = torch.min(self.critic_target_network(inputs_next_norm_tensor, target_next_action),
                              self.critic_target_network2(inputs_next_norm_tensor, target_next_action))
            # q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            target_q_value = r_tensor + self.gamma * q_min.detach()

            q1 = self.critic_network(inputs_norm_tensor, actions_tensor)
            td_errors1 = (q1 - target_q_value).abs().detach().cpu().data.numpy().flatten()
            q2 = self.critic_network2(inputs_norm_tensor, actions_tensor)
            td_errors2 = (q2 - target_q_value).abs().detach().cpu().data.numpy().flatten()

            if self.args.use_per:
                td_e = 0.5 *(td_errors1 + td_errors2)
                self.buffer.update_priorities(idxes, td_e)

            if self.args.use_laber:

                probs1 = td_errors1 / td_errors1.sum()
                indices1 = np.random.choice(int(self.args.m_factor * self.args.batch_size), self.args.batch_size,
                                            p=probs1)
                td_errors_for_selected_indices1 = td_errors1[indices1]
                inputs_norm_tensor1 = inputs_norm_tensor[indices1]
                actions_tensor1 = actions_tensor[indices1]
                q_targets1 = target_q_value[indices1]
                # Compute the weights for SGD update
                # LaBER:
                loss_weights1 = (1.0 / td_errors_for_selected_indices1) * td_errors1.mean()
                loss_weights1 = torch.from_numpy(loss_weights1).unsqueeze(1)

                # for the second critic

                probs2 = td_errors2 / td_errors2.sum()
                indices2 = np.random.choice(int(self.args.m_factor * self.args.batch_size), self.args.batch_size,
                                            p=probs2)
                td_errors_for_selected_indices2 = td_errors2[indices1]
                inputs_norm_tensor2 = inputs_norm_tensor[indices2]
                actions_tensor2 = actions_tensor[indices2]
                q_targets2 = target_q_value[indices2]
                # Compute the weights for SGD update
                # LaBER:
                loss_weights2 = (1.0 / td_errors_for_selected_indices2) * td_errors1.mean()
                loss_weights2 = torch.from_numpy(loss_weights2).unsqueeze(1)


            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)

        if self.args.use_laber:
            real_q_value1 = self.critic_network(inputs_norm_tensor1, actions_tensor1)
            value_loss1 = (0.5 * F.mse_loss(real_q_value1, q_targets1, reduction="none").cpu() * loss_weights1).mean()
            real_q_value2 = self.critic_network2(inputs_norm_tensor2, actions_tensor2)
            value_loss2 = (0.5 * F.mse_loss(real_q_value2, q_targets2, reduction="none").cpu() * loss_weights2).mean()
        else:
            real_q_value1 = self.critic_network(inputs_norm_tensor, actions_tensor)
            value_loss1 = (real_q_value1 - target_q_value).pow(2).mean()
            real_q_value2 = self.critic_network2(inputs_norm_tensor, actions_tensor)
            value_loss2 = (real_q_value2 - target_q_value).pow(2).mean()


        self.critic1_optim.zero_grad()
        value_loss1.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic_network.parameters(), self.max_grad_norm)
        sync_grads(self.critic_network)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        value_loss2.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic_network2.parameters(), self.max_grad_norm)
        sync_grads(self.critic_network2)
        self.critic2_optim.step()

        if self.args.use_laber:
            indice_actor = np.random.randint(sampled_batch_size, sampled_batch_size // self.args.m_factor, )
            inputs_norm_tensor = inputs_norm_tensor[indice_actor]

        if self.total_iter % self.policy_delay == 0:
            current_action = self.actor_network(inputs_norm_tensor)
            policy_loss = (-self.critic_network(inputs_norm_tensor, current_action)).mean()
            self.actor_optim.zero_grad()
            policy_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.actor_network.parameters(), self.max_grad_norm)
            sync_grads(self.actor_network)
            self.actor_optim.step()
            self.soft_update()

        if self.total_iter % self.args.n_batches == 0:
            self.count += 1

    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.epsilon_min * self.max_a * np.random.randn(*action.shape)
        action = np.clip(action, self.min_a, self.max_a)
        # random actions...
        if self.OU_noise:
            random_actions = np.random.uniform(low=self.min_a, high=self.max_a, size=self.env_params['action'])
        # choose if use the random actions
            action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action




