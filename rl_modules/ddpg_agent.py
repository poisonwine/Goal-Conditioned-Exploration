import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.replay_buffer import replay_buffer, goal_replay_buffer
from rl_modules.models import actor, critic, V_critic
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler
from collections import deque
from rl_modules.density import KernalDensityEstimator
from tensorboardX import SummaryWriter
from her_modules.goal_curriculum import Goal_Curriculum
import pandas as pd
import csv
import time
from her_modules.hgg import TrajectoryPool, MatchSampler
from envs.utils import goal_distance
from envs import make_env
"""
ddpg with HER (MPI-version)

"""
class ddpg_agent:
    def __init__(self, args, env, env_params, logger):
        self.args = args
        self.env = env
        self.env_params = env_params
        self.args.act_dim = env_params['action']
        self.args.episodes = int(self.args.n_cycles * self.args.num_rollouts_per_mpi)
        self.logger = logger
        self.env_steps = env_params['env_steps']
        # create the network
        self.actor_network = actor(env_params)
        self.critic_network = critic(env_params)
        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        # build up the target network
        self.actor_target_network = actor(env_params)
        self.critic_target_network = critic(env_params)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        if self.args.alg == "HGG":
            self.V_critic = V_critic(env_params)
            self.V_critic_target = V_critic(env_params)
            sync_networks(self.V_critic)
            self.V_critic_target.load_state_dict(self.V_critic.state_dict())
            self.V_critic_optim = torch.optim.Adam(self.V_critic.parameters(), lr=self.args.lr_critic)

        # if use gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.args.cuda:
            self.actor_network.to(self.device)
            self.critic_network.to(self.device)
            self.actor_target_network.to(self.device)
            self.critic_target_network.to(self.device)
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        # goal density estimator
        self.ag_density_estimator = KernalDensityEstimator(name='achieved_goal', logger=self.logger, sample_dim=self.env_params['goal'])
        self.dg_density_estimator = KernalDensityEstimator(name='desired_goal', logger=self.logger, sample_dim=self.env_params['goal'])
        self.robot_state_estimator = KernalDensityEstimator(name='robot_state', logger=self.logger, sample_dim=self.env_params['robot_state'])

        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.args, self.env.compute_reward)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_transitions, self.args)
        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
        # create the dict for store the model
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name, self.args.alg)
        self.sigma = 1
        self.w_diverse = 0.5
        self.w_sim = 0.5
        self.candidate_ags_buffer = goal_replay_buffer(name='candidate_ags', buffer_size=1500, sample_dim=env_params['goal'])
        self.candidate_dgs_buffer = goal_replay_buffer(name='candidate_dgs', buffer_size=1500, sample_dim=env_params['goal'])
        self.robot_state_buffer = goal_replay_buffer(name='robot_state', buffer_size=1000, sample_dim=env_params['robot_state'])
        self.GoalAgmentor = Goal_Curriculum(env_name=self.args.env_name, error_distance=self.args.error_dis, batch_size=self.args.argued_goal_size)
        os.makedirs(self.model_path, exist_ok=True)

        if self.args.alg == 'HGG':
            self.achieved_trajectory_pool = TrajectoryPool(self.args, self.args.hgg_pool_size)
            self.sampler = MatchSampler(self.args, self.achieved_trajectory_pool, self.V_critic, self.o_norm, self.g_norm)
            self.env_List = []
            for i in range(self.args.episodes):
                self.env_List.append(make_env(self.args))
        # if MPI.COMM_WORLD.Get_rank() == 0:
        #     if not os.path.exists(self.args.save_dir):
        #         os.mkdir(self.args.save_dir)
        #     # path to save the model
        #     self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
        #     if not os.path.exists(self.model_path):
        #         os.mkdir(self.model_path)

    def generate_rollouts(self, cycle):
        mb_obs, mb_ag, mb_g, mb_actions, mb_success_history = [], [], [], [], []
        for i in range(self.args.num_rollouts_per_mpi):
            # reset the rollouts
            ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
            # reset the environment
            if self.args.alg == 'HGG':
                observation = self.env_List[int(i+cycle)].get_obs()
                init_state = observation['observation'].copy()
                explore_goal = self.sampler.sample(int(i+cycle))
                self.env.goal = explore_goal.copy()
                observation = self.env.get_obs()
                self.achieved_init_states.append(init_state)
            elif self.args.goal_explore and self.env_steps >= self.args.warm_up_steps and self.args.density_estimate:
                if np.random.rand() > 0.5:
                    exp_goal = self.explore_goals[int(i+cycle)].copy()
                    self.env.goal = exp_goal.copy()
                    observation = self.env.get_obs()
                else:
                    observation = self.env.reset()
            else:
                observation = self.env.reset()
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']
            trajectory = [ag.copy()]
            # start to collect samples
            for t in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    action = self._select_actions(pi)
                # feed the actions into the environment
                if self.args.alg=="HGG":
                    observation_new, _, _, info = self.env_List[int(i+cycle)].step(action)
                else:
                    observation_new, _, _, info = self.env.step(action)
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']
                trajectory.append(ag_new.copy())
                # append rollouts
                ep_obs.append(obs.copy())
                ep_ag.append(ag.copy())
                ep_g.append(g.copy())
                ep_actions.append(action.copy())
                # re-assign the observation
                obs = obs_new
                ag = ag_new
            if self.args.alg =='HGG':
                self.achieved_trajectories.append(np.array(trajectory))
            mb_success_history.append(int(info['is_success']))
            ep_obs.append(obs.copy())
            ep_ag.append(ag.copy())
            mb_obs.append(ep_obs)
            mb_ag.append(ep_ag)
            mb_g.append(ep_g)
            mb_actions.append(ep_actions)
        # convert them into arrays
        mb_obs = np.array(mb_obs)
        mb_ag = np.array(mb_ag)
        mb_g = np.array(mb_g)
        mb_actions = np.array(mb_actions)
        return mb_obs, mb_ag, mb_g, mb_actions, mb_success_history




    def learn(self):
        """
        train the network
        """
        headers = ['epoch', 'o_norm/mean', 'o_norm/std', 'g_norm/mean', 'g_norm/std', 'test/success_rate']
        with open(os.path.join(self.model_path, 'progress_' + str(self.env_params['seed'])+'.csv'), mode='w') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        epoch_r_ratio = []
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            epoch_goal_pairs = []
            epoch_beh_goals = []
            cycle_r_ratio = []
            train_success = deque(maxlen=100)
            if self.env_steps >= self.args.warm_up_steps and self.args.density_estimate:
                ags, ag_log_density, ag_entropy = self.eval_candidate_entropy(sample_num=1000, density_estimator=self.ag_density_estimator)
                self.candidate_ags_buffer.extend(ags, ag_log_density, ag_entropy)
                dgs, dg_log_density, dg_entropy = self. eval_candidate_entropy(sample_num=1000, density_estimator=self.dg_density_estimator)
                self.candidate_dgs_buffer.extend(dgs, dg_log_density, dg_entropy)

                if self.args.goal_explore:
                    self.explore_goals = []
                    explore_goals = self.candidate_ags_buffer.sample(self.args.explore_goal_num)
                    self.explore_goals += list(explore_goals)
            if self.args.alg == "HGG":
                initial_goals = []
                desired_goals = []
                for i in range(self.args.episodes):
                    obs1 = self.env_List[i].reset()
                    goal_a = obs1['achieved_goal'].copy()
                    goal_d = obs1['desired_goal'].copy()
                    initial_goals.append(goal_a.copy())
                    desired_goals.append(goal_d.copy())
                self.sampler.update(initial_goals, desired_goals)
                self.achieved_trajectories = []
                self.achieved_init_states = []

            for cycle in range(self.args.n_cycles):

                obs, ag, dg, actions, dones = self.generate_rollouts(cycle)
                train_success.extend(dones)
                self.env_steps += self.args.num_rollouts_per_mpi * self.env_params['max_timesteps']
                episode_batch = [obs, ag, dg, actions]
                self.buffer.store_episode(episode_batch)
                self._update_normalizer(episode_batch)
                # goal density model
                extend_ags = ag.copy().reshape((-1, self.env_params['goal']))
                extend_dgs = dg.copy().reshape((-1,  self.env_params['goal']))
                self.ag_density_estimator.extend(extend_ags)
                self.dg_density_estimator.extend(extend_dgs)

                ##fit density model
                if (cycle+1) % self.args.fit_interval == 0:

                    if self.args.alg == 'mep' and self.args.episode_priority:
                        self.buffer.fit_density_model()
                        #print('epoch: successful fit the density model')
                    if self.args.episode_priority:
                        mean_diverse_score, mean_sim_score = self.buffer.update_episode_priority(self.sigma, self.w_diverse, self.w_sim)
                        #print('{}th epoch: successful update episode prioritize'.format(epoch))
                        self.logger.add_scalar('mean_diverse_score', mean_diverse_score, epoch)
                        self.logger.add_scalar('mean_sim_score', mean_sim_score, epoch)

                    self.ag_density_estimator.fit(n_kde_samples=5000)
                    self.dg_density_estimator.fit(n_kde_samples=5000)

                # store the episodes
                batch_ratio = []
                for n in range(self.args.n_batches):
                    # train the network
                    goal_pairs, r_ratio, beh_goal = self._update_network()
                    batch_ratio.append(r_ratio)
                    if (n+1) % 10 == 0:
                        epoch_goal_pairs += goal_pairs
                        epoch_beh_goals += beh_goal
                cycle_r_ratio.append(np.mean(np.array(batch_ratio)))
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)
                if self.args.alg=='HGG':
                    self._soft_update_target_network(self.V_critic_target, self.V_critic)
            goals = np.unique(np.array(epoch_beh_goals))
            if self.args.alg =='HGG':
                selection_trajectory_idx = {}
                for i in range(self.args.episodes):
                    if goal_distance(self.achieved_trajectories[i][0], self.achieved_trajectories[i][-1]) > 0.01:
                        selection_trajectory_idx[i] = True
                for idx in selection_trajectory_idx.keys():
                    self.achieved_trajectory_pool.insert(self.achieved_trajectories[idx].copy(), self.achieved_init_states[idx].copy())
            epoch_r_ratio.append(np.mean(np.array(cycle_r_ratio)))
            # start to do the evaluation
            success_rate = self._eval_agent()
            r_save_path = os.path.join(self.model_path, 'reward')
            os.makedirs(r_save_path, exist_ok=True)
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[{}] epoch is: {}, eval success rate is: {:.3f}'.format(datetime.now(), epoch, success_rate))
                o_norm_mean, o_norm_std = np.mean(self.o_norm.mean), np.var(self.o_norm.std)
                g_norm_mean, g_norm_std = np.mean(self.g_norm.mean), np.var(self.g_norm.std)
                data = {'epoch': epoch,
                        'o_norm/mean': o_norm_mean,
                        'o_norm/std': o_norm_std,
                        'g_norm/mean': g_norm_mean,
                        'g_norm/std': g_norm_std,
                        'test/success_rate': success_rate}
                self.logger.add_scalar('buffer_occupancy_ratio', self.buffer.current_size/self.buffer.size, epoch)
                self.logger.add_scalar("o_norm/mean", o_norm_mean, epoch)
                self.logger.add_scalar("o_norm/std", o_norm_std, epoch)
                self.logger.add_scalar("g_norm/mean", g_norm_mean, epoch)
                self.logger.add_scalar('g_norm/std', g_norm_std, epoch)
                self.logger.add_scalar("test_success_rate", success_rate, epoch)
                self.logger.add_scalar("train_success_rate", np.mean(np.array(train_success)), epoch)
                self.logger.add_scalar("transition_rewarded_ratio", np.mean(np.array(cycle_r_ratio)), epoch)
                with open(os.path.join(self.model_path, 'progress_' + str(self.env_params['seed'])+'.csv'), mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, o_norm_mean, o_norm_std, g_norm_mean, g_norm_std, success_rate])
                if self.args.save_data:
                    goal_analyse_path = os.path.join(self.model_path, 'goal_data')
                    os.makedirs(goal_analyse_path, exist_ok=True)
                    if (epoch+1) % 5 == 0 or epoch == 0:
                        with open(os.path.join(goal_analyse_path, 'goaldata_' + str(epoch) + '.csv'), 'w', newline='') as f:
                            writer = csv.writer(f)
                            goal_header = ['origin_dg', 'initial_ag', 'hindsight_goal']
                            writer.writerow(goal_header)
                            writer.writerows(epoch_goal_pairs)

                        with open(os.path.join(goal_analyse_path, 'beh_goal_' + str(epoch) + '.csv'), 'w', newline='') as f:
                            writer = csv.writer(f)
                            goal_header = ['behaviour_goal']
                            writer.writerow(goal_header)
                            writer.writerows(epoch_beh_goals)


                with open(os.path.join(r_save_path, 'r_ratio_per_cycle.csv'), mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(cycle_r_ratio)

                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()], self.model_path + '/model.pt')

        with open(os.path.join(r_save_path, 'r_ratio_per_epoch.csv'), mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(epoch_r_ratio)

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.to(self.device)
        return inputs
    
    # this function will choose action for the agent and do the exploration
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

    def eval_candidate_entropy(self, sample_num, density_estimator):
        if density_estimator.fitted_kde == None:
            density_estimator.fit(n_kde_samples=5000)
        #goals = density_estimator._sample(sample_num)
        if density_estimator.name=='achieved_goal':
            goals = density_estimator.random_sample(sample_num)
        else:
            goals = density_estimator.random_sample(sample_num)
        goals = np.unique(goals, axis=0)
        while goals.shape[0] == 0:
            goals = density_estimator.random_sample(sample_num)
            goals = np.unique(goals)
        if density_estimator.name == 'achieved_goal' and self.args.goal_arguement:
            goals = self.GoalAgmentor.goal_arguement(goals)
        g_entropy = self.dg_density_estimator.evaluate_elementwise_entropy(goals)
        g_entropy = g_entropy.reshape(-1, 1)
        g_log_density = self.dg_density_estimator.evaluate_log_density(goals)
        g_log_density = g_log_density.reshape(-1, 1)
        assert goals.shape[0] == g_entropy.shape[0]
        return goals, g_log_density, g_entropy


    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g, 
                       'actions': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_module.sample_her_trainsition_for_normalize(buffer_temp, num_transitions)

        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    def sample_instructive_transition(self, goal_buffer, batch_size):
        trans = self.buffer.sample(batch_size=batch_size, her=False)
        batch_size = trans['actions'].shape[0]
        candidate_goals = goal_buffer.sample(batch_size)
        trans['g'] = candidate_goals
        trans['r'] = np.expand_dims(self.env.compute_reward(trans['ag_next'], trans['g'], None), 1)
        instruct_transitions = {k: trans[k].reshape(batch_size, *trans[k].shape[1:])
                           for k in trans.keys()}
        return instruct_transitions
    # update the network
    def _update_network(self):
        # sample the episodes
        transitions, goal_pairs = self.buffer.sample(self.args.batch_size, her=True)
        if (self.env_steps > self.args.warm_up_steps) and self.args.density_estimate \
            and self.candidate_ags_buffer.current_size != 0 and self.candidate_dgs_buffer.current_size !=0:
            beh_batch_size = self.args.beh_batch_size
            if beh_batch_size != 0:
                beh_transitions = self.sample_instructive_transition(self.candidate_ags_buffer, beh_batch_size)
                for key in transitions.keys():
                    transitions[key] = np.vstack([transitions[key], beh_transitions[key].copy()])

            actual_batch_size = self.args.actual_batch_size
            if actual_batch_size != 0:
                actual_transitions = self.sample_instructive_transition(self.candidate_dgs_buffer, actual_batch_size)
                for key in transitions.keys():
                    transitions[key] = np.vstack([transitions[key], actual_transitions[key].copy()])
            beh_goal = beh_transitions['g'].copy()
        else: 
            beh_goal = transitions['g'].copy()
        beh_goal = list(beh_goal)
        reward_ratio = np.abs(np.sum(transitions['r'].flatten())) / transitions['r'].shape[0]

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
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
            if self.args.alg == 'HGG':
                v_next_value = self.V_critic_target(inputs_next_norm_tensor)
                v_next_value = v_next_value.detach()
                target_v_value = r_tensor + self.args.gamma * v_next_value
                target_v_value = target_v_value.detach()
                target_v_value = torch.clamp(target_v_value, -clip_return, 0)

        # the v loss
        if self.args.alg == "HGG":
            real_v_value = self.V_critic(inputs_norm_tensor)
            v_critic_loss = (target_v_value - real_v_value).pow(2).mean()
            self.V_critic_optim.zero_grad()
            v_critic_loss.backward()
            sync_grads(self.V_critic)
            self.V_critic_optim.step()
        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()

        return goal_pairs, reward_ratio, beh_goal
    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()
