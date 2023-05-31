import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from collections import deque
import csv
from envs.utils import goal_distance
import wandb


class Curricum_Agent(object):

    def __init__(self, args, policy, eval_agent, env, env_params, buffer, rollout_worker, her_sampler, o_norm, g_norm,
                 ag_density_estimator, dg_density_estimator, ag_buffer, dg_buffer, goal_sampler, goal_argumentor, 
                 goal_teacher, reward_teacher, logger):
        self.policy = policy
        self.env = env
        self.env_params = env_params
        self.args = args
        self.logger = logger
        self.o_norm = o_norm
        self.g_norm = g_norm
        self.ag_density_estimator = ag_density_estimator
        self.dg_density_estimator = dg_density_estimator
        # her sampler
        self.sampler = her_sampler
        # create the replay buffer
        self.buffer = buffer
        self.candidate_ags_buffer = ag_buffer
        self.candidate_dgs_buffer = dg_buffer
        self.goal_sampler = goal_sampler
        self.rollout_worker = rollout_worker
        self.eval_agent = eval_agent
        self.save_root = os.path.join(self.args.save_dir, self.args.env_name, self.args.alg, 'seed-'+str(self.args.seed))
        self.model_path = os.path.join(self.save_root, 'models')
        self.sigma = 1
        self.w_diverse = 0.5
        self.w_sim = 0.5
        self.best_success_rate = 0
        self.GoalAgmentor = goal_argumentor
        self.args.episodes = int(self.args.n_cycles * self.args.num_rollouts_per_mpi)
        self.goal_teacher = goal_teacher
        self.reward_teacher = reward_teacher
        self.save_interval = self.args.save_interval
        # if MPI.COMM_WORLD.Get_rank() == 0:
        os.makedirs(self.save_root, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)




    def learn(self):
        """
        train the network
        """
        headers = ['epoch', 'o_norm/mean', 'o_norm/std', 'g_norm/mean', 'g_norm/std', 'test/success_rate']
        with open(os.path.join(self.save_root, 'progress_' + str(self.env_params['seed'])+'.csv'), mode='w') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

        # start to collect samples
        for epoch in range(self.args.n_epochs):
            
            train_success = deque(maxlen=100)
         
            epoch_ags = []
            for cycle in range(self.args.n_cycles):

                obs, ag, dg, actions, mb_success = self.rollout_worker.generate_rollouts()
                train_success.extend(mb_success)
                episode_batch = [obs, ag, dg, actions]
                # print('cycle', cycle)
                self.buffer.store_episode(episode_batch)
                self._update_normalizer(episode_batch)
                # goal density model
                extend_ags = ag.copy().reshape((-1, self.env_params['goal']))
                extend_dgs = dg.copy().reshape((-1,  self.env_params['goal']))
                epoch_ags += extend_ags.tolist()

                idxs = np.where(extend_ags[:, 2] > 0.4)
                extend_ags = extend_ags[idxs[0],:]
                
                self.candidate_ags_buffer.extend(extend_ags)
                self.candidate_dgs_buffer.extend(extend_dgs)


                ##fit density model
                if (cycle+1) % self.args.fit_interval == 0:

                    if self.args.episode_priority and epoch>=1:
                        if self.args.traj_rank_method == 'entropy':
                            self.buffer.fit_density_model()
                        mean_diverse_score, mean_sim_score = self.buffer.update_episode_priority(self.sigma, self.w_diverse, self.w_sim)
                        #print('{}th epoch: successful update episode prioritize'.format(epoch))
                        self.logger.add_scalar('mean_diverse_score', mean_diverse_score, epoch)
                        self.logger.add_scalar('mean_sim_score', mean_sim_score, epoch)

                    if self.rollout_worker.env_steps >= self.args.warm_up_steps:
                        if self.args.goal_teacher:
                            self.goal_sampler.update(epoch)
                
                        if self.args.reward_teacher:
                            self.reward_teacher.update()
                   
                for n in range(self.args.n_batches):
                    # train the network
                    self.policy._update_network()


            if self.args.goal_teacher and self.args.teacher_method == 'HGG':
                selection_trajectory_idx = {}
                for i in range(self.args.episodes):
                    if goal_distance(self.rollout_worker.achieved_trajectories[i][0], self.rollout_worker.achieved_trajectories[i][-1]) > 0.01:
                        selection_trajectory_idx[i] = True
                for idx in selection_trajectory_idx.keys():
                    self.goal_teacher.achieved_trajectory_pool.insert(self.rollout_worker.achieved_trajectories[idx].copy(), self.rollout_worker.achieved_init_states[idx].copy())
                self.rollout_worker.achieved_init_states = []
                self.rollout_worker.achieved_trajectories = []
                # print('MPI rank',MPI.COMM_WORLD.Get_rank(),'trajectory counter:', self.goal_teacher.achieved_trajectory_pool.counter)
            # start to do the evaluation
            success_rate = self.eval_agent._eval_desired_goal()
            avg_train_success = np.mean(np.array(train_success))
            train_success = MPI.COMM_WORLD.allreduce(avg_train_success, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()

            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[{}] epoch is: {}, eval success rate is: {:.3f}'.format(datetime.now(), epoch, success_rate))
                o_norm_mean, o_norm_std = np.mean(self.o_norm.mean), np.var(self.o_norm.std)
                g_norm_mean, g_norm_std = np.mean(self.g_norm.mean), np.var(self.g_norm.std)
                data = {'epoch': epoch,
                        'o_norm/mean': o_norm_mean,
                        'o_norm/std': o_norm_std,
                        'g_norm/mean': g_norm_mean,
                        'g_norm/std': g_norm_std,
                        'test/success_rate': success_rate,
                        'train_success_rate': train_success}
                wandb.log(data)
                self.logger.add_scalar('buffer_occupancy_ratio', self.buffer.current_size/self.buffer.size, epoch)
                self.logger.add_scalar("o_norm/mean", o_norm_mean, epoch)
                self.logger.add_scalar("o_norm/std", o_norm_std, epoch)
                self.logger.add_scalar("g_norm/mean", g_norm_mean, epoch)
                self.logger.add_scalar('g_norm/std', g_norm_std, epoch)
                self.logger.add_scalar("test_success_rate", success_rate, epoch)
                self.logger.add_scalar("train_success_rate", np.mean(np.array(train_success)), epoch)
                # self.logger.add_scalar("transition_rewarded_ratio", np.mean(np.array(cycle_r_ratio)), epoch)
                with open(os.path.join(self.save_root, 'progress_' + str(self.env_params['seed'])+'.csv'), mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, o_norm_mean, o_norm_std, g_norm_mean, g_norm_std, success_rate])
                if self.args.save_data:
                    goal_save_path = os.path.join(self.save_root,'ags')
                    os.makedirs(goal_save_path, exist_ok=True)
                    with open(os.path.join(goal_save_path, 'epoch_'+str(epoch) + '.csv'), mode='a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerows(epoch_ags)

                if success_rate > self.best_success_rate:
                    self.best_success_rate = success_rate
                    torch.save({'o_norm_mean':self.o_norm.mean, 
                                'o_norm_std':self.o_norm.std, 
                                'g_norm_mean':self.g_norm.mean, 
                                'g_norm_std':self.g_norm.std,
                                'critic':self.policy.critic_network.state_dict(),
                                'v_critic':self.policy.V_critic.state_dict(), 
                                'v_critic_target':self.policy.V_critic_target.state_dict(),
                                'actor': self.policy.actor_network.state_dict()}, self.model_path + '/model_best.pt')
                if epoch % self.save_interval ==0:
                    torch.save({'o_norm_mean':self.o_norm.mean, 
                                'o_norm_std':self.o_norm.std, 
                                'g_norm_mean':self.g_norm.mean, 
                                'g_norm_std':self.g_norm.std,
                                'critic':self.policy.critic_network.state_dict(),
                                'v_critic':self.policy.V_critic.state_dict(), 
                                'v_critic_target':self.policy.V_critic_target.state_dict(),
                                'actor': self.policy.actor_network.state_dict()}, self.model_path + '/model_epoch_' + str(epoch)+ '.pt')
                            


    def eval_candidate_entropy(self, sample_num, density_estimator):
        if density_estimator.fitted_kde == None:
            density_estimator.fit(n_kde_samples=5000)
        # goals = density_estimator._sample(sample_num)
        if density_estimator.name == 'achieved_goal':
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
        transitions = self.sampler.sample_for_normlization(buffer_temp, num_transitions)

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
