import copy
import numpy as np
from envs import make_env
from envs.utils import goal_distance
from utils.gcc_utils import gcc_load_lib, c_double, c_int
import ctypes 
import torch
import csv
from mpi4py import MPI
def goal_concat(obs, goal):
    return np.concatenate([obs, goal], axis=0)


class TrajectoryPool:
    def __init__(self, args, pool_length):
        self.args = args
        self.length = pool_length

        self.pool = []
        self.pool_init_state = []
        self.counter = 0

    def insert(self, trajectory, init_state):
        if self.counter < self.length:
            self.pool.append(trajectory.copy())
            self.pool_init_state.append(init_state.copy())
        else:
            self.pool[self.counter % self.length] = trajectory.copy()
            self.pool_init_state[self.counter % self.length] = init_state.copy()
        self.counter += 1

    def pad(self):
        if self.counter >= self.length:
            return copy.deepcopy(self.pool), copy.deepcopy(self.pool_init_state)
        pool = copy.deepcopy(self.pool)
        pool_init_state = copy.deepcopy(self.pool_init_state)
        while len(pool) < self.length:
            pool += copy.deepcopy(self.pool)
            pool_init_state += copy.deepcopy(self.pool_init_state)
        return copy.deepcopy(pool[:self.length]), copy.deepcopy(pool_init_state[:self.length])


class MatchSampler:
    def __init__(self, args, env, achieved_trajectory_pool, policy, o_normalizer, g_normalizer):
        self.args = args
        self.env = env
        #self.env_test = make_env(args)
        self.dim = np.prod(self.env.reset()['achieved_goal'].shape)
        self.delta = self.env.distance_threshold
        self.critic = policy.critic_network
        
        # self.v_critic = policy.critic_network
        self.policy = policy
        if hasattr(policy, 'actor_target_network'):
            self.actor = policy.actor_target_network
        else:
            self.actor = policy.actor_network

        self.o_norm = o_normalizer
        self.g_norm = g_normalizer
        self.length = args.episodes
        init_goal = self.env.reset()['achieved_goal'].copy()
        self.pool = np.tile(init_goal[np.newaxis, :], [self.length, 1]) + np.random.normal(0, self.delta, size=(self.length, self.dim))
        self.init_state = self.env.reset()['observation'].copy()

        self.match_lib = ctypes.cdll.LoadLibrary('/data1/ydy/RL/HER_v3/her_modules/cost_flow.so')
        self.achieved_trajectory_pool = achieved_trajectory_pool

        # estimating diameter
        self.max_dis = 0
        for i in range(1000):
            obs = self.env.reset()
            dis = goal_distance(obs['achieved_goal'], obs['desired_goal'])
            if dis > self.max_dis: self.max_dis = dis

    def add_noise(self, pre_goal, noise_std=None):
        goal = pre_goal.copy()
        dim = 2 if self.args.env_name[:5] == 'Fetch' else self.dim
        if noise_std is None:
            noise_std = self.delta
        goal[:dim] += np.random.normal(0, noise_std, size=dim)
        return goal.copy()

    def sample(self, idx):
        if self.args.env_name[:5] == 'Fetch':
            return self.add_noise(self.pool[idx])
        else:
            return self.pool[idx].copy()

    def find(self, goal):
        res = np.sqrt(np.sum(np.square(self.pool - goal), axis=1))
        idx = np.argmin(res)
        # if test_pool:
        #     self.args.logger.add_record('Distance/sampler', res[idx])
        return self.pool[idx].copy()

    def update(self, initial_goals, desired_goals):
        if self.achieved_trajectory_pool.counter == 0:
            self.pool = np.array(copy.deepcopy(desired_goals))
            return

        achieved_pool, achieved_pool_init_state = self.achieved_trajectory_pool.pad()
        candidate_goals = []
        candidate_edges = []
        candidate_id = []

        achieved_value = []
        
        for i in range(len(achieved_pool)):
            # #obs = [goal_concat(achieved_pool_init_state[i], achieved_pool[i][j]) for j in
            #        range(achieved_pool[i].shape[0])]
            proc_o = np.clip(np.array(achieved_pool_init_state[i]), -self.args.clip_obs, self.args.clip_obs)
            norm_o = self.o_norm.normalize(proc_o)
            achieved_goals = np.array([achieved_pool[i][j] for j in range(achieved_pool[i].shape[0])])
            proc_g = np.clip(achieved_goals, -self.args.clip_obs, self.args.clip_obs)
            norm_g = self.g_norm.normalize(proc_g)
            input_obs = np.array([goal_concat(norm_o, norm_g[i, :]) for i in range(achieved_goals.shape[0])]) # [ts, dim]


            input_obs_tensor = torch.tensor(input_obs, dtype=torch.float32)
            with torch.no_grad():
                n_samples = 30
                tiled_obs = torch.tile(input_obs_tensor, (n_samples, 1, 1)).view((-1, input_obs_tensor.shape[-1])) # [ts, dim] --> [ts*n_samples, dim]
                if self.args.agent == 'DDPG' or 'TD3':
                    pi = self.actor(tiled_obs) # [ts*n_samples, dim(4)]
                    actions = torch.from_numpy(self.policy._select_actions(pi))
                elif self.args.agent == 'SAC':
                    mu, log_sigma = self.actor(tiled_obs)
                    actions = torch.from_numpy(self.policy._select_actions((mu, log_sigma)))
                
                q_value = self.critic(tiled_obs, actions)  # [ts*n_samples, 1]
                q_value = q_value.view(n_samples, -1, q_value.shape[-1])  # [n_samples, ts, 1]
                value = torch.mean(q_value, dim=0).detach().numpy().flatten()
                # value = self.critic(input_obs_tensor).detach().numpy().flatten()
                # value = q_value.detach().numpy().flatten()
                value = np.clip(value, -1.0 / (1.0 - self.args.gamma), 0)
                
                achieved_value.append(value.copy())

        value_mean= MPI.COMM_WORLD.allreduce(np.array(achieved_value).mean(), op=MPI.SUM)
        value_min= MPI.COMM_WORLD.allreduce(np.array(achieved_value).min(), op=MPI.SUM)
        value_max= MPI.COMM_WORLD.allreduce(np.array(achieved_value).max(), op=MPI.SUM)
        record_value = np.array([value_mean / MPI.COMM_WORLD.Get_size(), \
                                value_min / MPI.COMM_WORLD.Get_size(), \
                                value_max / MPI.COMM_WORLD.Get_size()]).tolist()
        # if MPI.COMM_WORLD.Get_rank() == 0:
        #     with open('value.csv',mode='a',newline='') as f:
        #         writer = csv.writer(f)
        #         writer.writerow(record_value)

        n = 0
        graph_id = {'achieved': [], 'desired': []}
        for i in range(len(achieved_pool)):
            n += 1
            graph_id['achieved'].append(n)
        for i in range(len(desired_goals)):
            n += 1
            graph_id['desired'].append(n)
        n += 1
        self.match_lib.clear(n)

        for i in range(len(achieved_pool)):
            self.match_lib.add(0, graph_id['achieved'][i], 1, 0)
        for i in range(len(achieved_pool)):



            for j in range(len(desired_goals)):
                res = np.sqrt(np.sum(np.square(achieved_pool[i] - desired_goals[j]), axis=1)) - achieved_value[i] / (
                            self.args.hgg_L / self.max_dis / (1 - self.args.gamma))
                match_dis = np.min(res) + goal_distance(achieved_pool[i][0], initial_goals[j]) * self.args.hgg_c
                match_idx = np.argmin(res)

                edge = self.match_lib.add(graph_id['achieved'][i], graph_id['desired'][j], 1, c_double(match_dis))
                candidate_goals.append(achieved_pool[i][match_idx])
                candidate_edges.append(edge)
                candidate_id.append(j)
        for i in range(len(desired_goals)):
            self.match_lib.add(graph_id['desired'][i], n, 1, 0)

        match_count = self.match_lib.cost_flow(0, n)
        assert match_count == self.length

        explore_goals = [0] * self.length
        for i in range(len(candidate_goals)):
            if self.match_lib.check_match(candidate_edges[i]) == 1:
                explore_goals[candidate_id[i]] = candidate_goals[i].copy()
        assert len(explore_goals) == self.length
        self.pool = np.array(explore_goals)



