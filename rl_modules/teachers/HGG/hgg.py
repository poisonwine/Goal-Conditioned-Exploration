import math
import torch
import numpy as np
from utils.goal_utils import sample_uniform_goal
from rl_modules.teachers.HGG.matchsampler import MatchSampler,TrajectoryPool
from rl_modules.teachers.abstract_teacher import AbstractTeacher




class HGGteacher(AbstractTeacher):
    def __init__(self,args, env_params, env, policy, o_norm, g_norm, ag_buffer):
        self.args = args
        self.env_params = env_params
        self.env = env
        self.policy = policy
        self.o_norm = o_norm
        self.g_norm = g_norm

        self.achieved_trajectory_pool = TrajectoryPool(self.args, self.args.hgg_pool_size)
        self.ag_buffer = ag_buffer
        self.sampler = MatchSampler(self.args, self.env, self.achieved_trajectory_pool, self.policy, self.o_norm, self.g_norm)


    def update(self):
        initial_goals = []
        desired_goals = []
        for i in range(self.args.episodes):
            obs = self.env.reset()
            goal_a = obs['achieved_goal'].copy()
            goal_d = obs['desired_goal'].copy()
            initial_goals.append(goal_a.copy())
            desired_goals.append(goal_d.copy())
        self.sampler.update(initial_goals, desired_goals)


    def sample(self, batchsize):
        goal_pool = self.sampler.pool.copy()
        goal_num = goal_pool.shape[0]
        # idxs = np.where(goal_pool[:, 2] > 0.4)
        # filtered_goal = goal_pool[idxs[0],:]
        # if batchsize <= goal_num:
        inds = np.random.randint(0, goal_pool.shape[0], size=batchsize)
        selected_goals = goal_pool[inds].copy()
        # for i in range(batchsize):
        #     selected_goals[i] = self.sampler.add_noise(selected_goals[i])
    
        return selected_goals


    def save(self, path):

        pass
    def load(self, path):
        pass





