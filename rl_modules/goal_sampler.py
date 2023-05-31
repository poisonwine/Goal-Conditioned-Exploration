from collections import deque
import numpy as np
from mpi4py import MPI
import csv
import os
import scipy.spatial.distance as scdis
import torch
import wandb

class GoalSampler(object):
    def __init__(self,env, args, ag_estimator, dg_estimator, ag_buffer, dg_buffer, replay_buffer, policy, teacher, o_norm, g_norm, evaluator):
        self.args = args
        self.env = env
        self.ag_estimator = ag_estimator
        self.dg_estimator = dg_estimator
        self.ag_buffer = ag_buffer
        self.dg_buffer = dg_buffer
        self.replay_buffer = replay_buffer
        self.o_norm = o_norm
        self.g_norm = g_norm
        self.policy = policy
        self.candidate_goal = []
        self.goal_teacher = teacher
        self.pool = []
        self.goal_save_root = os.path.join(self.args.save_dir, args.env_name, args.alg, 'seed-'+str(args.seed),'exploration_goal')
        if MPI.COMM_WORLD.Get_rank() == 0:
            os.makedirs(self.goal_save_root, exist_ok=True)
        self.step = 0
        self.epoch = 0
        self.balance = 2.0
        self.v_critic = policy.V_critic_target
        self.max_dis = 0
        for i in range(500):
            obs = self.env.reset()
            dis = np.linalg.norm(obs['achieved_goal']-obs['desired_goal'], axis=-1)
            if dis > self.max_dis: self.max_dis = dis
        self.candidate_num = 100
        self.evaluator = evaluator

        self.max_value = (1-self.args.gamma**(int(env._max_episode_steps)))/(1-self.args.gamma)
        print('max value',self.max_value,'max distance',self.max_dis)

    def sample(self, obs, init_goal, desired_goal):
        if not len(self.candidate_goal):
            dgs = self.dg_buffer.random_sample(batch_size=self.candidate_num)
            self.candidate_goal.extend(dgs)
        # if np.random.rand() > 0.5:
        if self.args.teacher_method=='AGE' and self.args.curriculum_select:
            # obs = np.array(obs).reshape(1,-1)
            # clip_obs = np.clip(obs, -self.args.clip_obs, self.args.clip_obs)
            # norm_o = self.o_norm.normalize(clip_obs)
            # proc_g = np.clip(np.array(self.candidate_goal), -self.args.clip_obs, self.args.clip_obs)
            # g_norm = self.g_norm.normalize(proc_g)
            # obs_norm = np.repeat(norm_o, repeats=np.array(self.candidate_goal).shape[0], axis=0) 
            # inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
            # input_norm_obs = torch.tensor(inputs_norm, dtype=torch.float32)
            # with torch.no_grad():
            #     v = self.v_critic(input_norm_obs).numpy().reshape(1,-1)
            #     value = np.clip(v, -1.0 / (1.0 - self.args.gamma), 0)
            # norm_dg = self.g_norm.normalize(desired_goal.reshape(1,-1))
            # target_input = torch.tensor(np.concatenate([norm_o, norm_dg],axis=1),dtype=torch.float32)
            # with torch.no_grad():
            #     v = self.v_critic(target_input).numpy().reshape(1,-1)
            #     target_value = np.clip(v, -1.0 / (1.0 - self.args.gamma), 0)
            init2goal_dis = scdis.cdist(init_goal.reshape(1,-1), np.array(self.candidate_goal))
            prob1 = init2goal_dis / init2goal_dis.sum()
            goal2desired =  scdis.cdist(desired_goal.reshape(1,-1), np.array(self.candidate_goal))
            # distance = goal2desired - v_current / (1-)
            # cudis = (goal2desired / self.max_dis) *self.max_value + value
            # value_diff = np.abs(value - target_value)
            # max_value_diff = np.max(value_diff)
            # prob2 = (max_value_diff - value_diff) / np.sum(max_value_diff - value_diff)
            # print('max prob2',np.max(prob2), 'max_prob1', np.max(prob1))

            max_dis = np.max(goal2desired)
            prob2 = (max_dis-goal2desired) / np.sum(max_dis-goal2desired)
            prob = prob1 + self.balance* prob2
            # idx = int(np.argmin(cudis, axis=1))
            idx = int(np.argmax(prob, axis=1))
            #  selected_goal[:2] += np.random.normal(0, 0.03, size=2)
        else:
            idx = np.random.randint(len(self.candidate_goal))
        selected_goal = self.candidate_goal[idx].copy()
        self.candidate_goal.pop(idx)
        # selected_goal[:2] += np.random.normal(0, 0.03, size=2)
        if MPI.COMM_WORLD.Get_rank() == 0:
            with open(os.path.join(self.goal_save_root, 'selected_goal_epoch_'+str(self.epoch)+'.csv'), mode='a',newline='') as f:
                writer = csv.writer(f)
                writer.writerow(selected_goal.tolist())
       
        return selected_goal

    def update(self, epoch):
        self.epoch = epoch
        # exp_success_rate = self.evaluator._eval_exploration_goal(self.candidate_goal)
      
        if self.args.goal_teacher:
            exp_success_rate = self.evaluator._eval_exploration_goal(self.candidate_goal)
            self.goal_teacher.update()
            if self.args.q_cuttoff and self.args.teacher_method=='AGE':
                selected_goals, value = self.goal_teacher.sample(batchsize=self.candidate_num)
                # print(selected_goals.shape, value.shape)
                content = np.concatenate((selected_goals, value), axis=-1)
            else:
                selected_goals = self.goal_teacher.sample(batchsize=self.candidate_num)
                content = selected_goals.copy()
            
            if MPI.COMM_WORLD.Get_rank() == 0:
                with open(os.path.join(self.goal_save_root, 'epoch_'+str(epoch)+'.csv'), mode='a',newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(content.tolist())
                wandb.log({'exploration goal success rate':exp_success_rate})
            # print('success writen.......')
            self.step += 1
            self.candidate_goal.clear()
            self.candidate_goal.extend(selected_goals)
        else:
            pass



