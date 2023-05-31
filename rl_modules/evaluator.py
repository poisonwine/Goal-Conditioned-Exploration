import torch
import numpy as np
from mpi4py import MPI

class eval_agent(object):
    def __init__(self, env, policy, args, env_params, o_norm, g_norm):
        self.env = env
        self.policy = policy
        self.args = args
        self.env_params = env_params
        self.o_norm = o_norm
        self.g_norm = g_norm
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def _eval_desired_goal(self):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    if self.args.agent.lower() == 'sac':
                        pi, _ = self.policy.actor_network(input_tensor)
                    else:
                        pi = self.policy.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        # print(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        # print(MPI.COMM_WORLD.Get_size())
        return global_success_rate / MPI.COMM_WORLD.Get_size()


    def _eval_exploration_goal(self, goal_list):
        exp_goals = np.array(goal_list)
        total_success_rate = []
        for i in range(exp_goals.shape[0]):
            per_success_rate = []
            observation = self.env.reset()
            self.env.goal = exp_goals[i].copy()
            observation = self.env.get_obs()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    if self.args.agent.lower() == 'sac':
                        pi, _ = self.policy.actor_network(input_tensor)
                    else:
                        pi = self.policy.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        # print(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        # print(MPI.COMM_WORLD.Get_size())
        return global_success_rate / MPI.COMM_WORLD.Get_size()
        


    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.to(self.device)
        return inputs



