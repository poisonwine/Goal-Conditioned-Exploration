from mpi_utils import normalizer
import numpy as np
from her_modules.hgg import TrajectoryPool, MatchSampler
# from goal_sampler import GoalSampler
import torch

class Rollouts(object):
    def __init__(self, env, env_params, o_norm, g_norm, policy, args, goal_sampler):
        self.env = env
        self.args = args
        self.env_steps = 0
        self.env_params = env_params
        self.policy = policy
        self.o_norm = o_norm
        self.g_norm = g_norm
        self.goal_sampler = goal_sampler
        self.V_critic = policy.critic_target_network
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # config for HGG learner
        # self.sampler = MatchSampler(self.args, self.achieved_trajectory_pool, self.V_critic, self.o_norm, self.g_norm)
        self.achieved_init_states = []
        self.achieved_trajectories = []


    def generate_rollouts(self):
        mb_obs, mb_ag, mb_g, mb_actions, mb_dones, mb_success_history = [], [], [], [], [], []
        for i in range(self.args.num_rollouts_per_mpi):
            # reset the rollouts
            ep_obs, ep_ag, ep_g, ep_actions, ep_dones = [], [], [], [], []
            # reset the environment
            if self.args.goal_teacher and self.env_steps >= self.args.warm_up_steps:
                observation = self.env.reset()
                init_state = observation['observation'].copy()
                init_goal = observation['achieved_goal']
                desired_goal = observation['desired_goal']
               
                if np.random.rand() > 1-self.args.explore_alpha:
                    explore_goal = self.goal_sampler.sample(init_state,init_goal, desired_goal)
                    self.env.goal = explore_goal.copy()
                    observation = self.env.get_obs()
                else:
                    observation = self.env.reset()
            
                self.achieved_init_states.append(observation['observation'])

                # elif self.args.teacher_method =='AGE' or self.args.teacher_method == 'VDS' or self.args.teacher_method=='AIM':
                #     # print('use goal teacher', self.args.goal_teacher) 
                #     observation = self.env.get_obs()
                #     init_goal = observation['achieved_goal']
                #     desired_goal = observation['desired_goal']
                #     if np.random.rand() > 0.5:
                #         ## to do, sample exploration goals
                #         exp_goal = self.goal_sampler.sample(init_goal, desired_goal)
                #         self.env.goal = exp_goal.copy()
                #         observation = self.env.get_obs()
                #     else:
                #         observation = self.env.reset()
                # elif self.args.teacher_method =='random':
                #     observation = self.env.reset()
            else:
                observation = self.env.reset()
                self.achieved_init_states.append(observation['observation'].copy())

            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']
            trajectory = [ag.copy()]
            # start to collect samples
            for t in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    if self.env_steps >= self.args.warm_up_steps:
                        pi = self.policy.actor_network(input_tensor)
                        action = self.policy._select_actions(pi)
                    else:
                        action = np.random.uniform(-1, 1, size=self.env_params['action'])
                # feed the actions into the environment
                observation_new, _, done, info = self.env.step(action)
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']
                trajectory.append(ag_new.copy())
                # append rollouts
                ep_obs.append(obs.copy())
                ep_ag.append(ag.copy())
                ep_g.append(g.copy())
                ep_actions.append(action.copy())
                # ep_dones.append([int(done)])

                obs = obs_new
                ag = ag_new
                # store achieved goal trajectory
            self.achieved_trajectories.append(np.array(trajectory))
            mb_success_history.append(int(info['is_success']))
            ep_obs.append(obs.copy())
            ep_ag.append(ag.copy())
            mb_obs.append(ep_obs)
            mb_ag.append(ep_ag)
            mb_g.append(ep_g)
            mb_actions.append(ep_actions)
            # mb_dones.append(ep_dones)
        # convert them into arrays
        mb_obs = np.array(mb_obs)
        mb_ag = np.array(mb_ag)
        mb_g = np.array(mb_g)
        mb_actions = np.array(mb_actions)
        # mb_dones = np.array(mb_dones)
        self.env_steps += self.args.num_rollouts_per_mpi * self.env_params['max_timesteps']
        return mb_obs, mb_ag, mb_g, mb_actions, mb_success_history


    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.to(self.device)
        return inputs




class ModelBasedRollouts(object):
    def __init__(self, env_name, env, o_norm, g_norm, policy, dynamic_model, model_buffer, obs_to_goal_func, reward_func) -> None:
        self.env = env
        self.env_name = env_name
        self.o_norm = o_norm
        self.g_norm = g_norm
        if hasattr(policy, 'actor_target_network'):
            self.actor = policy.actor_target_network
        else:
            self.actor = policy.actor_network
        
        self.dynamic_model = dynamic_model
        self.buffer = model_buffer
        self.obs_to_goal_func = obs_to_goal_func
        self.reward_func = reward_func


    def generate_modelbased_transitions(self, obs, g, steps, action_noise=0.1):
        batch_size = obs.shape[0]
        last_state = obs.copy()
        states_list,actions_list, next_states_list = [], [], []
        goals_list, ags_list, next_ags_list, reward_list = [], [], [], []
        for _ in range(0, steps):
            goals_list.append(g.copy())
            states_list.append(last_state.copy())
            ag_array = self.obs_to_goal_func(last_state, env_name=self.env_name).copy()
            ags_list.append(ag_array)

            norm_states =  self.o_norm.normalize(last_state)
            norm_g = self.g_norm.normalize(g)
            inputs_next_norm = np.concatenate([norm_states, norm_g], axis=1)
            inputs_next_norm = torch.tensor(inputs_next_norm, dtype=torch.float32)
            action_array = self.actor(inputs_next_norm).numpy()

            action_array += action_noise * np.random.randn(*action_array.shape)  # gaussian noise
            action_array = np.clip(action_array, -1, 1)
            next_state_array = self.dynamic_model.predict_next_state(last_state, action_array)

            actions_list.append(action_array.copy())
            next_states_list.append(next_state_array.copy())
            next_ag_array = self.obs_to_goal_func(next_state_array, env_name=self.env_name).copy()
            next_ags_list.append(next_ag_array)
            reward_list.append(np.expand_dims(self.reward_func(next_ag_array, g, None), 1))
            last_state = next_state_array
            
        transitions = {}
        transitions['obs'] = np.concatenate(states_list,axis=0).reshape(batch_size * steps, -1) 
        transitions['obs_next'] = np.concatenate(next_states_list,axis=0).reshape(batch_size * steps, -1) 
        transitions['ag'] = np.concatenate(ags_list,axis=0).reshape(batch_size * steps, -1) 
        transitions['ag_next'] = np.concatenate(next_ags_list,axis=0).reshape(batch_size * steps, -1) 
        transitions['g'] = np.concatenate(goals_list,axis=0).reshape(batch_size * steps, -1) 
        transitions['r'] = np.concatenate(reward_list,axis=0).reshape(batch_size * steps, -1)
        transitions['actions'] = np.concatenate(actions_list,axis=0).reshape(batch_size * steps, -1)
        return transitions



    def dynamic_interaction(self, obs, g, steps, act_noise=0.1):
        last_state = obs.copy() 
        next_states_list = []
        for _ in range(0, steps):
            norm_states =  self.o_norm.normalize(last_state)
            norm_g = self.g_norm.normalize(g)
            inputs_next_norm = np.concatenate([norm_states, norm_g], axis=1)
            inputs_next_norm = torch.tensor(inputs_next_norm, dtype=torch.float32)
            action_array = self.actor(inputs_next_norm)
            if act_noise > 0: # action noise
                action_array += np.random.normal(scale=act_noise, size=action_array.shape)
                action_array = np.clip(action_array, -1, 1)

            next_state_array = self.dynamic_model.predict_next_state(last_state, action_array)
            next_states_list.append(next_state_array.copy()) 
            last_state = next_state_array
        return next_states_list

    def update_forward_model(self, obs, actions, next_obs, update_times, add_noise):
        loss = self.dynamic_model.update(obs, actions, next_obs, update_times, add_noise=add_noise)
        return loss







