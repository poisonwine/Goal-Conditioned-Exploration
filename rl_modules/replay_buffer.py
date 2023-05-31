import threading
import numpy as np
from sklearn import mixture
import abc
from sklearn.neighbors import KernelDensity
import os
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from utils.goal_utils import split_robot_state_from_observation
import math
"""
the replay buffer here is basically from the openai baselines code

"""
def quaternion_to_euler_angle(array):
    w = array[0]
    x = array[1]
    y = array[2]
    z = array[3]
    ysqr = y * y
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.atan2(t3, t4)
    result = np.array([X, Y, Z])
    return result



class ReplayBuffer(object):
    def __init__(self, env_params, sampler, args):
        self.env_params = env_params
        self.args = args
        self.T = env_params['max_timesteps']
        self.size = args.buffer_size // self.T
        self.priority_delete = args.buffer_priority
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sampler = sampler
        # create the buffer to store info
        self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                        'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                        'g': np.empty([self.size, self.T, self.env_params['goal']]),
                        'actions': np.empty([self.size, self.T, self.env_params['action']]),
                        'dones': np.empty([self.size, self.T, 1]),
                        'sim_p': np.zeros([self.size, 1]),
                        'traj_p': np.zeros([self.size, 1]),
                        }

        # thread lock
        self.lock = threading.Lock()


    def update_episode_priority(self, sigma, w_diverse, w_sim):
        trajectory_final_goal = self.buffers['ag'][0:self.current_size].copy()
        trajectory_final_goal = trajectory_final_goal[:, self.T, :]
        origin_dgs = self.buffers['g'][0:self.current_size].copy()
        origin_dgs = origin_dgs[:, self.T-1, :]
        dis = np.linalg.norm(trajectory_final_goal-origin_dgs, ord=2, axis=1)
        sim_score = np.exp(-dis.reshape((-1, 1)) / sigma)
        diverse_score = self.buffers['entropy'][:self.current_size].copy()
        with self.lock:
            self.buffers['sim_p'][:self.current_size] = sim_score.reshape(-1, 1).copy()
        pri_score = w_diverse * diverse_score + w_sim * sim_score
        with self.lock:
            self.buffers['traj_p'][:self.current_size] = pri_score.reshape(-1, 1).copy()
        mean_diverse_score = np.sum(diverse_score) / self.current_size
        mean_sim_score = np.sum(sim_score) / self.current_size
        return mean_diverse_score, mean_sim_score

    
    # store the episode
    def store_episode(self, episode_batch, extra_buffer=None):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            self.buffers['obs'][idxs] = mb_obs
            self.buffers['ag'][idxs] = mb_ag
            self.buffers['g'][idxs] = mb_g
            self.buffers['actions'][idxs] = mb_actions
            if extra_buffer is not None:
                for key in extra_buffer.keys():
                    self.buffers[key][idxs] = extra_buffer[key]
            self.n_transitions_stored += self.T * batch_size
        return idxs

    
    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions

        transitions = self.sampler.sample(temp_buffers, batch_size)

        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            if self.priority_delete:
                pri = self.buffers['sim_p'][0:self.current_size].copy().flatten()
                rank = sorted(range(len(pri)), key=pri.__getitem__)
                idx_b = np.array(rank)[:overflow]
            else:
                idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0
            self.n_transitions_stored = 0
            self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                            'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                            'g': np.empty([self.size, self.T, self.env_params['goal']]),
                            'actions': np.empty([self.size, self.T, self.env_params['action']]),
                            }


class TrajectoryPriorityBuffer(ReplayBuffer):
    def __init__(self, env_params, sampler, args, priority):
        super(TrajectoryPriorityBuffer, self).__init__(env_params=env_params, sampler=sampler, args=args)

        # for MEP method
        self.pred_min = 0
        self.pred_sum = 0
        self.pred_avg = 0
        self.clf = 0
        self.priority = priority
        self.w_potential = args.w_potential
        self.w_linear = args.w_linear
        self.w_rotational = args.w_rotational
        self.clip_energy = args.clip_energy
        self.buffers['entropy'] = np.zeros([self.size, 1])
        self.buffers['energy'] = np.zeros([self.size, 1])

    def fit_density_model(self):
        ag = self.buffers['ag'][0: self.current_size].copy()
        dg = self.buffers['g'][0: self.current_size].copy()
        dg = dg.reshape(-1, dg.shape[1] * dg.shape[2])
        X_train = ag.reshape(-1, ag.shape[1] * ag.shape[2])
        self.clf = mixture.BayesianGaussianMixture(weight_concentration_prior_type="dirichlet_distribution", n_components=3, reg_covar=1e-3)
        self.clf.fit(X_train)
        pred = -self.clf.score_samples(X_train)
        self.pred_min = pred.min()
        pred = pred - self.pred_min
        pred = np.clip(pred, 0, None)
        self.pred_sum = pred.sum()
        pred = pred / self.pred_sum
        self.pred_avg = (1 / pred.shape[0])

        with self.lock:
            self.buffers['entropy'][:self.current_size] = pred.reshape(-1, 1).copy()



    def calculate_energy(self, buffers):

        energy_priority = 0
        if self.args.env_name.lower().startswith('fetch') and self.args.env_name[:10] != 'FetchReach':
            height = buffers['ag'][:, :, 2]
            height_0 = np.repeat(height[:, 0].reshape(-1, 1), height[:, 1::].shape[1], axis=1)
            height = height[:, 1::] - height_0
            g, m, delta_t = 9.81, 1, 0.04
            potential_energy = g * m * height
            diff = np.diff(buffers['ag'], axis=1)
            velocity = diff / delta_t
            kinetic_energy = 0.5 * m * np.power(velocity, 2)
            kinetic_energy = np.sum(kinetic_energy, axis=2)
            energy_totoal = self.w_potential * potential_energy + self.w_linear * kinetic_energy
            energy_diff = np.diff(energy_totoal, axis=1)
            energy_transition = energy_totoal.copy()
            energy_transition[:, 1::] = energy_diff.copy()
            energy_transition = np.clip(energy_transition, 0, self.clip_energy)
            energy_transition_total = np.sum(energy_transition, axis=1)
            energy_priority = energy_transition_total.reshape(-1, 1)

        elif self.args.env_name.lower().startswith('handmanipulate'):
            g, m, delta_t, inertia = 9.81, 1, 0.04, 1
            quaternion = buffers['ag'][:, :, 3:].copy()
            angle = np.apply_along_axis(quaternion_to_euler_angle, 2, quaternion)
            diff_angle = np.diff(angle, axis=1)
            angular_velocity = diff_angle / delta_t
            rotational_energy = 0.5 * inertia * np.power(angular_velocity, 2)
            rotational_energy = np.sum(rotational_energy, axis=2)
            buffers['ag'] = buffers['ag'][:, :, :3]
            height = buffers['ag'][:, :, 2]
            height_0 = np.repeat(height[:, 0].reshape(-1, 1), height[:, 1::].shape[1], axis=1)
            height = height[:, 1::] - height_0
            potential_energy = g * m * height
            diff = np.diff(buffers['ag'], axis=1)
            velocity = diff / delta_t
            kinetic_energy = 0.5 * m * np.power(velocity, 2)
            kinetic_energy = np.sum(kinetic_energy, axis=2)
            energy_totoal = self.w_potential * potential_energy + self.w_linear * kinetic_energy + self.w_rotational * rotational_energy
            energy_diff = np.diff(energy_totoal, axis=1)
            energy_transition = energy_totoal.copy()
            energy_transition[:, 1::] = energy_diff.copy()
            energy_transition = np.clip(energy_transition, 0, self.clip_energy)
            energy_transition_total = np.sum(energy_transition, axis=1)
            energy_priority = energy_transition_total.reshape(-1, 1)
        else:
            print('Trajectory Energy Function Not Implemented')
            exit()
        return energy_priority

    def store_episode(self, episode_batch, extra_buffer=None):
        idxs = super().store_episode(episode_batch=episode_batch, extra_buffer=extra_buffer)
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        buffer = {'ag': mb_ag}
        if self.priority == 'energy':
            energy_pri = self.calculate_energy(buffer)
            with self.lock:
                self.buffers['energy'][idxs] = energy_pri
        if self.priority == 'entropy':
            if not isinstance(self.clf, int):
                ag = mb_ag.copy()
                X = ag.reshape(-1, ag.shape[1] * ag.shape[2])
                pred = -self.clf.score_samples(X)
                pred = pred - self.pred_min
                pred = np.clip(pred, 0, None)
                pred = pred / self.pred_sum
                with self.lock:
                    self.buffers['entropy'][idxs] = pred.reshape(-1, 1)

        return idxs




class TransitionPriorityBuffer(TrajectoryPriorityBuffer):
    def __init__(self, env_params, sampler, args, priority):
        super(TransitionPriorityBuffer, self).__init__(env_params, sampler, args, priority)

        self.buffers['td_error'] = np.zeros([self.size, self.T])

    def store_episode(self, episode_batch, extra_buffer=None):
        episode_idxs = super().store_episode(episode_batch=episode_batch, extra_buffer=extra_buffer)
        self.sampler.update_new_priorities(episode_idxs)

    def update_priorities(self, idxs, priorities):
        self.sampler.update_priorities(idxs, priorities)





class goal_buffer:
    def __init__(self, name, buffer_size, sample_dim):
        self.name = name
        self.size = buffer_size
        self.sample_dim = sample_dim
        self.current_size = 0
        self.buffer = None
        self.init_buffer()
        self.lock = threading.Lock()

    def init_buffer(self):
        self.buffer = {'candidate_goals': np.zeros([self.size, self.sample_dim])}

    def extend(self, data):
        goals = np.asarray(data)
        batch_size = goals.shape[0]
        if batch_size==0:
            pass
        else:
            with self.lock:
                idxs = self.get_storage_inds(inc=batch_size)
                self.buffer['candidate_goals'][idxs] = goals

    def get_storage_inds(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx

    def clear_buffer(self):
        self.current_size = 0
        self.buffer = {'candidate_goals': np.empty([self.size, self.sample_dim])}

    def random_sample(self, batch_size):
        # print('current size', self.current_size)
        idx = np.random.randint(self.current_size, size=batch_size)
        with self.lock:
            goals = self.buffer['candidate_goals'][:self.current_size].copy()
        select_goals = goals[idx].copy()
        return select_goals

    def prority_sample(self):
        assert NotImplementedError



class density_priority_buffer(goal_buffer):
    def __init__(self, name, buffer_size, sample_dim):
        super(density_priority_buffer, self).__init__(name=name, buffer_size=buffer_size, sample_dim=sample_dim)
        self.min_entropy = 0
        self.sum_entropy = 0

    def init_buffer(self):
        self.buffer = {'candidate_goals': np.zeros([self.size, self.sample_dim]),
                      'log_density': np.zeros([self.size, 1]),
                      'entropy': np.zeros([self.size, 1])}

    def extend(self, data):
        goal, log_density, entropy = data
        goal = np.asarray(goal)
        entr = np.asarray(entropy).copy().reshape(-1, 1)
        log_density = log_density.copy().reshape(-1, 1)
        batch_size = goal.shape[0]
        # buffer_size = self.buffer['candidate_goals'].shape[0]
        with self.lock:
            idxs = self.get_storage_inds(inc=batch_size)
            self.buffer['candidate_goals'][idxs] = goal
            self.buffer['entropy'][idxs] = entr
            self.buffer['log_density'][idxs] = log_density

    def sample(self, batch_size):
        with self.lock:
            goals = self.buffer['candidate_goals'][:self.current_size].copy()
            entropy = self.buffer['entropy'][:self.current_size].copy()
        self.min_entropy = np.clip(entropy.min(), -1000, 1000)
        entropy = np.clip(entropy - self.min_entropy, 0, 100)
        self.sum_entropy = entropy.sum()
        if entropy.sum() < 1e-5:
            entropy = np.ones(entropy.shape) / len(entropy)
        else:
            entropy = np.power(entropy, 1)
            entropy = entropy / entropy.sum()
        inds = np.random.choice(goals.shape[0], size=batch_size, replace=True, p=entropy.flatten())
        selected_goals = goals[inds].copy()
        return selected_goals


    def clear_buffer(self):
        self.current_size=0
        self.buffer = {'candidate_goals': np.empty([self.size, self.sample_dim]),
                       'log_density': np.empty([self.size, 1]),
                       'entropy': np.empty([self.size, 1])}



class goal_replay_buffer(object):
    def __init__(self, name, buffer_size, sample_dim):
        self.name = name
        self.size = buffer_size
        self.sample_dim = sample_dim
        self.current_size = 0
        self.buffer = {'candidate_goals': np.zeros([self.size, self.sample_dim]),
                       'log_density': np.zeros([self.size, 1]),
                       'entropy': np.zeros([self.size, 1])}

        self.lock = threading.Lock()
        self.min_entropy = 0
        self.sum_entropy = 0



    def extend(self, data):
        goal, log_density, entropy = data
        data = np.asarray(goal)
        e = np.asarray(entropy).copy().reshape(-1, 1)
        log_density = log_density.copy().reshape(-1, 1)
        batch_size = data.shape[0]
        #buffer_size = self.buffer['candidate_goals'].shape[0]
        with self.lock:
            idxs = self.get_storage_inds(inc=batch_size)
            self.buffer['candidate_goals'][idxs] = data
            self.buffer['entropy'][idxs] = e
            self.buffer['log_density'][idxs] = log_density

    def sample(self, batch_size):
        with self.lock:
            goals = self.buffer['candidate_goals'][:self.current_size].copy()
            entropy = self.buffer['entropy'][:self.current_size].copy()
        self.min_entropy = np.clip(entropy.min(), -1000, 1000)
        entropy = np.clip(entropy - self.min_entropy, 0, 100)
        self.sum_entropy = entropy.sum()
        #print(self.sum_entropy,self.min_entropy)
            #entropy = entropy / self.sum_entropy
        if entropy.sum() < 1e-5:
            entropy = np.ones(entropy.shape) / len(entropy)
        else:
            entropy = np.power(entropy, 1)
            entropy = entropy / entropy.sum()
        inds = np.random.choice(goals.shape[0], size=batch_size, replace=True, p=entropy.flatten())
        selected_goals = goals[inds].copy()
        # self.step += 1
        # save_path = './selected_goals'
        # os.makedirs(save_path, exist_ok=True)
        # plt.figure(figsize=(8, 6))
        # plt.scatter(selected_goals[:, 0], selected_goals[:, 1],color='r', label='selected goal')
        # plt.legend()
        # plt.xlim(0, 1.8)
        # plt.ylim(0, 1.5)
        # plt.savefig(os.path.join(save_path, '2D_visual_' + str(self.step) + '.jpg'))
        return selected_goals


    def random_sample(self, batch_size):
        print('current size', self.current_size)
        idx = np.random.randint(self.current_size, size=batch_size)
        with self.lock:
            goals = self.buffer['candidate_goals'][:self.current_size].copy()
        select_goals = goals[idx].copy()
        # plt.figure()
        # plt.scatter(goals[:, 0], goals[:, 1])
        # plt.show()
        return select_goals

    def clear_buffer(self):
        self.current_size=0

        self.buffer = {'candidate_goals': np.empty([self.size, self.sample_dim]),
                       'log_density': np.empty([self.size, 1]),
                       'entropy': np.empty([self.size, 1])}

    def get_storage_inds(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx

if __name__ == '__main__':

    ag_buffer = goal_buffer(name='ag', buffer_size=1000, sample_dim=3)
    ag_pri_buffer = density_priority_buffer(name='ag_pri', buffer_size=200, sample_dim=3)
    goals = np.random.rand(30, 3)
    density = np.random.rand(30, 1)
    entr = np.random.rand(30, 1)
    data = [goals, density, entr]
    ag_pri_buffer.extend(data)
    print(ag_pri_buffer.min_entropy)
    selected_goals = ag_pri_buffer.sample(5)
    print(selected_goals)
    ag_pri_buffer.clear_buffer()
    print(ag_pri_buffer.current_size)