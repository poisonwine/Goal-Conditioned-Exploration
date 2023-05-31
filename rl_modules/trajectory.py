import numpy as np
import copy
from envs.utils import quaternion_to_euler_angle


def goal_concat(obs, goal):
    return np.concatenate([obs, goal], axis=0)


def goal_based_process(obs):
    return goal_concat(obs['observation'], obs['desired_goal'])


class Trajectory:
    def __init__(self, init_obs):
        self.ep = {
            'obs': [copy.deepcopy(init_obs)],
            'rews': [],
            'acts': [],
            'done': []
        }
        self.length = 0

    def store_step(self, action, obs, reward, done):
        self.ep['acts'].append(copy.deepcopy(action))
        self.ep['obs'].append(copy.deepcopy(obs))
        self.ep['rews'].append(copy.deepcopy([reward]))
        self.ep['done'].append(copy.deepcopy([np.float32(done)]))
        self.length += 1

    def energy(self, env_id, w_potential=1.0, w_linear=1.0, w_rotational=1.0):
        # from "Energy-Based Hindsight Experience Prioritization"
        if env_id[:5] == 'Fetch':
            obj = []
            for i in range(len(self.ep['obs'])):
                obj.append(self.ep['obs'][i]['achieved_goal'])
            obj = np.array([obj])

            clip_energy = 0.5
            height = obj[:, :, 2]
            height_0 = np.repeat(height[:, 0].reshape(-1, 1), height[:, 1::].shape[1], axis=1)
            height = height[:, 1::] - height_0
            g, m, delta_t = 9.81, 1, 0.04
            potential_energy = g * m * height
            diff = np.diff(obj, axis=1)
            velocity = diff / delta_t
            kinetic_energy = 0.5 * m * np.power(velocity, 2)
            kinetic_energy = np.sum(kinetic_energy, axis=2)
            energy_totoal = w_potential * potential_energy + w_linear * kinetic_energy
            energy_diff = np.diff(energy_totoal, axis=1)
            energy_transition = energy_totoal.copy()
            energy_transition[:, 1::] = energy_diff.copy()
            energy_transition = np.clip(energy_transition, 0, clip_energy)
            energy_transition_total = np.sum(energy_transition, axis=1)
            energy_final = energy_transition_total.reshape(-1, 1)
            return np.sum(energy_final)
        else:
            assert env_id[:4] == 'Hand'
            obj = []
            for i in range(len(self.ep['obs'])):
                obj.append(self.ep['obs'][i]['observation'][-7:])
            obj = np.array([obj])

            clip_energy = 2.5
            g, m, delta_t, inertia = 9.81, 1, 0.04, 1
            quaternion = obj[:, :, 3:].copy()
            angle = np.apply_along_axis(quaternion_to_euler_angle, 2, quaternion)
            diff_angle = np.diff(angle, axis=1)
            angular_velocity = diff_angle / delta_t
            rotational_energy = 0.5 * inertia * np.power(angular_velocity, 2)
            rotational_energy = np.sum(rotational_energy, axis=2)
            obj = obj[:, :, :3]
            height = obj[:, :, 2]
            height_0 = np.repeat(height[:, 0].reshape(-1, 1), height[:, 1::].shape[1], axis=1)
            height = height[:, 1::] - height_0
            potential_energy = g * m * height
            diff = np.diff(obj, axis=1)
            velocity = diff / delta_t
            kinetic_energy = 0.5 * m * np.power(velocity, 2)
            kinetic_energy = np.sum(kinetic_energy, axis=2)
            energy_totoal = w_potential * potential_energy + w_linear * kinetic_energy + w_rotational * rotational_energy
            energy_diff = np.diff(energy_totoal, axis=1)
            energy_transition = energy_totoal.copy()
            energy_transition[:, 1::] = energy_diff.copy()
            energy_transition = np.clip(energy_transition, 0, clip_energy)
            energy_transition_total = np.sum(energy_transition, axis=1)
            energy_final = energy_transition_total.reshape(-1, 1)
            return np.sum(energy_final)
