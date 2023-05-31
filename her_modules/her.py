import numpy as np
import her_modules.config_cher as config_cur
from her_modules.cher import curriculum
from her_modules.goal_curriculum import GoalAugmentor
from utils.segment_tree import MinSegmentTree,SumSegmentTree
import torch

def obs_to_goal(env_name):
    pass



class RandomSampler(object):
    def __init__(self, args, reward_func):
        self.args = args
        self.reward_func = reward_func
        
    def eposide_prioritize(self, episode_batch, batch_size):
        rollout_batch_size = episode_batch['actions'].shape[0]
        entropy_trajectory = episode_batch[self.args.traj_rank_method]
        if np.sum(entropy_trajectory) == 0:
            p_trajectory = np.ones(entropy_trajectory.shape) / len(entropy_trajectory)
        else:
            p_trajectory = np.power(entropy_trajectory, 1 / (self.args.temperature + 1e-2))
            p_trajectory = p_trajectory / p_trajectory.sum()
        episode_idxs_entropy = np.random.choice(rollout_batch_size, size=batch_size, replace=True, p=p_trajectory.flatten())
        episode_idxs = episode_idxs_entropy
        return episode_idxs

    def sample_idxs(self, episode_batch, batch_size, update_normalizer=False):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        # select which rollouts and which timesteps to be used
        if not self.args.episode_priority:
            episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        elif self.args.episode_priority:
            if update_normalizer:
                episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
            else:
                episode_idxs = self.eposide_prioritize(episode_batch, batch_size)

        t_samples = np.random.randint(T, size=batch_size)

        return episode_idxs, t_samples



    def get_transitions(self, episode_batch, episode_idxs, t_samples):
        transitions = {}

        for key in episode_batch.keys():
            if key in ['obs', 'ag', 'g', 'actions', 'obs_next', 'ag_next']:
                transitions[key] = episode_batch[key][episode_idxs, t_samples].copy()
        return transitions

    def sample_transitions(self, episode_batch, batch_size_in_transitions, update_normalizer=False):
        episode_idxs, t_samples = self.sample_idxs(episode_batch, batch_size_in_transitions, update_normalizer)
        transitions = self.get_transitions(episode_batch, episode_idxs, t_samples)
        info = {'episode_idxs':episode_idxs,
                't_samples': t_samples}
        return transitions, info

    def recompute_reward(self, transitions):
        reshape_r = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        return reshape_r

    def reshape_transitions(self, transitions, batch_size):
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert (transitions['actions'].shape[0] == batch_size)
        return transitions


    def sample_for_normlization(self, episode_batch, batch_size):
        transitions, _ = self.sample_transitions(episode_batch=episode_batch, batch_size_in_transitions=batch_size, update_normalizer=True)
        transitions['r'] = self.recompute_reward(transitions)
        transitions = self.reshape_transitions(transitions, batch_size)
        return transitions

    def GHERWrapper(self, transitions, batch_size):
        all_transitions = {}
        for _ in range(self.args.n_GER):
            PER_transitions = {key: transitions[key].copy() for key in transitions.keys()}
            ger_machine = GoalAugmentor(env_name=self.args.env_name, error_distance=self.args.error_dis, batch_size=batch_size)
            PER_indexes = np.array((range(0, batch_size)))
            HER_KER_future_ag = PER_transitions['g'][PER_indexes].copy()
            PER_future_g = ger_machine.goal_arguement(HER_KER_future_ag.copy())
            PER_transitions['g'][PER_indexes] = PER_future_g.copy()
            for key in transitions.keys():
                all_transitions[key] = np.vstack([transitions[key], PER_transitions[key].copy()])
        return all_transitions


    def sample(self, episode_batch, batch_size):
        if self.args.use_laber:
            transitions, _ = self.sample_transitions(episode_batch=episode_batch, batch_size_in_transitions= int(batch_size*self.args.m_factor))
        else:
            transitions, _ = self.sample_transitions(episode_batch=episode_batch, batch_size_in_transitions=batch_size)
        batch_size = transitions['obs'].shape[0]
        if self.args.use_ger and self.args.n_GER > 0:
            transitions = self.GHERWrapper(transitions, batch_size)
            batch_size = int((1 + self.args.n_GER) * batch_size)

        transitions['r'] = self.recompute_reward(transitions)
        transitions = self.reshape_transitions(transitions, batch_size)
        return transitions


class HERSampler(RandomSampler):
    def __init__(self, replay_strategy, replay_k, args, reward_func, *kwargs):
        super(HERSampler, self).__init__(args=args, reward_func=reward_func)
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        self.args = args
        if self.replay_strategy in ['future', 'final', 'episode', 'cut', 'random']:
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0

    def get_relabel_ag(self, episode_batch, episode_idxs, t_samples, batch_size):
        T = episode_batch['actions'].shape[1]
        relabel_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        if self.replay_strategy == 'future':
            future_offset = (np.random.uniform(size=batch_size) * (T - t_samples)).astype(int)
            future_t = (t_samples + 1 + future_offset)[relabel_indexes]
            future_ag = episode_batch['ag'][episode_idxs[relabel_indexes], future_t]
        elif self.replay_strategy == 'final':
            future_ag = episode_batch['ag'][episode_idxs[relabel_indexes], -1]
        # replace go with achieved goal
        elif self.replay_strategy == 'episode':
            random_t_samples = np.random.randint(T, size=batch_size)[relabel_indexes]
            future_ag = episode_batch['ag'][episode_idxs[relabel_indexes], random_t_samples]
        elif self.replay_strategy == 'random':
            rollout_batch_size = episode_batch['actions'].shape[0]
            random_episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)[relabel_indexes]
            random_t_samples = np.random.randint(T, size=batch_size)[relabel_indexes]
            future_ag = episode_batch['ag'][random_episode_idxs, random_t_samples]

        return future_ag, relabel_indexes

    def relabel_transitions(self, transitions, relable_indexs, relabel_ag):

        transitions['g'][relable_indexs] = relabel_ag
        transitions['r'] = self.recompute_reward(transitions)
        return transitions

    def sample(self, episode_batch, batch_size):
        if self.args.use_laber:
            transitions, info = self.sample_transitions(episode_batch=episode_batch, batch_size_in_transitions=int(batch_size*self.args.m_factor))
        else:
            transitions, info = self.sample_transitions(episode_batch=episode_batch, batch_size_in_transitions= batch_size)
        batch_size = transitions['obs'].shape[0]
        relabel_ags, relabel_indexs = self.get_relabel_ag(episode_batch, info['episode_idxs'], info['t_samples'], batch_size)
        transitions = self.relabel_transitions(transitions, relabel_indexs, relabel_ags)
        if self.args.use_cher and self.args.her: # only when her=True, can use CHER
            transitions = self.CHERWrapper(transitions)
            batch_size = transitions['obs'].shape[0]
        if self.args.use_ger and self.args.n_GER > 0:
            transitions = self.GHERWrapper(transitions, batch_size)
            batch_size = int((1 + self.args.n_GER) * batch_size)
        transitions['r'] = self.recompute_reward(transitions)
        transitions = self.reshape_transitions(transitions, batch_size)
        return transitions

    def sample_for_normlization(self, episode_batch, batch_size):
        transitions, info = self.sample_transitions(episode_batch=episode_batch, batch_size_in_transitions=batch_size, update_normalizer=True)
        relabel_ags, relabel_indexs = self.get_relabel_ag(episode_batch, info['episode_idxs'], info['t_samples'], batch_size)
        transitions = self.relabel_transitions(transitions, relabel_indexs, relabel_ags)
        transitions = self.reshape_transitions(transitions, batch_size)
        return transitions

    def CHERWrapper(self, transitions):

        if self.args.use_laber:
            batch_size_intransition = int(config_cur.learning_selected * self.args.m_factor)
        else:
            batch_size_intransition = config_cur.learning_selected

        transitions = curriculum(transitions, batch_size_intransition)
        return transitions


    def sample_mbpo_transitions(self, episode_batch, batch_size, mb_worker):
        if self.args.use_laber:
            transitions, _ = self.sample_transitions(episode_batch=episode_batch, batch_size_in_transitions= int(batch_size*self.args.m_factor))
        else:
            transitions, _ = self.sample_transitions(episode_batch=episode_batch, batch_size_in_transitions=batch_size)
        mb_worker.generate_modelbased_transitions(transitions['obs'], transitions['g'], )

        
        
        



class PrioritizedSampler(RandomSampler):
    def __init__(self, T, args, reward_func, alpha, beta, *kwargs):
        super(PrioritizedSampler, self).__init__(args=args, reward_func=reward_func, *kwargs)
        assert alpha >= 0 and beta >= 0
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-5
        self.T = T  # env_params['max_timesteps']
        capacity = 1
        size_in_transitions = args.buffer_size
        while capacity < size_in_transitions:
            capacity *= 2
        self.sum_tree = SumSegmentTree(capacity)
        self.min_tree = MinSegmentTree(capacity)
        self.capacity = size_in_transitions
        self._max_priority = 1.0
        self.n_transitions_stored = 0

    def update_new_priorities(self, episode_idxs):
        N = len(episode_idxs) * self.T
        priority_array = np.zeros(N) + self._max_priority
        x = (episode_idxs * self.T).repeat(self.T)
        # transfer episode idxs to transitions idxs
        episode_idxs_repeat = (episode_idxs * self.T).repeat(self.T) + np.arange(self.T).reshape(1, -1).repeat(len(episode_idxs)).flatten()
        self.update_priorities(episode_idxs_repeat, priority_array)
        self.n_transitions_stored += len(episode_idxs) * self.T
        self.n_transitions_stored = min(self.n_transitions_stored, self.capacity)

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions"""
        assert len(idxes) == len(priorities) and np.all(priorities >= 0)
        priorities += self.eps  # avoid zero
        new_priority = np.power(priorities.flatten(), self.alpha)
        self.sum_tree.set_items(idxes, new_priority)
        self.min_tree.set_items(idxes, new_priority)
        self._max_priority = max(np.max(priorities), self._max_priority)



    def _per_sample_idxs(self, episode_batch, batch_size):
        T = episode_batch['actions'].shape[1]
        culm_sums = np.random.random(size=batch_size) * self.sum_tree.sum()
        idxes = np.zeros(batch_size)
        for i in range(batch_size):
            idxes[i] = self.sum_tree.find_prefixsum_idx(culm_sums[i])
        episode_idxs = idxes // T
        t_samples = idxes % T
        return episode_idxs.astype(np.int), t_samples.astype(np.int), idxes.astype(np.int)


    def priority_sample(self, episode_batch, batch_size):
        episode_idxs, t_samples, idxes = self._per_sample_idxs(episode_batch, batch_size)
        p_min = self.min_tree.min() / self.sum_tree.sum()
        transitions = self.get_transitions(episode_batch, episode_idxs, t_samples)
        p_samples = self.sum_tree.get_items(idxes) / self.sum_tree.sum()
        weights = np.power(p_samples / p_min, -self.beta)
        transitions['w'] = weights.reshape(-1, 1)
        info = {
            'episode_idxs': episode_idxs,
            't_samples': t_samples,
            'idxes': idxes,
            'num_episodes': episode_batch['obs'].shape[0]
        }
        return transitions, info

    def sample(self, episode_batch, batch_size):
        if self.args.use_laber:
            transitions, info = self.priority_sample(episode_batch, int(self.args.m_factor * batch_size))
        else:
            transitions, info = self.priority_sample(episode_batch, batch_size)
        batch_size = transitions['obs'].shape[0]
        if self.args.use_ger and self.args.n_GER > 0:
            transitions = self.GHERWrapper(transitions, batch_size)
            batch_size = int((1 + self.args.n_GER) * batch_size)
        transitions['r'] = self.recompute_reward(transitions)
        transitions = self.reshape_transitions(transitions, batch_size)
        return (transitions, info['idxes'])




class PrioritizedHERSampler(PrioritizedSampler, HERSampler):
    def __init__(self, T, args, reward_func, alpha, beta, replay_strategy, replay_k):
        super().__init__(T, args, reward_func, alpha, beta, replay_strategy, replay_k)


    def sample(self, episode_batch, batch_size):
        if self.args.use_laber:
            transitions, info = self.priority_sample(episode_batch, int(self.args.m_factor * batch_size))
        else:
            transitions, info = self.priority_sample(episode_batch, batch_size)
        batch_size = transitions['obs'].shape[0]
        relabel_ag, relabel_indexes = self.get_relabel_ag(episode_batch, info['episode_idxs'], info['t_samples'], batch_size)
        transitions = self.relabel_transitions(transitions, relabel_indexes, relabel_ag)
        if self.args.use_cher and self.args.her:
            transitions = self.CHERWrapper(transitions)
            batch_size = transitions['obs'].shape[0]
        if self.args.use_ger and self.args.n_GER > 0:
            transitions = self.GHERWrapper(transitions, batch_size)
            batch_size = int((1 + self.args.n_GER) * batch_size)
            info['idxes'] = info['idxes'].reshape(1, -1).repeat(1+self.args.n_GER).flatten()

        transitions['r'] = self.recompute_reward(transitions)
        transitions = self.reshape_transitions(transitions, batch_size)
        return (transitions, info['idxes'])











