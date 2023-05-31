from sklearn.neighbors import NearestNeighbors
import her_modules.config_cher as config_cur
import numpy as np
import random
import math


def curriculum(transitions, batch_size_in_transitions):
    sel_list = lazier_and_goals_sample_kg(transitions['g'], transitions['ag'], transitions['obs'], batch_size_in_transitions)
    transitions = {
        key: transitions[key][sel_list].copy()
        for key in transitions.keys()
    }
    config_cur.learning_step += 1
    return transitions

def fa(k, a_set, v_set, sim, row, col):
    if len(a_set) == 0:
        init_a_set = []
        marginal_v = 0
        for i in v_set:
            max_ki = 0
            if k == col[i]:
                max_ki = sim[i]
            init_a_set.append(max_ki)
            marginal_v += max_ki
        return marginal_v, init_a_set

    new_a_set = []
    marginal_v = 0
    for i in v_set:
        sim_ik = 0
        if k == col[i]:
            sim_ik = sim[i]

        if sim_ik > a_set[i]:
            max_ki = sim_ik
            new_a_set.append(max_ki)
            marginal_v += max_ki - a_set[i]
        else:
            new_a_set.append(a_set[i])
    return marginal_v, new_a_set


def lazier_and_goals_sample_kg(goals, ac_goals, obs, batch_size_in_transitions):
    if config_cur.goal_type == "ROTATION":
        goals, ac_goals = goals[..., 3:], ac_goals[..., 3:]

    num_neighbor = 1
    kgraph = NearestNeighbors(
        n_neighbors=num_neighbor, algorithm='kd_tree',
        metric='euclidean').fit(goals).kneighbors_graph(
            mode='distance').tocoo(copy=False)
    row = kgraph.row
    col = kgraph.col
    sim = np.exp(
        -np.divide(np.power(kgraph.data, 2),
                   np.mean(kgraph.data)**2))
    delta = np.mean(kgraph.data)

    sel_idx_set = []
    idx_set = [i for i in range(len(goals))]
    balance = config_cur.fixed_lambda
    if int(balance) == -1:
        balance = math.pow(
            1 + config_cur.learning_rate,
            config_cur.learning_step) * config_cur.lambda_starter
    v_set = [i for i in range(len(goals))]
    max_set = []
    for i in range(0, batch_size_in_transitions):
        sub_size = 3
        sub_set = random.sample(idx_set, sub_size)
        sel_idx = -1
        max_marginal = float("-inf")  #-1 may have an issue
        for j in range(sub_size):
            k_idx = sub_set[j]
            marginal_v, new_a_set = fa(k_idx, max_set, v_set, sim, row, col)
            euc = np.linalg.norm(goals[sub_set[j]] - ac_goals[sub_set[j]])
            marginal_v = marginal_v - balance * euc
            if marginal_v > max_marginal:
                sel_idx = k_idx
                max_marginal = marginal_v
                max_set = new_a_set

        idx_set.remove(sel_idx)
        sel_idx_set.append(sel_idx)
    return np.array(sel_idx_set)