import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
import numpy as np
import json
import argparse
import os
from utils.envbuilder import make_env
import pandas as pd
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib as mpl
import torch
from rl_modules.models import MRNCritic

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='./')
    parser.add_argument('--env_name', type=str, default='FetchPushWallObstacle-v1')
    parser.add_argument('--alg', type=str, default='ours2')
    parser.add_argument('--load_dir', type=str, default='../results')
    parser.add_argument('--epoch',type=int, default=29)
    args = parser.parse_args()
    return args


def read_beh_goal(epoch):
    data_path = os.path.join(args.dir, args.env_name, args.alg, 'goal_data', 'beh_goal_' + str(epoch) + '.csv')
    print(data_path)
    if os.path.exists(data_path):
        data = []
        with open(data_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                x = line.split(',')
                point = []
                for num in x:
                    point.append(float(num))
                data.append(point)
        data = np.array(data)
        return data
    else:
        print('file is not exist')
        return 0

def load_goal_result(epoch):
    filepath = os.path.join(args.dir, args.env_name, args.alg, 'goal_data', 'goaldata_' + str(epoch) + '.csv')
    if not os.path.exists(filepath):
        print('no such file')
        return None
    with open(filepath, 'r') as f:
        lines = [line for line in f]
    keys = lines[0].strip('\n').split(',')
    result = dict.fromkeys(keys)
    origin_dgs = []
    initial_ags = []
    hindsight_goals = []
    for line in lines[1:]:
        data = line.strip('\n').split(',')
        for i in range(len(data)):
            data[i] = float(data[i])
        origin_dgs.append(data[:3])
        initial_ags.append(data[3:6])
        hindsight_goals.append(data[6:])
    result['origin_dg'] = np.array(origin_dgs)
    result['initial_ag'] = np.array(initial_ags)
    result['hindsight_goal'] = np.array(hindsight_goals)
    return result

def Visual_FetchPushObstacle(bgs, ags):
    assert bgs.shape[1]==3
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(ags[:, 0], ags[:, 1],color='g')
    plt.scatter(bgs[:, 0], bgs[:, 1],color='r')
    # plt.scatter(ags[:, 0], ags[:, 1],color='g')
    rect1 = patches.Rectangle((1.05, 0.4), 0.5, 0.7, linewidth=5, edgecolor="#7f7f7f", fill=False)
    rect2 = patches.Rectangle((1.22, 0.81), 0.04, 0.28, linewidth=3, fill=True, facecolor='g')
    rect3 = patches.Rectangle((1.22, 0.41), 0.04, 0.28, linewidth=3, fill=True, facecolor='g')
    plt.xlim(1, 1.8)
    plt.ylim(0.3, 1.15)
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    plt.savefig('./figures/fetchpnpobstacle_goal.png')
    # plt.show()

def Visual_FetchPnPObstacle(data):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #plt.scatter(data[:, 0], data[:, 1], color='r')
    rect1 = patches.Rectangle((1.0, 0.4), 0.6, 0.7, linewidth=5, edgecolor="#7f7f7f", fill=False)
    rect2 = patches.Rectangle((1, 0.785), 0.6, 0.03, linewidth=3, fill=True, facecolor='g')
    #rect3 = patches.Rectangle((1.22, 0.41), 0.04, 0.28, linewidth=3, fill=True, facecolor='g')
    plt.scatter(data[:, 0], data[:, 1], color='r')
    plt.xlim(0.9, 1.8)
    plt.ylim(0.3, 1.2)
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    #ax.add_patch(rect3)
    plt.show()


def goal_distribution_visualize_FetchPushObstacle(alg, seed, epoch, env_name):
    goal_root_path = os.path.join('./saved_models', env_name, alg, 'seed-'+str(seed))
    ags_path = os.path.join(goal_root_path, 'ags','epoch_'+str(epoch)+'.csv')
    candidate_goal_path = os.path.join(goal_root_path, 'candidate_goal','epoch_'+str(epoch)+'.csv')
    exploration_goal_path = os.path.join(goal_root_path, 'exploration_goal','epoch_'+str(epoch)+'.csv')

    ags = np.genfromtxt(ags_path,delimiter=',')
    # print(ags.shape)
    exploration_goal = np.genfromtxt(exploration_goal_path, delimiter=',')[:,:2]

    goal_host = np.genfromtxt(candidate_goal_path, delimiter=',')
    candidate_goal = goal_host[:,:2]
    
    q_values = goal_host[:,3]
    density_pri = goal_host[:,4]
    value_distance = goal_host[:, 5]
    lp_distance = goal_host[:, 6]
    mega_vad = goal_host[:, 7]
    mega_lp = goal_host[:, 8]
    print(candidate_goal.shape, lp_distance.shape)
    def normalize_values(values):
        # v_min, v_max = np.array(values).min(), np.array(values).max()
        norm_values = values / (values.sum()+1e-5)
        v_min, v_max = np.array(norm_values).min(), np.array(norm_values).max()
        return v_min, v_max, norm_values

    fig, ax = plt.subplots()
    # plot obstacle
    if env_name == 'FetchPushWallObstacle-v1':
        rect1 = patches.Rectangle((1.05, 0.4), 0.5, 0.7, linewidth=5, edgecolor="#7f7f7f", fill=False)
        rect2 = patches.Rectangle((1.22, 0.81), 0.04, 0.28, linewidth=3, fill=True, facecolor='g')
        rect3 = patches.Rectangle((1.22, 0.41), 0.04, 0.28, linewidth=3, fill=True, facecolor='g')
       
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)
        plt.xlim(1, 1.6)
        plt.ylim(0.3, 1.15)

    elif env_name == 'FetchPushDoubleObstacle-v1':
        rect1 = patches.Rectangle((1.05, 0.45), 0.5, 0.6, linewidth=5, edgecolor="#7f7f7f", fill=False)
        rect2 = patches.Rectangle((1.145, 0.64), 0.22, 0.04, linewidth=3, fill=True, facecolor='g')
        rect3 = patches.Rectangle((1.235, 0.86), 0.22, 0.04, linewidth=3, fill=True, facecolor='b')
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        plt.xlim(1, 1.6)
        plt.ylim(0.4, 1.1)
    lp_min, lp_max, norm_lp = normalize_values(q_values)

    viridis = cm.get_cmap('RdBu', 100)
    newcolors = viridis(np.linspace(0, 1, 100))
    print(lp_min, lp_max)
    x = np.linspace(0, lp_max, 100)
    
    for i in range(candidate_goal.shape[0]):
        color_idx = np.argmin(np.abs(x - norm_lp[i]))
        ax.scatter(candidate_goal[i,0], candidate_goal[i,1], color=newcolors[color_idx], vmin=lp_min, vmax=lp_max)
    for i in range(exploration_goal.shape[0]):
        ax.scatter(exploration_goal[i,0], exploration_goal[i,1], color='r')
    # im = ax.pcolor(candidate_goal[:,0],candidate_goal[:,1], np.tile(norm_lp.reshape(-1,1), (1,candidate_goal.shape[0])), cmap=newcmp,vmin=lp_min,vmax=lp_max)
    ax.set_title("Learning progress in FetchPushObstacle env") 
    plt.xlabel('X') ; plt.ylabel('Y')  
    # norm = mpl.colors.Normalize(vmin=0, vmax=1)

    # fig.colorbar(mappable=fig ,ax=ax)
    save_name = alg+ '-'+env_name+'-seed-'+str(seed)+'-epoch-'+str(epoch)+'-candidate_goal_qvalues.png'
    save_path = os.path.join('./figures', save_name)
    plt.savefig(save_path)

def AIM_goal_visualize(alg, seed, epoch, env_name):
    goal_root_path = os.path.join('./saved_models', env_name, alg, 'seed-'+str(seed))
    ags_path = os.path.join(goal_root_path, 'ags','epoch_'+str(epoch)+'.csv')
    # candidate_goal_path = os.path.join(goal_root_path, 'candidate_goal','epoch_'+str(epoch)+'.csv')
    exploration_goal_path = os.path.join(goal_root_path, 'exploration_goal','epoch_'+str(epoch)+'.csv')

    ags = np.genfromtxt(ags_path,delimiter=',')
    # print(ags.shape)
    exploration_goal = np.genfromtxt(exploration_goal_path, delimiter=',')[:,:2]

    fig, ax = plt.subplots()
    # plot obstacle
    if env_name == 'FetchPushWallObstacle-v1':
        rect1 = patches.Rectangle((1.05, 0.4), 0.5, 0.7, linewidth=5, edgecolor="#7f7f7f", fill=False)
        rect2 = patches.Rectangle((1.22, 0.81), 0.04, 0.28, linewidth=3, fill=True, facecolor='g')
        rect3 = patches.Rectangle((1.22, 0.41), 0.04, 0.28, linewidth=3, fill=True, facecolor='g')
       
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)
        plt.xlim(1, 1.6)
        plt.ylim(0.3, 1.15)

    elif env_name == 'FetchPushDoubleObstacle-v1':
        rect1 = patches.Rectangle((1.05, 0.45), 0.5, 0.6, linewidth=5, edgecolor="#7f7f7f", fill=False)
        rect2 = patches.Rectangle((1.145, 0.64), 0.22, 0.04, linewidth=3, fill=True, facecolor='g')
        rect3 = patches.Rectangle((1.235, 0.86), 0.22, 0.04, linewidth=3, fill=True, facecolor='b')
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        plt.xlim(1, 1.6)
        plt.ylim(0.4, 1.1)
    
    for i in range(2000):
        ax.scatter(ags[i,0], ags[i,1], color='r')

    for i in range(exploration_goal.shape[0]):
        ax.scatter(exploration_goal[i,0], exploration_goal[i,1], color='g')


    save_name = alg+ '-'+env_name+'-seed-'+str(seed)+'-epoch-'+str(epoch)+'-exploration.png'
    save_path = os.path.join('./figures', save_name)
    plt.savefig(save_path)




def selected_goal_visualize(alg, seed, epoch, env_name):
    select_goal_path = os.path.join('saved_models',env_name + '/'+alg +'/seed-'+str(seed)+'/exploration_goal/'+'selected_goal_epoch_'+str(epoch)+'.csv')
    

   
    goals = np.array(pd.read_csv(select_goal_path))[:,:2]

    fig, ax = plt.subplots()
    # plot obstacle
    if env_name == 'FetchPushWallObstacle-v1':
        rect1 = patches.Rectangle((1.05, 0.4), 0.5, 0.7, linewidth=5, edgecolor="#7f7f7f", fill=False)
        rect2 = patches.Rectangle((1.22, 0.81), 0.04, 0.28, linewidth=3, fill=True, facecolor='g')
        rect3 = patches.Rectangle((1.22, 0.41), 0.04, 0.28, linewidth=3, fill=True, facecolor='g')
       
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)
        plt.xlim(1, 1.6)
        plt.ylim(0.3, 1.15)

    elif env_name == 'FetchPushDoubleObstacle-v1':
        rect1 = patches.Rectangle((1.05, 0.45), 0.5, 0.6, linewidth=5, edgecolor="#7f7f7f", fill=False)
        rect2 = patches.Rectangle((1.175, 0.66), 0.25, 0.04, linewidth=3, fill=True, facecolor='g')
        rect3 = patches.Rectangle((1.235, 0.86), 0.25, 0.04, linewidth=3, fill=True, facecolor='b')
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        plt.xlim(1, 1.6)
        plt.ylim(0.4, 1.1)
    ax.scatter(goals[:,0], goals[:,1],s=20, c='blue')
    save_name = alg+ '-'+env_name+'-seed-'+str(seed)+'-epoch-'+str(epoch)+'-selectgoals.png'
    save_path = os.path.join('./figures', save_name)
    plt.savefig(save_path)



def selected_goal_visualize_forall(alg, seed, env_name):
    epoch5 = os.path.join('saved_models',env_name + '/'+alg +'/seed-'+str(seed)+'/exploration_goal/'+'selected_goal_epoch_5'+'.csv')
    epoch14= os.path.join('saved_models',env_name + '/'+alg +'/seed-'+str(seed)+'/exploration_goal/'+'selected_goal_epoch_15'+'.csv')
    epoch21= os.path.join('saved_models',env_name + '/'+alg +'/seed-'+str(seed)+'/exploration_goal/'+'selected_goal_epoch_25'+'.csv')


    def goal_filter(goals):
            idxs_1 = np.where(goals[:,0]>1.06)
            idxs_2 = np.where(goals[:,0]<1.55)
            idxs_3 = np.where(goals[:,1]>0.4)
            idxs_4 = np.where(goals[:,1]<1.04)
            
            from functools import reduce
            idxs = reduce(np.intersect1d, [idxs_1[0], idxs_2[0],idxs_3[0], idxs_4[0]])
            # idxs = idxs_1[0] and idxs_2[0] and idxs_3[0] and idxs_4[0]
            return goals[idxs, :]
    if alg=='OMEGA':
        goals_5 = np.array(pd.read_csv(epoch5, delimiter='\t'))[:,:2]
        goals_14 = np.array(pd.read_csv(epoch14,delimiter='\t'))[:,:2]
        goals_21 = np.array(pd.read_csv(epoch21,delimiter='\t'))[:,:2]


        goals_5 = goal_filter(np.unique(goals_5, axis=0))
        goals_5 = goals_5[np.random.randint(goals_5.shape[0],size=100), :]
        goals_14 = goal_filter(np.unique(goals_14, axis=0))
        goals_14 = goals_14[np.where(goals_14[:,1]>0.6)[0],:]
        print(goals_14.shape)
        goals_14 = goals_14[np.random.randint(goals_14.shape[0],size=50), :]
        goals_21 = goal_filter(np.unique(goals_21, axis=0))
        goals_21 = goals_21[np.where(goals_21[:,1]>0.6)[0],:]
        goals_21 = goals_21[np.random.randint(goals_21.shape[0],size=100), :]


    else:
        
        goals_5 = goal_filter(np.array(pd.read_csv(epoch5))[:,:2])
        goals_14 = goal_filter(np.array(pd.read_csv(epoch14))[:,:2])
        goals_21 = goal_filter(np.array(pd.read_csv(epoch21))[:,:2])
        goals_5 = goal_filter(np.unique(goals_5, axis=0))
        goals_5 = goals_5[np.random.randint(goals_5.shape[0],size=100), :]
        goals_14 = goal_filter(np.unique(goals_14, axis=0))
        # print(goals_14.shape)
        goals_14 = goals_14[np.random.randint(goals_14.shape[0],size=80), :]
        goals_21 = goal_filter(np.unique(goals_21, axis=0))
        goals_21 = goals_21[np.random.randint(goals_21.shape[0],size=100), :]

    fig, ax = plt.subplots()
    # plot obstacle
    if env_name == 'FetchPushWallObstacle-v1':
        rect1 = patches.Rectangle((1.05, 0.4), 0.5, 0.7, linewidth=5, edgecolor="#7f7f7f", fill=False)
        rect2 = patches.Rectangle((1.22, 0.81), 0.04, 0.28, linewidth=3, fill=True, facecolor='g')
        rect3 = patches.Rectangle((1.22, 0.41), 0.04, 0.28, linewidth=3, fill=True, facecolor='g')
       
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)
        plt.xlim(1, 1.6)
        plt.ylim(0.3, 1.15)

    elif env_name == 'FetchPushDoubleObstacle-v1':
        rect1 = patches.Rectangle((1.05, 0.45), 0.5, 0.6, linewidth=5, edgecolor="#7f7f7f", fill=False)
        rect2 = patches.Rectangle((1.055, 0.64), 0.25, 0.05, linewidth=3, fill=True, facecolor='g')
        rect3 = patches.Rectangle((1.235, 0.86), 0.25, 0.05, linewidth=3, fill=True, facecolor='g')
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)
        plt.xlim(1, 1.6)
        plt.ylim(0.4, 1.1)
    elif env_name == 'FetchPushNew-v1':
        rect1 = patches.Rectangle((1.05, 0.45), 0.5, 0.6, linewidth=5, edgecolor="#7f7f7f", fill=False)
        rect2 = patches.Rectangle((1.05, 0.71), 0.25, 0.08, linewidth=3, fill=True, facecolor='g')
        # rect3 = patches.Rectangle((1.235, 0.86), 0.25, 0.05, linewidth=3, fill=True, facecolor='g')
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        # ax.add_patch(rect3)
        plt.xlim(1, 1.6)
        plt.ylim(0.4, 1.1)
    
    colors1 = [0 for _ in range(goals_5.shape[0])]
    colors2 = [0.2 for _ in range(goals_14.shape[0])]
    colors3 = [1 for _ in range(goals_21.shape[0])]
    colors =np.array(colors1+ colors2+colors3)
    print(colors.shape)
    goals = np.concatenate((goals_5, goals_14, goals_21),axis=0)
    c = ax.scatter(goals[:,0], goals[:,1],s=30, c=colors, cmap='RdBu')
    fig.colorbar(c, ax=ax)

    save_name = alg+ '-'+env_name+'-seed-'+str(seed)+'-selectgoals.png'
    save_path = os.path.join('./figures', save_name)
    plt.savefig(save_path)

if __name__ == '__main__':
    args = get_args()
    env,_ = make_env(args.env_name)
    ags = []
    dgs = []
    # for ep in range(1000):
    #     obs = env.reset()
    #     ags.append(obs['achieved_goal'])
    #     dgs.append(obs['desired_goal'])

    csv_path ='/data1/ydy/RL/HER_v3/saved_models/FetchPushWallObstacle-v1/MRN-MinQ/seed-6/ags/epoch_3.csv'
    path2 = '/data1/ydy/RL/HER_v3/saved_models/FetchPushDoubleObstacle-v1/MRN_MinQ/seed-5/exploration_goal/epoch_0.csv'
    # with open(csv_path, 'r') as f:
    #     bgs = pd.read_csv(csv_path)
    #     bgs = np.array(bgs)[:,:3]
    # with open(path2, 'r') as f:
    #     ags = pd.read_csv(path2)
    #     ags = np.array(ags)[:,:3]
 
    # Visual_FetchPushObstacle(np.array(bgs),np.array(ags))
    
    # for i in range(20,25):
        # print('epoch:',i)
    #     # goal_distribution_visualize_FetchPushObstacle(alg='MRN_MinQ', seed=7, epoch=i, env_name='FetchPushDoubleObstacle-v1')
    #     AIM_goal_visualize(alg='AIM',seed=5, epoch=i, env_name='FetchPushWallObstacle-v1')
        # selected_goal_visualize(alg='MEGA_MinV',seed=5, epoch=i,env_name='FetchPushWallObstacle-v1')
    selected_goal_visualize_forall(alg='OMEGA',seed=5, env_name='FetchPushNew-v1')