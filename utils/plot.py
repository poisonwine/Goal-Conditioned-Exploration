# DEPRECATED, use baselines.common.plot_util instead

import os
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns; sns.set()
import glob2
import glob
import argparse
import seaborn as sns; sns.set()

sns.set_style('darkgrid')
import matplotlib
# matplotlib.use('TkAgg') # Can change to 'Agg' for non-interactive mode
matplotlib.rcParams.update({'font.size': 15})
def smooth_curve(x, y):
    halfwidth = int(np.ceil(len(x) / 30))  # Halfwidth of our smoothing convolution

    #print(halfwidth)
    k = halfwidth
    xsmoo = x
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(2 * k + 1),
        mode='same')
    # print(xsmoo, ysmoo)
    return xsmoo, ysmoo


def load_results(file):
    if not os.path.exists(file):
        return None
    with open(file, 'r') as f:
        lines = [line for line in f]
    if len(lines) < 2:
        return None
    keys = [name.strip().lower() for name in lines[0].split(',')]
    data = np.genfromtxt(file, delimiter=',', skip_header=1, filling_values=0.)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.ndim == 2
    assert data.shape[-1] == len(keys)
    result = {}
    for idx, key in enumerate(keys):
        result[key] = data[:, idx]
    return result


def pad(xs, value=np.nan):
    maxlen = np.max([len(x) for x in xs])
    padded_xs = []
    for x in xs:
        if x.shape[0] >= maxlen:
            padded_xs.append(x)

        padding = np.ones((maxlen - x.shape[0],) + x.shape[1:]) * value
        x_padded = np.concatenate([x, padding], axis=0)
        assert x_padded.shape[1:] == x.shape[1:]
        assert x_padded.shape[0] == maxlen
        padded_xs.append(x_padded)
    return np.array(padded_xs)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='./figures')
    parser.add_argument('--smooth', type=int, default=True)
    parser.add_argument('--env_name', type=str, default='FetchThrowRubberBall-v0')
    parser.add_argument('--alg', type=str, default='baseline')
    parser.add_argument('--load_dir', type=str, default='../results')
    parser.add_argument('--plot_mode', type=str, default='baseline',help='choose to plot baseline or ablation study')
    args = parser.parse_args()
    return args

# Load all data.
def load_all_data(alg, args):
    data = []
    y_values = []
    smooth_value=[]
    #paths = [os.path.abspath(os.path.join(path, '..')) for path in glob2.glob(os.path.join(args.dir, '**', 'progress_cher.csv'))]
    path = os.path.join(args.load_dir, args.env_name, alg)
    paths = glob.glob(os.path.join(path, 'progress*.csv'))
    print(paths)
    for curr_path in paths:
        results = load_results(os.path.join(curr_path))
        if not results:
            print('skipping {}'.format(curr_path))
            continue
        #print('loading {} ({})'.format(curr_path, len(results['epoch'])))

        success_rate = np.array(results['test/success_rate'])
        epoch = np.arange(1, success_rate.shape[0]+1)
        #epoch = np.array(results['epoch']) + 1
        # Process and smooth data.
        assert success_rate.shape == epoch.shape
        x = epoch
        y = success_rate
        y_values.append(y)
        if args.smooth:
            x, y = smooth_curve(epoch, success_rate)
            smooth_value.append(y)
        assert x.shape == y.shape


    y_std = np.std(np.array(smooth_value), axis=0)
    y_mean = np.mean(np.array(smooth_value), axis=0)
    data.append([x, y_mean, y_std])
    return data


def load_reward_ratio(alg, args):
    data = []
    path = os.path.join(args.load_dir, args.env_name, alg, 'reward', 'r_ratio_per_epoch.csv')
    print(path)
    with open(path, 'r') as f:
        line = f.readlines()[0]
    y = line.split(',')
    for x in y:
        data.append(float(x))
    epoch = np.arange(1, len(y)+1)
    epoch, r = smooth_curve(epoch, np.array(data))
    return epoch, r

def plot_baselines():
    all_data = []
    args = get_args()
    args.env_name = 'FetchPickAndPlace-v1'
    #algs = ['ours','HER']
    # algs = ['HER', 'CHER','MEP','HGG','OMEGA','ours','ours_density']
    # algs = ['MEGA_MinV','HER', 'PER','CHER','HGG','OMEGA','VDS']
    algs = ['HER_(0-10cm)','HER_(5-20cm)','HER_(10-30cm)']
    #colors =['r', 'b', 'g', '#9467bd']
    colors = [
        '#d62728',  # brick red
        '#1f77b4',  # muted blue
        '#2ca02c',  # cooked asparagus green
        '#ff7f0e',  # safety orange
        '#e377c2',  # raspberry yogurt pink
        '#9370DB',  # light green
        # '#d62728',  # brick red
        'slategray',
        # '#e377c2',  # raspberry yogurt pink
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf'  # blue-teal
    ]
    linestyles = ['-', '-.', ':', '--', '--', '-', '--']
    linestyles = ['-', '-', '-', '-','-', '-','-', '-']

    #data = load_results('../results/FetchPnPObstacle-v1/OMEGA/progress_444_explor_ratio_0.5_03_000.csv')

    for alg in algs:
        data1 = load_all_data(alg, args)
        all_data.append(data1)
        print(all_data)
    plt.figure(figsize=(8, 6))
    sns.despine(left=True, bottom=True)
    for i, data in enumerate(all_data):
        #xs, y_mean, y_std = zip(*all_data[i])
        xs, y_mean, y_std = zip(*data)
        xs = np.array(xs).flatten()
        y_lower = np.array(y_mean) - np.array(y_std)
        y_lower = np.clip(y_lower, 0, 1)
        y_lower = y_lower.flatten()
        y_upper = np.array(y_mean) + np.array(y_std)
        y_upper = np.clip(y_upper, 0, 1)
        y_upper = y_upper.flatten()
        assert xs.shape == y_lower.shape == y_upper.shape
        if algs[i] == 'OMEGA':
            plt.plot(xs, np.array(y_mean).flatten(), label=algs[i], color=colors[i], linewidth=2.5, linestyle=linestyles[i])
        elif algs[i] == 'MEP':
            plt.plot(xs, np.array(y_mean).flatten(), label=algs[i], color=colors[i], linewidth=2.5, linestyle=linestyles[i])
        # elif algs[i] == 'ours_density':
        #     plt.plot(xs, np.array(y_mean).flatten(), label=algs[i], color=colors[i], linewidth=2, marker='d', linestyle=linestyles[i])
        elif algs[i]== 'MEGA_MinV':
            plt.plot(xs, np.array(y_mean).flatten(), label='Ours', color=colors[i], linewidth=2.5, linestyle=linestyles[i])
        else:
            plt.plot(xs, np.array(y_mean).flatten(), label=algs[i], color=colors[i], linewidth=2.5, linestyle=linestyles[i])
        #plt.fill_between(xs[0], np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), color=colors[i], alpha=0.25)
        plt.fill_between(xs, y_lower, y_upper, color=colors[i], alpha=0.15)
    plt.xlabel('Epoch',fontdict=dict(fontsize=14))
    plt.ylabel('Average Success Rate',fontdict=dict(fontsize=16))
    plt.xticks(np.arange(0, 101, 20), fontproperties='Times New Roman', size=14)
    plt.yticks(np.arange(0, 1.05, 0.2),fontproperties='Times New Roman', size=14)
    plt.legend(fancybox=True)
    if args.env_name == 'FetchPushDoubleObstacle-v1':
        plt.title('PushDoubleObstacle',fontdict=dict(fontsize=15))
    elif args.env_name == 'FetchPushNew-v1':
        plt.title('PushObstacle',fontdict=dict(fontsize=15))
    elif args.env_name == 'FetchPushWallObstacle-v1':
        plt.title('PushMiddleGap',fontdict=dict(fontsize=15))
    elif args.env_name == 'FetchPickAndPlace-v2':
        plt.title('PnPInAir',fontdict=dict(fontsize=15))
    elif args.env_name == 'FetchPnPObstacle-v1':
        plt.title('PnPObstacle',fontdict=dict(fontsize=15))
    if args.env_name == 'FetchSlide-v1' or args.env_name == 'FetchThrowRubberBall-v0':
        plt.xlim(0, 100)
    else:
        plt.xlim(0, 50)

    os.makedirs(args.dir, exist_ok=True)
    plt.savefig(os.path.join(args.dir, 'fig_{}_{}.png'.format(args.alg, args.env_name)), dpi=400, bbox_inches='tight')
    # plt.show()

plot_baselines()



def plot_ablation_study():
    args = get_args()
    args.env_name='FetchPushWallObstacle-v1'
    all_data = []
    ablation_names = ['HER','ours','no_goal_argue', 'no_goal_explore', 'no_rfb']
    labels = ['HER', 'ours', 'ours w/o GA', 'ours w/o GE', 'ours w/o TA']
    for name in ablation_names:
        data1 = load_all_data(name, args)
        all_data.append(data1)
    colors = [
            '#1f77b4',  # muted blue
            '#d62728',  # brick red
            '#2ca02c',  # cooked asparagus green
            '#ff7f0e',  # safety orange
            '#e377c2',  # raspberry yogurt pink
            '#9370DB',  # light green
            '#d62728',  # brick red

            '#e377c2',  # raspberry yogurt pink
            '#7f7f7f',  # middle gray
            '#bcbd22',  # curry yellow-green
            '#17becf'  # blue-teal
        ]
    linestyles = ['-', '-.', ':', '--', '--', '-']

    # data = load_results('../results/FetchPnPObstacle-v1/OMEGA/progress_444_explor_ratio_0.5_03_000.csv')

    plt.figure(figsize=(8, 6))
    sns.despine(left=True, bottom=True)
    for i, data in enumerate(all_data):
        # xs, y_mean, y_std = zip(*all_data[i])
        xs, y_mean, y_std = zip(*data)
        xs = np.array(xs).flatten()
        y_lower = np.array(y_mean) - np.array(y_std)
        y_lower = np.clip(y_lower, 0, 1)
        y_lower = y_lower.flatten()
        y_upper = np.array(y_mean) + np.array(y_std)
        y_upper = np.clip(y_upper, 0, 1)
        y_upper = y_upper.flatten()
        assert xs.shape == y_lower.shape == y_upper.shape
        if ablation_names[i] == 'HER':
            plt.plot(xs, np.array(y_mean).flatten(), label=labels[i], color=colors[i], linewidth=2,
                     linestyle=linestyles[i], marker='*')
        else:
            plt.plot(xs, np.array(y_mean).flatten(), label=labels[i], color=colors[i], linewidth=3,
                     linestyle=linestyles[i])
        # plt.fill_between(xs[0], np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), color=colors[i], alpha=0.25)
        plt.fill_between(xs, y_lower, y_upper, color=colors[i], alpha=0.25)
        #plt.legend(fancybox=True, fontsize=13,loc='upper left')
    plt.xlabel('Epoch',fontdict=dict(fontsize=14))
    plt.ylabel('Average Success Rate',fontdict=dict(fontsize=16))
    plt.xticks(np.arange(0, 101, 20), fontproperties='Times New Roman', size=14)
    plt.yticks(np.arange(0, 1.05, 0.2), fontproperties='Times New Roman', size=14)

    #plt.title(args.env_name)
    if args.env_name == 'FetchSlide-v1' or args.env_name == 'FetchThrowRubberBall-v0':
        plt.xlim(0, 100)
    else:
        plt.xlim(0, 80)
    if args.env_name == 'FetchSlide-v1':
        plt.ylim(-0.01, 0.9)
    elif args.env_name == 'FetchThrowRubberBall-v0':
        plt.ylim(-0.01, 0.8)
    else:
        plt.ylim(-0.02, 1.1)

    plt.savefig(os.path.join(args.dir, 'ablation_fig_{}_{}.png'.format(args.alg, args.env_name)), dpi=400,
                bbox_inches='tight')
    plt.show()
    pass
#plot_ablation_study()

def plot_label():
    args =get_args()
    colors = [
        '#1f77b4',  # muted blue
        '#2ca02c',  # cooked asparagus green
        '#ff7f0e',  # safety orange
        '#e377c2',  # raspberry yogurt pink
        '#9370DB',  # light green
        '#d62728',  # brick red

        '#e377c2',  # raspberry yogurt pink
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf'  # blue-teal
    ]
    linestyles = ['-', '-.', ':', '--', '--', '-']
    algs = ['HER', 'CHER', 'MEP', 'HGG', 'OMEGA', 'ours']
    x = np.arange(0, 100).flatten()
    y = np.zeros((1, 100)).flatten()
    print(x)
    plt.figure(figsize=(8, 6))
    for i in range(len(linestyles)):
        if algs[i]== 'OMEGA':
            plt.plot(x, y,label=algs[i], color=colors[i], linewidth=2, linestyle=linestyles[i],marker='*')
        elif algs[i] =='MEP':
            plt.plot(x, y,label=algs[i], color=colors[i], linewidth=4, linestyle=linestyles[i])
        else:
            plt.plot(x, y, label=algs[i], color=colors[i], linewidth=2,linestyle=linestyles[i])
        leg = plt.legend(loc='center',fontsize=20)
        leg.get_frame().set_linewidth(0.0)
    plt.axis('off')
    plt.ylim(0.5, 1)
    plt.savefig(os.path.join(args.dir, 'label.png'), dpi=120,bbox_inches='tight')
    plt.show()

#plot_label()

def plot_explore_alpha():
    args = get_args()
    args.env_name = 'FetchPushNew-v1'
    all_data = []
    ablation_names = ['explore_alpha_0.2', 'MEGA_MinV', 'explore_alpha_0.8']
    labels = [ r'explore $\alpha=0.2$', r'explore $\alpha=0.5$',r'explore $\alpha=0.8$']
    for name in ablation_names:
        data1 = load_all_data(name, args)
        all_data.append(data1)
    colors = [
        '#1f77b4',  # muted blue
        '#d62728',  # brick red
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#ff7f0e',  # safety orange
        '#e377c2',  # raspberry yogurt pink
        '#9370DB',  # light green
        '#d62728',  # brick red

        '#e377c2',  # raspberry yogurt pink
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf'  # blue-teal
    ]
    linestyles = ['-', '-', '-', '-','-', '-','-', '-']

    # data = load_results('../results/FetchPnPObstacle-v1/OMEGA/progress_444_explor_ratio_0.5_03_000.csv')

    plt.figure(figsize=(8, 6))
    sns.despine(left=True, bottom=True)
    for i, data in enumerate(all_data):
        # xs, y_mean, y_std = zip(*all_data[i])
        xs, y_mean, y_std = zip(*data)
        xs = np.array(xs).flatten()
        y_lower = np.array(y_mean) - np.array(y_std)
        y_lower = np.clip(y_lower, 0, 1)
        y_lower = y_lower.flatten()
        y_upper = np.array(y_mean) + np.array(y_std)
        y_upper = np.clip(y_upper, 0, 1)
        y_upper = y_upper.flatten()
        assert xs.shape == y_lower.shape == y_upper.shape


        plt.plot(xs, np.array(y_mean).flatten(), label=labels[i], color=colors[i], linewidth=2.5, linestyle=linestyles[i])
        # plt.fill_between(xs[0], np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), color=colors[i], alpha=0.25)
        plt.fill_between(xs, y_lower, y_upper, color=colors[i], alpha=0.25)
        plt.legend(fancybox=True, fontsize=13, loc='upper left')
    plt.xlabel('Epoch', fontdict=dict(fontsize=14))
    plt.ylabel('Average Success Rate', fontdict=dict(fontsize=16))
    plt.xticks(np.arange(0, 101, 20), fontproperties='Times New Roman', size=14)
    plt.yticks(np.arange(0, 1.05, 0.2), fontproperties='Times New Roman', size=14)
    if args.env_name == 'FetchPushDoubleObstacle-v1':
        plt.title('PushDoubleObstacle')
    elif args.env_name == 'FetchPushNew-v1':
        plt.title('PushObstacle')
    elif args.env_name == 'FetchPushWallObstacle-v1':
        plt.title('PushMiddleGap')
    elif args.env_name == 'FetchPickAndPlace-v2':
        plt.title('PnPInAir')
    elif args.env_name == 'FetchPnPObstacle-v1':
        plt.title('PnPObstacle')
    # plt.title(args.env_name)
    if args.env_name == 'FetchSlide-v1' or args.env_name == 'FetchThrowRubberBall-v0':
        plt.xlim(0, 100)
    else:
        plt.xlim(0, 80)
    if args.env_name == 'FetchSlide-v1':
        plt.ylim(-0.01, 0.9)
    elif args.env_name == 'FetchThrowRubberBall-v0':
        plt.ylim(-0.01, 0.8)
    else:
        plt.ylim(-0.02, 1.1)

    plt.savefig(os.path.join(args.dir, 'explore_alpha_{}_{}.png'.format(args.alg, args.env_name)), dpi=400,
                bbox_inches='tight')
    plt.show()
# plot_explore_alpha()

def plot_batch():
    args = get_args()
    args.env_name = 'FetchSlide-v1'
    all_data = []
    ablation_names = ['HER', 'batch_64', 'ours', 'batch_256']
    labels = ['HER','TA_batchsize=64', 'TA_batchsize=128', 'TA_batchsize=256']
    for name in ablation_names:
        data1 = load_all_data(name, args)
        all_data.append(data1)
    colors = [
        '#1f77b4',  # muted blue
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#ff7f0e',  # safety orange
        '#e377c2',  # raspberry yogurt pink
        '#9370DB',  # light green
        '#d62728',  # brick red

        '#e377c2',  # raspberry yogurt pink
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf'  # blue-teal
    ]
    linestyles = ['-', '-', '-', '-','-', '-','-', '-']

    # data = load_results('../results/FetchPnPObstacle-v1/OMEGA/progress_444_explor_ratio_0.5_03_000.csv')

    plt.figure(figsize=(8, 6))
    sns.despine(left=True, bottom=True)
    for i, data in enumerate(all_data):
        # xs, y_mean, y_std = zip(*all_data[i])
        xs, y_mean, y_std = zip(*data)
        xs = np.array(xs).flatten()
        y_lower = np.array(y_mean) - np.array(y_std)
        y_lower = np.clip(y_lower, 0, 1)
        y_lower = y_lower.flatten()
        y_upper = np.array(y_mean) + np.array(y_std)
        y_upper = np.clip(y_upper, 0, 1)
        y_upper = y_upper.flatten()
        assert xs.shape == y_lower.shape == y_upper.shape

        plt.plot(xs, np.array(y_mean).flatten(), label=labels[i], color=colors[i], linewidth=3, linestyle=linestyles[i])
        # plt.fill_between(xs[0], np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), color=colors[i], alpha=0.25)
        plt.fill_between(xs, y_lower, y_upper, color=colors[i], alpha=0.25)
        plt.legend(fancybox=True, fontsize=13, loc='upper left')
    plt.xlabel('Epoch', fontdict=dict(fontsize=14))
    plt.ylabel('Average Success Rate', fontdict=dict(fontsize=16))
    plt.xticks(np.arange(0, 101, 20), fontproperties='Times New Roman', size=14)
    plt.yticks(np.arange(0, 1.05, 0.2), fontproperties='Times New Roman', size=14)

    # plt.title(args.env_name)
    if args.env_name == 'FetchSlide-v1' or args.env_name == 'FetchThrowRubberBall-v0':
        plt.xlim(0, 100)
    else:
        plt.xlim(0, 100)
    if args.env_name == 'FetchSlide-v1':
        plt.ylim(-0.01, 0.9)
    elif args.env_name == 'FetchThrowRubberBall-v0':
        plt.ylim(-0.01, 0.8)
    else:
        plt.ylim(-0.02, 1.1)

    plt.savefig(os.path.join(args.dir, 'behbatch_alpha_{}_{}.png'.format(args.alg, args.env_name)), dpi=400,
                bbox_inches='tight')
    plt.show()
#plot_batch()

def plot_ablation_goal_evaluate_module(mode=1,env_name='FetchPushWallObstacle-v1'):
    all_data = []
    args = get_args()
    args.env_name = env_name
    # mode = 5
    if mode ==1:
        name = 'different_module'
        algs = [ 'MEGA_MinV','MinV_mine','MEGA','no_curriculum_select']
        labels = ['Ours','Ours--no Coverage Measure','Ours--no Novelty Measure','Ours--no Curriculum Select']
    elif mode==2:
        name = 'state_discover_method'
        algs = [ 'MinV_mine','MinV_icm','MinV_random']
        labels = ['MINE','ICM','Random']
    elif mode ==3:
        name = 'age_lambda'
        algs = [ 'MEGA_MinV','MEGA_MinV_05','MEGA_MinV_08']
        labels = [r'$\lambda=0.2$',r'$\lambda=0.5$',r'$\lambda=0.8$']
    elif mode == 4:
        name = 'bandwith'
        algs = ['MEGA','MEGA_bandwith_03','MEGA_bandwith_05']
        labels = ['bandwith 0.1(only Coverage Measure)', 'bandwith 0.3','bandwith 0.5']
    elif mode==5:
        name = 'baseline'
        algs  = ['MEGA_MinV','HER', 'PER','CHER','HGG','OMEGA','VDS']
        labels =  ['DEST(Ours)','HER', 'PER','CHER','HGG','OMEGA','VDS']
    elif mode ==6:
        name = 'explore_alpha'
        algs = ['MEGA_MinV','explore_alpha_0.2',  'explore_alpha_0.8']
        labels = [ r'explore $\alpha=0.5$', r'explore $\alpha=0.2$',r'explore $\alpha=0.8$']
    elif mode == 7:
        name = 'density_method'
        algs = ['MEGA_MinV','MEGA','MEGA_rnd','MEGA_flow']
        labels = ['DEST(Ours)','DEST-only Kernel Density','DEST-only Rnd Density','DEST-only Flow Density']
    
    colors = [
        '#d62728',  # brick red
        '#1f77b4',  # muted blue
        '#2ca02c',  # cooked asparagus green
        '#ff7f0e',  # safety orange
        '#e377c2',  # raspberry yogurt pink
        '#9370DB',  # light green
        # '#d62728',  # brick red
        'slategray',
        # '#e377c2',  # raspberry yogurt pink
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf'  # blue-teal
    ]
    # linestyles = ['-', '-.', ':', '--', '--', '-', '--']
    linestyles = ['-', '-', '-', '-','-', '-','-', '-']

    #data = load_results('../results/FetchPnPObstacle-v1/OMEGA/progress_444_explor_ratio_0.5_03_000.csv')

    for alg in algs:
        data1 = load_all_data(alg, args)
        all_data.append(data1)
        print(all_data)
    plt.figure(figsize=(10, 7))
    sns.despine(left=True, bottom=True)
    for i, data in enumerate(all_data):
        #xs, y_mean, y_std = zip(*all_data[i])
        xs, y_mean, y_std = zip(*data)
        xs = np.array(xs).flatten()
        y_lower = np.array(y_mean) - np.array(y_std)
        y_lower = np.clip(y_lower, 0, 1)
        y_lower = y_lower.flatten()
        y_upper = np.array(y_mean) + np.array(y_std)
        y_upper = np.clip(y_upper, 0, 1)
        y_upper = y_upper.flatten()
        assert xs.shape == y_lower.shape == y_upper.shape
      
        plt.plot(xs, np.array(y_mean).flatten(), label=labels[i], color=colors[i], linewidth=3, linestyle=linestyles[i])
        #plt.fill_between(xs[0], np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), color=colors[i], alpha=0.25)
        plt.fill_between(xs, y_lower, y_upper, color=colors[i], alpha=0.15)
    plt.xlabel('Epoch',fontdict=dict(fontsize=14))
    plt.ylabel('Average Success Rate',fontdict=dict(fontsize=16))
    plt.xticks(np.arange(0, 101, 20), fontproperties='Times New Roman', size=14)
    plt.yticks(np.arange(0, 1.05, 0.2),fontproperties='Times New Roman', size=14)
    plt.legend(fancybox=True,fontsize=14)
    if args.env_name == 'FetchPushDoubleObstacle-v1':
        plt.title('PushDoubleObstacle',fontdict=dict(fontsize=15))
    elif args.env_name == 'FetchPushNew-v1':
        plt.title('PushObstacle',fontdict=dict(fontsize=15))
    elif args.env_name == 'FetchPushWallObstacle-v1':
        plt.title('PushMiddleGap',fontdict=dict(fontsize=15))
    elif args.env_name == 'FetchPickAndPlace-v2':
        plt.title('PnPInAir',fontdict=dict(fontsize=15))
    elif args.env_name == 'FetchPnPObstacle-v1':
        plt.title('PnPObstacle',fontdict=dict(fontsize=15))
    elif args.env_name == 'FetchSlide-v1':
        plt.title('Slide',fontdict=dict(fontsize=15))  
    if args.env_name == 'FetchSlide-v1' or args.env_name == 'FetchThrowRubberBall-v0':
        plt.xlim(0, 70)
    else:
        plt.xlim(0, 80)

    os.makedirs(args.dir, exist_ok=True)
    plt.savefig(os.path.join(args.dir, 'fig_{}_{}.png'.format(name, args.env_name)), dpi=400, bbox_inches='tight')
    # plt.show()
# for i in range(1,7):
#     plot_ablation_goal_evaluate_module(mode=i, env_name='FetchPushWallObstacle-v1')
#     plot_ablation_goal_evaluate_module(mode=i, env_name='FetchPushNew-v1')
# for i in [1,2,3,5]:
#     plot_ablation_goal_evaluate_module(mode=i, env_name='FetchPnPObstacle-v1')
# for i in [5]:
#     plot_ablation_goal_evaluate_module(mode=i, env_name='FetchPushDoubleObstacle-v1')
#     plot_ablation_goal_evaluate_module(mode=i, env_name='FetchPickAndPlace-v2')
# plot_ablation_goal_evaluate_module(mode=6, env_name='FetchPushNew-v1')