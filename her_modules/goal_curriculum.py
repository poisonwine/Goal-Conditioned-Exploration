import numpy as np
import os
from math import pi, cos, sin, acos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

GoalOnDeskEnv = ['FetchSlide-v1', 'FetchPushWallObstacle-v1', 'FetchPushWallObstacle-v2', 'FetchThrowRubberBall-v0']
GoalInAirEnv = ['FetchPickAndPlace-v1','FetchReach-v1', 'FetchPnPObstacle-v1']
class GoalAugmentor(object):
    def __init__(self, env_name, error_distance, batch_size):
        self.env_name = env_name
        self.error_distance = error_distance
        self.batch_size = batch_size

    def goal_arguement(self, ags):
        assert ags.shape[1] == 3
        ags = np.array(ags)
        goal_num = ags.shape[0]
        selected_inds = np.random.randint(0, goal_num, self.batch_size)
        selected_goals = ags[selected_inds].copy()
        xs, ys, zs = self.generate_random_point_in_sphere(self.batch_size)
        for i, (offset_x, offset_y, offset_z) in enumerate(zip(xs, ys, zs)):
            if np.random.rand() < 0.8:
                selected_goals[i] = selected_goals[i] + np.array([offset_x, offset_y, offset_z])
            else:
                selected_goals[i] = selected_goals[i]
        # arguemented_goals = np.concatenate((ags, selected_goals), axis=0)
        return selected_goals

    def generate_random_point_in_sphere(self, goals_len):
        angle_xy = np.random.random(size=goals_len) * 2 * pi
        angle_z = np.random.random(size=goals_len) * 0.5 * pi

        xs = []
        ys = []
        zs = []

        if self.env_name in GoalOnDeskEnv:
            for a1, a2 in zip(angle_xy, angle_z):
                x = cos(a1) * cos(a2) * self.error_distance
                y = sin(a1) * cos(a2) * self.error_distance
                z = 0
                xs.append(x)
                ys.append(y)
                zs.append(z)
        elif self.env_name in GoalInAirEnv:
            for a1, a2 in zip(angle_xy, angle_z):
                x = cos(a1) * cos(a2) * self.error_distance
                y = sin(a1) * cos(a2) * self.error_distance
                z = sin(a2) * self.error_distance
                xs.append(x)
                ys.append(y)
                zs.append(z)
        else:
            print('no such env')

        return xs, ys, zs

if __name__ == '__main__':

    batchsize=256
    goal_sampler = Goal_Curriculum('FetchSlide-v1', 0.03, batch_size=256)
    x = (-1 + 2*np.random.random(size=128)) * 0.05
    y = (-1 + 2*np.random.random(size=128)) * 0.05

    goals = []
    for off_x, off_y in zip(x,y):
        goals.append(np.array([1.2, 0.8, 0.45]) + np.array([off_x, off_y, 0]))
    goals= np.array(goals)
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    ax.scatter(goals[:,0], goals[:, 1], goals[:, 2], color='r', label='origin_goal')
    arguement_goal = goal_sampler.goal_arguement(goals)
    ax.scatter(arguement_goal[goals.shape[0]:, 0], arguement_goal[goals.shape[0]:,1],arguement_goal[goals.shape[0]:,2], color='g',label='argment_goal')
    plt.legend()
    ax.view_init(elev=25, azim=20)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    ax.set_zbound(lower=0, upper=0.8)
    plt.ylim(0.6, 0.9)
    plt.show()