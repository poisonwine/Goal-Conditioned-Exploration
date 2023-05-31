import os
from gym import utils
from gym.envs.robotics import fetch_env
import numpy as np
import copy

# Ensure we get the path separator correct on windows
# MODEL_XML_PATH = os.path.join('fetch', 'push_moving_double_obstacle.xml')
MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'push_moving_double_obstacle.xml')
class FetchPushMovingDoubleObstacleEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        self.further = False

        # TODO: configure adaption parameters
        self.adapt_dict = dict()
        self.adapt_dict["field"] = [1.3, 0.75, 0.6, 0.25, 0.25, 0.2]

        #centers of the interval where goal and initial position will be sampld
        self.target_goal_center = np.array([1.28, 0.55, 0.425])
        self.object_center = np.array([1.3, 0.93, 0.425])
        #for moving
        self.vel_lims = [0.6, 0.9]
        self.n_moving_obstacles = 0
        self.current_obstacle_vels = []
        self.obstacle_directions = []
        self.obstacle_upper_limits = []
        self.obstacle_lower_limits = []
        self.pos_difs = []
        for _ in range(self.n_moving_obstacles):
            self.current_obstacle_vels.append(0.9)
            self.obstacle_directions.append(1)


        self.obstacle_upper_limits.append(1.33)
        self.obstacle_lower_limits.append(1.18)
        self.obstacle_upper_limits.append(1.42)
        self.obstacle_lower_limits.append(1.27)
        for i in range(self.n_moving_obstacles):
            self.pos_difs.append((self.obstacle_upper_limits[i] - self.obstacle_lower_limits[i]) / 2.)

        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.3, 0.93, 0.42505, 1., 0., 0., 0.],  # origin 0.53
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.02, target_range=0.05, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
        self.obstacle_slider_idxs = []
        self.obstacle_slider_idxs.append(self.sim.model.joint_names.index('obstacle:joint'))
        self.obstacle_slider_idxs.append(self.sim.model.joint_names.index('obstacle2:joint'))
        self.geom_id_object = self.sim.model.geom_name2id('object0')
        self.geom_ids_obstacles = []
        for name in ['o', 'o2']:
            self.geom_ids_obstacles.append(self.sim.model.geom_name2id(name))
        self.use_reset_sim = True

    # RobotEnv methods
    # ----------------------------

    def test_setup(self, new_vel_lims=[1.1, 1.4]):
        '''
        changes the parameter for further tests after training an agent
        '''
        # the default values makes the obstacle in average faster
        self.vel_lims = new_vel_lims
        for i in range(self.n_moving_obstacles):
            self.current_obstacle_vels[i] = new_vel_lims[1]

    def set_obstacle_slide_pos(self, positions):
        qpos = self.sim.data.qpos.flat[:]
        for i, pos in enumerate(positions):
            # move obstacles
            qpos[self.obstacle_slider_idxs[i]] = pos
        to_mod = copy.deepcopy(self.sim.get_state())
        to_mod = to_mod._replace(qpos=qpos)
        self.sim.set_state(to_mod)
        self.sim.forward()

    def set_obstacle_slide_vel(self, velocities):
        qvel = self.sim.data.qvel.flat[:]
        for i, vel in enumerate(velocities):
            qvel[self.obstacle_slider_idxs[i]] = vel
        to_mod = copy.deepcopy(self.sim.get_state())
        to_mod = to_mod._replace(qvel=qvel)
        self.sim.set_state(to_mod)
        self.sim.forward()

    def move_obstacle(self):
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        qpos = self.sim.data.qpos.flat[:]
        new_positions = []

        for i in range(self.n_moving_obstacles):
            current_qpos = qpos[self.obstacle_slider_idxs[i]]

            if self.obstacle_directions[i] == 1:
                if current_qpos >= self.pos_difs[i]:
                    new_pos = current_qpos - self.current_obstacle_vels[i] * dt
                    #self.set_obstacle_slide_pos(new_pos)
                    self.obstacle_directions[i] = -1
                else:
                    extra_dist = self.current_obstacle_vels[i] * dt
                    if current_qpos + extra_dist >= self.pos_difs[i]:
                        new_pos = self.pos_difs[i]
                        #self.set_obstacle_slide_pos(new_pos)
                        self.obstacle_directions[i] = -1
                    else:
                        new_pos = current_qpos + extra_dist
                        #self.set_obstacle_slide_pos(new_pos)

            else:
                if current_qpos <= -self.pos_difs[i]:
                    new_pos = current_qpos + self.current_obstacle_vels[i] * dt
                    #self.set_obstacle_slide_pos(new_pos)
                    self.obstacle_directions[i] = 1
                else:
                    extra_dist = self.current_obstacle_vels[i] * dt
                    if current_qpos - extra_dist <= -self.pos_difs[i]:
                        new_pos = -self.pos_difs[i]
                        #self.set_obstacle_slide_pos(new_pos)
                        self.obstacle_directions[i] = 1
                    else:
                        new_pos = current_qpos - extra_dist
                        #self.set_obstacle_slide_pos(new_pos)
            new_positions.append(new_pos)
        self.set_obstacle_slide_pos(new_positions)

    def step(self, action):
        self.move_obstacle()
        return super(FetchPushMovingDoubleObstacleEnv, self).step(action)

    def _set_gripper_during_setup(self):
        # Move end effector into position.
        orig_pos = self.sim.data.get_site_xpos('robot0:grip')
        gripper_target = np.array([-0.5399, 0.305, -0.306 + self.gripper_extra_height]) + orig_pos
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

    def _sample_goal(self):
        goal = self.target_goal_center + self.np_random.uniform(-self.target_range, self.target_range, size=3)
        goal[2] = self.height_offset
        return goal.copy()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        a = np.random.randint(2)
        if a == 0 :
            self.obstacle_directions = [1, -1]
        else:
            self.obstacle_directions = [-1, 1]
        velocities = []

        for i in range(self.n_moving_obstacles):
            possible_vels = np.linspace(start=self.vel_lims[0], stop=self.vel_lims[1], num=10, endpoint=True)
            vel = np.random.choice(possible_vels)
            self.current_obstacle_vels[i] = vel
            velocities.append(vel*self.obstacle_directions[i])
        self.set_obstacle_slide_vel(velocities)

        object_xpos = self.object_center[:2] + self.np_random.uniform(-self.obj_range,
                                                                      self.obj_range, size=2)
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        object_qpos[3:] = np.array([1., 0., 0., 0.])
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        self.sim.forward()

        return True

    def _get_obs(self):
        obs = super(FetchPushMovingDoubleObstacleEnv, self)._get_obs()
        body_id = self.sim.model.body_name2id('obstacle')
        pos1 = np.array(self.sim.data.body_xpos[body_id].copy())
        body_id2 = self.sim.model.body_name2id('obstacle2')
        pos2 = np.array(self.sim.data.body_xpos[body_id2].copy())
        dims = np.array([0.11, 0.02, 0.035])
        ob1 = np.concatenate((pos1, dims.copy()))
        ob2 = np.concatenate((pos2, dims.copy()))
        obs['real_obstacle_info'] = np.array([ob1, ob2])
        obs['real_size_goal'] = np.array([0.04, 0.04, 0.02])
        return obs
