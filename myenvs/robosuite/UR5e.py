import robosuite as suite
from robosuite.wrappers import GymWrapper
import numpy as np
from gym import spaces


class UR5eLift(object):
    def __init__(self, robot_name='UR5e', controller_name='OSC_POSITION', render=False, max_steps=50, distance_thresh=0.05, reward_shaping=False,reward_type='sparse'):
        # controll mode: OSC_POSITION,  OSC_POSE,JOINT_VELOCITY, JOINT_POSITION
        # robot name: UR5e,Baxter, Panda, Sawyer

        controller = suite.load_controller_config(default_controller=controller_name)
        self.env = GymWrapper(suite.make(env_name='Lift',
                              robots=robot_name,
                              controller_configs=controller,
                              use_camera_obs=False,
                              has_offscreen_renderer=False,
                              has_renderer=render,
                              reward_shaping=reward_shaping,
                              control_freq=20,
                              horizon=max_steps,
                              ))
        self.distance_threshold = distance_thresh
        self.robot = self.env.robots[0]
        self.sim = self.env.sim
        self.unwrapped = self.env.unwrapped
        self.spec = None
        self.xpos = self.robot.robot_model.base_xpos_offset["table"](self.env.table_full_size[0])
        self.joints_qpos = self.sim.data.qpos[self.robot._ref_gripper_joint_pos_indexes]
        obs = self.env.reset()
        self.cube_pos = self.sim.data.body_xpos[self.env.cube_body_id]
        self.goal_dim = self.cube_pos.shape[0]
        self.gripper_site_pos = self.sim.data.site_xpos[self.robot.eef_site_id]
        self.object_rel_pos = self.cube_pos - self.gripper_site_pos
        self.observation_dim = self.joints_qpos.shape[0] + self.goal_dim + self.gripper_site_pos.shape[0] * 2
        self.observation_space = spaces.Dict({
            # "observation": self.env.observation_space,
            "observation": spaces.Box(- np.inf * np.ones(self.observation_dim), np.inf * np.ones(self.observation_dim)),
            "desired_goal": spaces.Box(- np.inf * np.ones(self.goal_dim), np.inf * np.ones(self.goal_dim)),
            "achieved_goal": spaces.Box(- np.inf * np.ones(self.goal_dim), np.inf * np.ones(self.goal_dim)),
        })

        self.action_space = self.env.action_space
        self.action_dim = self.action_space.sample().shape[0]
        self._max_episode_steps = max_steps
        self.goal = self._sample_goal().copy()
        # self.xpos = self.robot.robot_model.base_xpos_offset["table"](self.env.table_full_size[0])
        # print(self.xpos)
        self.obj_range = 0.15

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        obs = self.get_obs()
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }

        return obs, reward, False, info

    def _is_success(self, achieved_goal, desired_goal):
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def reset(self):
        obs = self.env.reset()
        self.goal = self._sample_goal()
        obs_new = self.get_obs()
        return obs_new

    def get_obs(self):
        joint_qpos = self.sim.data.qpos[self.robot._ref_joint_pos_indexes]
        cube_pos = self.sim.data.body_xpos[self.env.cube_body_id] - self.xpos
        gripper_pos = self.sim.data.site_xpos[self.robot.eef_site_id] - self.xpos
        obj_rel_pos = cube_pos - gripper_pos
        obs = np.concatenate([joint_qpos, cube_pos, gripper_pos, obj_rel_pos])
        achieved_goal = cube_pos.copy()

        # print('joint', joint_qpos)
        return {'observation': obs.copy(),
                'achieved_goal': achieved_goal.copy(),
                'desired_goal': self.goal.copy()
        }


    def _sample_goal(self):

        noise_xy = np.random.uniform(low=-0.05, high=0.05, size=2)

        height = np.random.uniform(low=0, high=0.1, size=1)
        goal = self.sim.data.body_xpos[self.env.cube_body_id] - self.xpos
        
        goal[:2] += noise_xy
        if np.random.rand() > 0.5:
            goal[2] += height
        return goal.copy()

    def compute_reward(self, achieved, goal, info):
        dis = self.goal_distance(achieved, goal)
        return -(dis > self.distance_threshold).astype(np.float32)

    def goal_distance(self, achieved, goal):
        assert achieved.shape == goal.shape
        dis = np.linalg.norm(achieved - goal, axis=-1)
        return dis

    def render(self):
        self.env.render()


    def seed(self,seed=None):
        return self.env.seed(seed=seed)

if __name__ == '__main__':
    env = UR5eLift(robot_name='UR5e',
                   controller_name='OSC_POSITION',
                   render=False,
                   )
    # print(env.xpos)
    for i_episode in range(2):
        observation = env.reset()
        print(observation['desired_goal'], observation['achieved_goal'])

        # print(observation)
        for t in range(50):
            # env.render()
            action = env.action_space.sample()
            # print(action)
            observation, reward, done, info = env.step(action)
            # print(observation['desired_goal'], observation['achieved_goal'])
            # print(done)
            # print(env.sim.data.body_xpos[env.cube_body_id])

            # print(info)
            # print(env.robots[0].robot_joints)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break