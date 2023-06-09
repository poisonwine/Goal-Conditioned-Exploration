import os
from gym import utils
from myenvs.baxter import baxter_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('baxter', 'pick_and_place.xml')


class BaxterPickAndPlaceEnv(baxter_env.BaxterEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            # 'robot0:slide0': 0.0, #-----baxter
            # 'robot0:slide1': 0.0, #-----baxter
            # 'robot0:slide2': 0.0, #-----baxter
            'robot0:right_s0': 0.07669903930664063, #-----baxter
            'robot0:right_s1': -0.9660244000671387, #-----baxter
            'robot0:right_w0': -0.6438884349792481, #-----baxter
            'robot0:right_w1':  1.0074418812927246, #-----baxter
            'robot0:right_w2':  0.483203947631836, #-----baxter
            'robot0:right_e0':  1.1481846184204103, #-----baxter
            'robot0:right_e1':  1.9232284106140138, #-----baxter
            'robot0:left_w0': 0.6477233869445801, #-----baxter
            'robot0:left_w1': 1.007825376489258, #-----baxter
            'robot0:left_w2': -0.48282045243530275, #-----baxter
            'robot0:left_e0': -1.1504855895996096, #-----baxter
            'robot0:left_e1': 1.9232284106140138, #-----baxter
            'robot0:left_s0': -0.07823302009277344, #-----baxter
            'robot0:left_s1': -0.9675583808532715, #-----baxter
            'object0:joint': [1.65, 0.259027, -0.15, 1., 0., 0., 0.],
            # 'object0:joint': [0., 0., 0., 0., 0., 0., 0.],
            

        }
        baxter_env.BaxterEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20, #TODO: How n_substeps slow down the step in viewer?
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0, #TODO: what's the extra height means?
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, single_goal=False)
        utils.EzPickle.__init__(self)
