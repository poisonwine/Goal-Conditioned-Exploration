
import numpy as np
from utils.envbuilder import make_env
from arguments import get_args, TD3_config, SAC_config

def split_robot_state_from_observation(env_name, observation, type):
    obs = np.asarray(observation)
    dimo = obs.shape[-1]
    if env_name.lower().startswith('fetch'):
        assert dimo == 25, "Observation dimension changed."
        grip_pos, object_pos, object_rel_pos, gripper_state, object_rot, object_velp, object_velr, grip_velp, gripper_vel =\
            np.hsplit(obs, np.array([3, 6, 9, 11, 14, 17, 20, 23]))
        if type =='PV':
            robot_state =np.concatenate((grip_pos.copy(), grip_velp.copy()), axis=-1)
        elif type == 'P':
            robot_state = grip_pos.copy()
        obs_achieved_goal = object_pos.copy()
        return robot_state, obs_achieved_goal
    elif env_name.lower().startswith('hand'):
        assert NotImplementedError
        return None



def sample_uniform_goal(env, batchsize):
    selected_goals = []
    if env.has_object:
        obs = env.reset()
        goal_height = obs['achieved_goal'][2]
        for _ in range(batchsize):
            object_xpos = env.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - env.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = env.initial_gripper_xpos[:2] +np.random.uniform(-env.obj_range, env.obj_range, size=2)
            object_pos = list(object_xpos)+[goal_height]
            selected_goals.append(object_pos)
    else:

        for _ in range(batchsize):
            object_xpos = env.initial_gripper_xpos.copy()
            while np.linalg.norm(object_xpos - env.initial_gripper_xpos) < 0.1:
                object_xpos = env.initial_gripper_xpos +np.random.uniform(-env.obj_range, env.obj_range, size=3)
            selected_goals.append(list(object_xpos))

    return np.array(selected_goals)



if __name__ == '__main__':
    args = get_args()
    args.env_name = 'FetchSlide-v1'
    env = make_env(args.env_name)
    obs_dict = env.reset()
    obs = obs_dict['observation']
    input_obs = np.repeat(obs.reshape(1, -1), repeats=50, axis=0)
    a = np.random.rand(5,3)
    b = np.random.rand(6, 3)
    c = np.random.rand(10, 3)

    print(np.concatenate((a,b,c),axis=0).shape)
    import torch
    a = torch.rand(5, 13)
    b = torch.rand(5, 8)
    c = torch.cat((a,b), dim=-1).type(torch.float32)
    print(c.size())