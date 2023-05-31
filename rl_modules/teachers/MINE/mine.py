import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import wandb
import os
from mpi4py import MPI


class MineNet(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(MineNet, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.fc1.bias, val=0)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.fc2.bias, val=0)
        nn.init.normal_(self.fc3.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.fc3.bias, val=0)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        y = self.fc3(x)
        return y

class MineTeacher(object):
    def __init__(self, args, env_params, buffer, o_norm, g_norm):
        self.env_params = env_params
        self.buffer = buffer
        self.args = args
        self.o_norm = o_norm
        self.g_norm = g_norm
        self.iteration = self.args.mine_iter
        self.batch_size = self.args.mine_batch
        self.mine_net = MineNet(input_size=env_params['goal']*2)
        self.unbias_loss = False
        self.avg_et = 1.0
        self.epoch = 0
        self.mine_net_optim = torch.optim.Adam(self.mine_net.parameters(), lr=1e-3)
        self.save_root = os.path.join(self.args.save_dir, self.args.env_name, self.args.alg, 'seed-'+str(self.args.seed))
        self.model_path = os.path.join(self.save_root, 'models')
        if MPI.COMM_WORLD.Get_rank() == 0:
            os.makedirs(self.model_path, exist_ok=True)
        self.save_frequency = self.args.save_interval
        self.scale = self.args.reward_scale
    
    def update(self):
        result = []
        avg_et = 1.0
        for i in range(self.iteration):
            joint_data = self.sample_batch(self.batch_size, sample_mode='joint')
            marginal_data = self.sample_batch(self.batch_size, sample_mode='marginal')
            
            
            joint, marginal = torch.from_numpy(joint_data), torch.from_numpy(marginal_data)

        
            T_joint = self.mine_net(joint)
            T_marginal = self.mine_net(marginal)
            T_marginal_exp = torch.exp(T_marginal)

            if self.unbias_loss:
                avg_et = 0.99 * avg_et + 0.01 * torch.mean(T_marginal_exp)
                # mine_estimate_unbiased = torch.mean(T_joint) - (1/avg_et).detach() * torch.mean(T_marginal_exp)
                mine_estimate = torch.mean(T_joint) - (torch.mean(T_marginal_exp)/avg_et).detach() * torch.log(torch.mean(T_marginal_exp))
                loss = -1. * mine_estimate

            else:
                mine_estimate = torch.mean(T_joint) - torch.log(torch.mean(T_marginal_exp))
                loss = -1. * mine_estimate
            self.mine_net.zero_grad()
            loss.backward()
            self.mine_net_optim.step()
            result.append(mine_estimate.detach().numpy())
        mi_lb = np.array(result).mean()
        mi_lb_mean = MPI.COMM_WORLD.allreduce(mi_lb, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        # if MPI.COMM_WORLD.Get_rank() == 0:
        #     wandb.log({'mutual_info_mean': mi_lb_mean})
        #     if self.epoch % self.save_frequency == 0:
        #         self.save_model()
        self.epoch += 1
      
    def compute_reward(self, obs, obs_next, clip=True):
        
        robot_state, object_state = self.split_robot_state_from_observation(self.args.env_name, obs)
        next_robot_state, object_next_state  =self.split_robot_state_from_observation(self.args.env_name, obs_next)
        joint = torch.from_numpy(np.hstack([robot_state, object_state]).astype(np.float32))
        marginal = torch.from_numpy(np.hstack([next_robot_state, object_state]).astype(np.float32))
        with torch.no_grad():
            T_joint = self.mine_net(joint)
            T_marginal = self.mine_net(marginal)
            T_marginal_exp = torch.exp(T_marginal)
            mine_estimate = torch.mean(T_joint, dim=1) - torch.log(torch.mean(T_marginal_exp,dim=1))
        if clip:
            r = np.clip(self.scale * mine_estimate.detach().numpy(), 0, 0.5).reshape(1, -1)
        else:
            r = self.scale * (mine_estimate.detach().numpy())
        return r


    def sample_batch(self, batchsize, sample_mode='joint'):
        assert sample_mode in ['joint', 'marginal']
        if self.args.use_per:
            transitions, _ = self.buffer.sample(batchsize)
        else:
            transitions = self.buffer.sample(batchsize)
        state_batch, skill_batch = self.process_transitions(transitions=transitions)

        if sample_mode == 'joint':
            return np.hstack([state_batch, skill_batch]).astype(np.float32)
        elif sample_mode=='marginal':
            index = np.arange(state_batch.shape[0])
            np.random.shuffle(index)
            skill_batch_new = skill_batch[index]
            return np.hstack([state_batch, skill_batch_new]).astype(np.float32)

    
        


    def process_transitions(self, transitions):
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        obs_norm = self.o_norm.normalize(transitions['obs'])
        # g_norm = self.g_norm.normalize(transitions['g'])
        gripper_pos, object_pos = self.split_robot_state_from_observation(self.args.env_name, obs_norm)
        return gripper_pos, object_pos



    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    def split_robot_state_from_observation(self, env_name, observation, type='gripper_pos'):
        obs = np.asarray(observation)
        dimo = obs.shape[-1]
        if env_name.lower().startswith('fetch'):
            assert dimo == 25, "Observation dimension changed."
            grip_pos, object_pos, object_rel_pos, gripper_state, object_rot, object_velp, object_velr, grip_velp, gripper_vel =\
                np.hsplit(obs, np.array([3, 6, 9, 11, 14, 17, 20, 23]))
            if type =='gripper_pos_vel':
                robot_state =np.concatenate((grip_pos.copy(), grip_velp.copy()), axis=-1)
            elif type == 'gripper_pos':
                robot_state = grip_pos.copy()
            obs_achieved_goal = object_pos.copy()
            return robot_state, obs_achieved_goal
        elif env_name.lower().startswith('hand'):
            assert NotImplementedError
            return None

    def save_model(self):
        torch.save(self.mine_net.state_dict(),os.path.join(self.model_path, 'mine_net_'+'epoch_'+str(self.epoch)+'.pt'))

    def load(self, epoch):
        path = os.path.join(self.model_path, 'mine_net_'+'epoch_'+str(epoch)+'.pt')
        if os.path.exists(path):
            weights = torch.load(path)
            self.mine_net.load_state_dict(weights)

       