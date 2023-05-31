import math
import torch
import numpy as np
from utils.goal_utils import sample_uniform_goal
from rl_modules.teachers.abstract_teacher import AbstractTeacher
import os
import csv
from mpi4py import MPI

class EnsembleLinear(torch.nn.Module):
    __constants__ = ['in_features', 'out_features', 'k']
    in_features: int
    out_features: int
    k: int
    weight: torch.Tensor
    def __init__(self, in_features: int, out_features: int, k: int, bias: bool = True):
        super(EnsembleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.weight = torch.nn.Parameter(torch.Tensor(k, out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(k, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i in range(self.k):
            torch.nn.init.kaiming_uniform_(self.weight[i, ...], a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight[0, ...])
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # In this case we compute the predictions of the ensembles for the same data
        if len(input.shape) == 2:
            x = torch.einsum("kij,nj->kni", self.weight, input)
        # Here we compute the predictions of the ensembles for the data independently
        elif len(input.shape) == 3:
            x = torch.einsum("kij,knj->kni", self.weight, input)
        else:
            raise RuntimeError("Ensemble only supports predictions with 2- or 3D input")

        if self.bias is not None:
            return x + self.bias[:, None, :]
        else:
            return x

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, k={}, bias={}'.format(
            self.in_features, self.out_features, self.k, self.bias is not None
        )



class EnsembleQFunction(torch.nn.Module):

    def __init__(self, input_dim, layers, act_func, k):
        super().__init__()
        layers_ext = [input_dim] + layers + [1]
        torch_layers = []
        for i in range(len(layers_ext) - 1):
            torch_layers.append(EnsembleLinear(layers_ext[i], layers_ext[i + 1], k, bias=True))
        self.layers = torch.nn.ModuleList(torch_layers)
        self.act_fun = act_func

    def __call__(self, x):
        h = x
        for l in self.layers[:-1]:
            h = self.act_fun(l(h))

        return self.layers[-1](h)


class VDSteacher(AbstractTeacher):
    def __init__(self, args, env_params, env, buffer, ag_buffer, dg_buffer, policy, o_norm, g_norm):
        self.args = args
        self.env_params = env_params
        self.env = env
        self.buffer = buffer
        self.policy = policy
        self.o_norm = o_norm
        self.g_norm = g_norm
        self.gamma = self.args.vds_gamma
        self.act_dim = self.env_params['action']
        self.obs_dim = self.env_params['obs']
        self.goal_dim = self.env_params['goal']
        self.q_n = self.args.Q_num
        self.lr = self.args.vds_lr
        self.hidden_dim = self.args.vds_hidden_dim
        self.layer_num = self.args.vds_layer_num
        self.update_num = self.args.vds_batches
        self.batch_size = self.args.vds_batch_size
        net_layers = {"layers": [self.hidden_dim for _ in range(self.layer_num)], "act_func": torch.nn.ReLU()}

        self.Q_ensemble = EnsembleQFunction(**net_layers, input_dim=self.obs_dim+self.goal_dim+self.act_dim, k=self.q_n)
        self.q_optimizer = torch.optim.Adam(self.Q_ensemble.parameters(), lr=self.lr)
        self.ag_buffer = ag_buffer
        self.dg_buffer = dg_buffer
        self.n_candidate = self.args.n_candidate
    
       
        self.achieved_size =self.n_candidate
        # self.uniform_size = int(r_list[1] * self.n_candidate)
        
        self.candidate_goals = None
        self.likelihoods = None
        self.save_root = os.path.join(self.args.save_dir, self.args.env_name, self.args.alg)
        self.goal_save_path = os.path.join(self.save_root, 'seed-'+str(self.args.seed),'candidate_goal')
        self.model_path = os.path.join(self.save_root, 'models')
        if MPI.COMM_WORLD.Get_rank() == 0:
            os.makedirs(self.goal_save_path, exist_ok=True)
            os.makedirs(self.model_path, exist_ok=True)
        self.epoch = 0 
        self.save_fre =self.args.save_interval




    def update(self):
        if self.buffer.current_size > 0:
            for _ in range(self.update_num):
                if self.args.use_per:
                    transitions, _ = self.buffer.sample(self.batch_size * self.q_n)
                else:
                    transitions = self.buffer.sample(self.batch_size * self.q_n)
                obs, acts, next_obs, rewards = self.process_transitions(transitions)
                if self.args.agent == 'DDPG' or 'TD3':
                    next_actions = self.policy.actor_network(next_obs)
                elif self.args.agent=='SAC':
                    mu, log_sigma = self.policy.actor_network(next_obs)
                    next_actions = torch.from_numpy(self.policy._select_actions((mu, log_sigma)))
                #  shape (self.q_n, self.batchsize, -1)
                next_act = torch.reshape(next_actions, (self.q_n, self.batch_size, -1))
                next_ob = torch.reshape(next_obs, (self.q_n, self.batch_size, -1))
                cur_ob = torch.reshape(obs, (self.q_n, self.batch_size, -1))
                cur_act = torch.reshape(acts, (self.q_n, self.batch_size, -1))
                r = torch.reshape(rewards, (self.q_n, self.batch_size, -1))
                with torch.no_grad():
                    next_q_value = self.Q_ensemble(torch.cat((next_ob, next_act), axis=-1))
                target_q_values = r + self.gamma * next_q_value
                cur_q_value = self.Q_ensemble(torch.cat((cur_ob, cur_act),axis=-1))
                loss = torch.sum(torch.nn.functional.mse_loss(cur_q_value, target_q_values))
                self.q_optimizer.zero_grad()
                loss.backward()
                self.q_optimizer.step()
            # print('finish vds Q ensemble update')
            self.update_disagreements()
          
            pri = self.likelihoods.reshape(-1, 1)
            write_content = np.concatenate((self.candidate_goals, pri), axis=-1)
            if MPI.COMM_WORLD.Get_rank() == 0:
                with open(os.path.join(self.goal_save_path, 'epoch_'+str(self.epoch)+'.csv'), mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(write_content.tolist())
                if self.epoch % self.save_fre ==0:
                    self.save_models()
            self.epoch += 1

    def process_transitions(self, transitions):
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)
        return inputs_norm_tensor, actions_tensor, inputs_next_norm_tensor, r_tensor

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g


    def update_disagreements(self):

        ags = self.ag_buffer.random_sample(self.achieved_size)
       
        # ags +=  np.random.normal(scale=0.03, size=ags.shape)
        # candidate_goals = np.concatenate((ags, noisy_ags), axis=0)
        # self.candidate_goals = candidate_goals.copy()/
        self.candidate_goals = ags.copy()
        ## use statics obs
        obs_dict = self.env.reset()
        obs = obs_dict['observation']
        input_obs = np.repeat(obs.reshape(1, -1), repeats=self.candidate_goals.shape[0], axis=0)
        input_obs, candidate_goals = self._preproc_og(input_obs, self.candidate_goals)
        g_norm = self.g_norm.normalize(candidate_goals)
        obs_norm = self.o_norm.normalize(input_obs)
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        input_norm_obs = torch.tensor(inputs_norm, dtype=torch.float32)

        if self.args.agent == 'DDPG' or 'TD3':
            actions = self.policy.actor_network(input_norm_obs)
        elif self.args.agent == 'SAC':
            mu, log_sigma = self.policy.actor_network(input_norm_obs)
            actions = torch.from_numpy(self.policy._select_actions((mu, log_sigma)))
        q_input = torch.cat((input_norm_obs, actions), dim=-1).type(torch.float32)
        disagreements = np.std(np.squeeze(self.Q_ensemble(q_input).detach().numpy()), axis=0)
        self.likelihoods = disagreements / np.sum(disagreements)
        # print('####')


    def sample(self, batchsize):
        if self.likelihoods is not None and self.candidate_goals is not None:
            inds = np.random.choice(self.candidate_goals.shape[0], size=batchsize, replace=False, p=self.likelihoods.flatten())
            selected_goals = self.candidate_goals[inds].copy()

        else:
            ags = self.ag_buffer.random_sample(int(batchsize * 0.5))
            dgs = self.dg_buffer.random_sample(int(batchsize * 0.5))
            selected_goals = np.concatenate((ags, dgs), axis=0)
        return selected_goals


    def save_models(self):
        save_path = os.path.join(self.model_path, 'Q_ensemble_'+'epoch_'+str(self.epoch)+'.pt')
        torch.save(self.Q_ensemble, save_path)

    def save(self, path):

        pass
    def load(self, path):
        pass




if __name__=='__main__':
    net_arch = {"layers": [128, 128, 128], "act_func": torch.nn.ReLU()}
    qs = EnsembleQFunction(**net_arch, input_dim=10, k=3)
    x = torch.rand(3, 256, 10)
    print(qs(x).size())






