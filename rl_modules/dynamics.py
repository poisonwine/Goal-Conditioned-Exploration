import numpy as np
import torch
import torch.nn as nn

class ForwardModel(nn.Module):
    def __init__(self, env_param, hidden_dim):
        super(ForwardModel,self).__init__()
        input_shape = env_param['obs'] + env_param['action']
        out_shape = env_param['obs']
        self.model = nn.Sequential(
                nn.Linear(input_shape, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_shape) 
        )
        self.max_action = env_param['action_max']

    def forward(self, obs, action):
        if len(obs.shape) == 1:
            x = torch.cat([obs, action], dim=-1).unsqueeze(0)
        else:
            x = torch.cat([obs, action], dim=-1)
        out = self.model(x)
        return out


    def init_weight(self):
        pass

class ForwardDynamics(object):
    def __init__(self, args, env, env_params, o_norm, buffer, hidden_dim, learning_rate=1e-4, name='1'):
        self.args = args
        self.model = ForwardModel(env_param=env_params, hidden_dim=hidden_dim)
        self.env = env
        self.o_norm = o_norm
        self.buffer = buffer
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.max_ir = 0.5
        self.name = name
        

    def predict_next_state(self, obs, action):
        # for i in range(step):
        norm_obs = self.o_norm.normalize(obs)
        obs_tensor = torch.tensor(norm_obs, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        with torch.no_grad():
            next_obs_norm = self.model(obs_tensor, action_tensor) + obs_tensor
        next_obs = self.o_norm.denormalize(next_obs_norm.numpy())
        return next_obs


    def get_intrinsic_reward(self, obs, action, next_obs):
        obs0_norm = self.o_norm.normalize(obs)
        obs1_norm = self.o_norm.normalize(next_obs)
        diff_obs = torch.tensor(obs1_norm - obs0_norm, dtype=torch.float32)
        obs_tensor = torch.tensor(obs0_norm, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        with torch.no_grad():
            predict_diff = self.model(obs_tensor, action_tensor)
        intrinsic_r = (predict_diff - diff_obs).pow(2).mean(dim=-1).numpy().reshape(-1, 1)
        intrinsic_r = np.clip(intrinsic_r, 0, self.max_ir)
        return intrinsic_r


    def update(self, obs, actions, next_obs, update_times, add_noise=False):
        self.o_norm.update(obs)
        self.o_norm.update(next_obs)
        loss_list=[]
        for _ in range(update_times):
            obs_norm = self.o_norm.normalize(obs) 
            next_obs_norm = self.o_norm.normalize(next_obs)
            if add_noise: 
                obs_norm += np.random.normal(loc=0, scale=self.args.obs_noise, size=obs_norm.shape)
                next_obs_norm += np.random.normal(loc=0, scale=self.args.obs_noise, size=next_obs_norm.shape)
            obs_norm = torch.tensor(obs_norm, dtype=torch.float32)
            next_obs_norm = torch.tensor(next_obs_norm, dtype=torch.float32)
            action_tensor = torch.tensor(actions, dtype=torch.float32)
            predict_nextobs= self.model(obs_norm, action_tensor) + obs_norm
            loss = (predict_nextobs - next_obs_norm).pow(2).mean()
            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()
            loss_list.append(loss.item())
        return np.mean(np.array(loss_list))



class ForwardDynamicsEnsemble(object):
    def __init__(self) -> None:
        pass


    






if __name__=='__main__':
    env_param = {'obs':13, 'action':4, 'action_max':1}
    model = ForwardModel(env_param=env_param, hidden_dim=256)
    x= torch.rand(4, 13)
    y = torch.rand(4, 4)
    print(model(x, y))
