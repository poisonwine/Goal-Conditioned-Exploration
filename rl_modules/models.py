import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""

# define the actor network
class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions



class SacActor(nn.Module):
    def __init__(self, env_params, min_log_sigma=-20.0, max_log_sigma=2.0):
        super(SacActor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, env_params['action'])
        self.fc_sigma = nn.Linear(256, env_params['action'])
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma
        # self.action_out = nn.Linear(256, env_params['action'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.fc_mu(x)
        log_sigma = self.fc_sigma(x)
        log_sigma = torch.clamp(log_sigma, self.min_log_sigma, self.max_log_sigma)
        return mu, log_sigma

    def act(self, mu, log_sigma):
        sigma = torch.exp(log_sigma)
       
        dist = Normal(mu, sigma)
        # * reparameterization trick: recognize the difference of sample() and rsample()
        action = dist.rsample()
        tanh_action = torch.tanh(action)
        # * the log-probabilities of actions can be calculated in closed forms
        log_prob = dist.log_prob(action)
        log_prob = (log_prob - torch.log(1 - torch.tanh(action).pow(2))).sum(-1)
        return tanh_action, log_prob

class critic(nn.Module):
    def __init__(self, env_params):
        super(critic, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value

class V_critic(nn.Module):
    def __init__(self, env_params):
        super(V_critic, self).__init__()
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.v_out = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        v_value = self.v_out(x)
        
        return v_value

class BVNCritic(nn.Module):
    def __init__(self, env_params, latent_dim):
        super(BVNCritic, self).__init__()
        self.max_action = env_params['action_max']

        self.F = nn.Sequential(nn.Linear(env_params['obs']+env_params['action'], 176),
                               nn.ReLU(),
                               nn.Linear(176, 176),
                               nn.ReLU(),
                               nn.Linear(176, latent_dim))
        self.Phi = nn.Sequential(nn.Linear(env_params['obs']+env_params['goal'], 176),
                               nn.ReLU(),
                               nn.Linear(176, 176),
                               nn.ReLU(),
                               nn.Linear(176, latent_dim))
        self.goal_dim = env_params['goal']

    def forward(self, x, actions):
        g = x[:, -self.goal_dim:]
        s = x[:, :-self.goal_dim]
        F_in = torch.cat([s, actions / self.max_action], dim=1)
        Phi_in = torch.cat([s, g], dim=1)
        F_out = self.F(F_in).unsqueeze(-1)
        Phi_out = self.Phi(Phi_in).unsqueeze(-1)
        
        q_out = torch.matmul(torch.transpose(F_out, 1, 2), Phi_out)
        return q_out.squeeze(-1)

class MRNCritic(nn.Module):
    # Metric Residual Networks for Sample Efficient Goal-Conditioned Reinforcement Learning
    # https://github.com/Cranial-XIX/metric-residual-network
    #  Q = - (d_sym+d_asym)
    # d_sym = (phi(x)-phi(y)).pow
    # d_asym = max(h(x)-h(y))
    def __init__(self, env_params, emb_dim, hidden_dim):
        super(MRNCritic, self).__init__()

        self.max_action = env_params['action_max']
        self.embedding_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.goal_dim = env_params['goal']
        self.act_dim = env_params['action']
        self.obs_dim = env_params['obs']
        self.f_emb = nn.Sequential(nn.Linear(self.obs_dim + self.act_dim, self.hidden_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.hidden_dim, self.hidden_dim),
                               nn.ReLU(inplace=True))
        self.phi_emb = nn.Sequential(nn.Linear(self.obs_dim+self.act_dim + self.goal_dim, self.hidden_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.hidden_dim, self.hidden_dim),
                               nn.ReLU(inplace=True))
        self.sym = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.hidden_dim, self.embedding_dim))
        self.asym = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.hidden_dim, self.embedding_dim))


    def forward(self, x, actions):
        g = x[:, -self.goal_dim:]
        s = x[:, :-self.goal_dim]
        x1 = torch.cat([s, actions/self.max_action], dim=-1)
        x2 = torch.cat([s, actions / self.max_action, g], dim=-1)
        fh = self.f_emb(x1)
        phih= self.phi_emb(x2)

        sym1 = self.sym(fh)
        sym2 = self.sym(phih)
        asym1 = self.asym(fh)
        asym2 = self.asym(phih)
        dist_s = (sym1 - sym2).pow(2).mean(-1, keepdims=True)
        res = F.relu(asym1 - asym2)
        dist_a = res.max(-1)[0].view(-1, 1)
        q = - (dist_a+dist_s)
        return q
    
    def evaluate(self,x, actions):
        g = x[:, -self.goal_dim:]
        s = x[:, :-self.goal_dim]
        x1 = torch.cat([s, actions/self.max_action], dim=-1)
        x2 = torch.cat([s, actions / self.max_action, g], dim=-1)
        fh = self.f_emb(x1)
        phih= self.phi_emb(x2)

        sym1 = self.sym(fh)
        sym2 = self.sym(phih)
        asym1 = self.asym(fh)
        asym2 = self.asym(phih)
        dist_s = (sym1 - sym2).pow(2).mean(-1, keepdims=True)
        res = F.relu(asym1 - asym2)
        dist_a = res.max(-1)[0].view(-1, 1)
        q = - (dist_a+dist_s)
        return q, -dist_a, -dist_s
    




class ResidualBlock(nn.Module):
    """实现一个残差块"""
    def __init__(self,inchannel, outchannel, stride = 1,shortcut = None):

        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel,outchannel,3,stride,1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel,outchannel,3,1,1,bias=False), # 这个卷积操作是不会改变w h的
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, input):
        out = self.left(input)
        residual = input if self.right is None else self.right(input)
        out+=residual
        return F.relu(out)



class ResNet(nn.Module):
    """实现主reset"""

    def __init__(self, num_class=1000):
        super().__init__()
        # 前面几层普通卷积
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )

        # 重复layer，每个layer都包含多个残差块 其中第一个残差会修改w和c，其他的残差块等量变换
        # 经过第一个残差块后大小为 w-1/s +1 （每个残差块包括left和right，而left的k = 3 p = 1，right的shortcut k=1，p=0）
        self.layer1 = self._make_layer(64, 128, 3)  # s默认是1 ,所以经过layer1后只有channle变了
        self.layer2 = self._make_layer(128, 256, 4, stride=2)  # w-1/s +1
        self.layer3 = self._make_layer(256, 512, 6, stride=2)
        self.layer4 = self._make_layer(512, 512, 3, stride=2)
        self.fc = nn.Linear(512, num_class)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        # 之后的cahnnle同并且 w h也同，而经过ResidualBloc其w h不变，
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.pre(input)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)  # 如果图片大小为224 ，经过多个ResidualBlock到这里刚好为7，所以做一个池化，为1，
        # 所以如果图片大小小于224，都可以传入的，因为经过7的池化，肯定为1，但是大于224则不一定
        x = x.view(x.size(0), -1)
        return self.fc(x)



