import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np



def wasserstein_reward(d: torch.Tensor) -> torch.Tensor:
    """
    return the wasserstein reward
    """
    return d


def gail_reward(d: torch.Tensor) -> torch.Tensor:
    """
    Take discriminaotr output and return the gail reward
    :param d:
    :return:
    """
    d = torch.sigmoid(d)
    return d.log()  # - (1 - d).log()


def airl_reward(d: torch.Tensor) -> torch.Tensor:
    """
    Take discriminaotr output and return AIRL reward
    :param d:
    :return:
    """
    s = torch.sigmoid(d)
    reward = s.log() - (1 - s).log()
    return reward


def fairl_reward(d: torch.Tensor) -> torch.Tensor:
    """
    Take discriminator output and return FAIRL reward
    :param d:
    :return:
    """

    d = torch.sigmoid(d)
    h = d.log() - (1 - d).log()
    h = torch.clamp(h, -10., 10.)
    return h.exp() * (-h)

reward_mapping = {'aim': wasserstein_reward,
                  'gail': gail_reward,
                  'airl': airl_reward,
                  'fairl': fairl_reward}


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)        
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class MlpNetwork(nn.Module):
    """
    Basic feedforward network uesd as building block of more complex policies
    """
    def __init__(self, input_dim, output_dim=1, activ=F.relu, output_nonlinearity=None, n_units=64, tanh_constant=1.0):
        super(MlpNetwork, self).__init__()
        
        self.h1 = nn.Linear(input_dim, n_units)
        self.h2 = nn.Linear(n_units, n_units)
        # self.h3 = nn.Linear(n_units, n_units)
        self.out = nn.Linear(n_units, output_dim)
        self.out_nl = output_nonlinearity
        self.activ = activ
        self.tanh_constant = tanh_constant
        self.apply(weight_init)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass of network
        :param x:
        :return:
        """
        x = self.activ(self.h1(x))
        x = self.activ(self.h2(x))
        # x = self.activ(self.h3(x))
        x = self.out(x)
        if self.out_nl is not None:
            if self.out_nl == F.log_softmax:
                x = F.log_softmax(x, dim=-1)
            else:
                if self.out_nl==torch.tanh:                    
                    x = self.out_nl(self.tanh_constant*x)
                else:
                    x = self.out_nl(x)
        return x




class Discriminator(nn.Module):
    def __init__(self, x_dim=1, reward_type='aim', lr = 1e-4, lipschitz_constant=0.1, output_activation= None, device = 'cpu', tanh_constant = 1.0, lambda_coef = 10.0, adam_eps=1e-8, optim='adam'):
        self.use_cuda = False
        self.device = device # torch.device("cuda" if self.use_cuda else "cpu")
        
        self.adam_eps = adam_eps
        self.optim = optim        
        super(Discriminator, self).__init__()
        self.input_dim = x_dim
        assert reward_type in ['aim', 'gail', 'airl', 'fairl']
        self.reward_type = reward_mapping[reward_type]
        if self.reward_type == 'aim':
            self.d = MlpNetwork(self.input_dim, n_units=64)  # , activ=f.tanh)
        else:
            if output_activation is None:
                self.d = MlpNetwork(self.input_dim, n_units=64, activ=torch.tanh)
            elif output_activation=='tanh':
                self.d = MlpNetwork(self.input_dim, n_units=64, activ=torch.relu, output_nonlinearity=torch.tanh, tanh_constant = tanh_constant)
            
        self.d.to(self.device)
        self.lr = lr
        if optim=='adam':
            self.discriminator_optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=adam_eps)
        elif optim=='sparse_adam':
            self.discriminator_optimizer = torch.optim.SparseAdam(self.parameters(), lr=lr)
        elif optim=='rmsprop':
            self.discriminator_optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
        elif optim=='sgd':
            self.discriminator_optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        elif optim=='adamw':
            self.discriminator_optimizer = torch.optim.AdamW(self.parameters(), lr=lr, eps=adam_eps)
        self.lipschitz_constant = lipschitz_constant 
        # self.env_name = env_name
        self.lambda_coef = lambda_coef
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.d(x)
        return output

    def reward(self, x: torch.Tensor) -> np.ndarray:
        """
        return the reward
        """
        
        r = self.forward(x)
        if self.reward_type is not None:
            r = self.reward_type(r)
        return r.cpu().detach().numpy()

    def compute_graph_pen(self,
                          prev_state: torch.Tensor,
                          next_state_state: torch.Tensor):
        """
        Computes values of the discriminator at different points
        and constraints the difference to be 0.1
        """
        if self.use_cuda:
            prev_state = prev_state.cuda()
            next_state_state = next_state_state.cuda()
            zero = torch.zeros(size=[int(next_state_state.size(0))]).cuda()
        else:
            zero = torch.zeros(size=[int(next_state_state.size(0))])
        prev_out = self(prev_state)
        next_out = self(next_state_state)
        penalty = self.lambda_coef * torch.max(torch.abs(next_out - prev_out) - self.lipschitz_constant, zero).pow(2).mean()
        return penalty

    def compute_grad_pen(self,
                         target_state: torch.Tensor,
                         policy_state: torch.Tensor,
                         lambda_=10.):
        """
        Computes the gradients by mixing the data randomly
        and creates a loss for the magnitude of the gradients.
        """
        if self.use_cuda:
            target_state = target_state.cuda()
            policy_state = policy_state.cuda()
        alpha = torch.rand(target_state.size(0), 1)
        # expert_data = torch.cat([expert_state, expert_action], dim=1)
        # policy_data = torch.cat([policy_state, policy_action], dim=1)

        alpha = alpha.expand_as(target_state).to(target_state.device)

        mixup_data = alpha * target_state + (1 - alpha) * policy_state
        mixup_data.requires_grad = True

        disc = self(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = torch.autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen


    def _get_repeated_network_outputs(self, network, pure_obs, goals):
        # obs : [bs, dim]
        # goals : [bs*n_repeat, dim]
        goal_shape = goals.shape[0] 
        pure_obs_shape = pure_obs.shape[0] 
        num_repeat = int (goal_shape / pure_obs_shape)        
        
        # [bs, num_goal, dim] -> [bs*num_goal, dim]
        pure_obs_temp = pure_obs.unsqueeze(1).repeat(1, num_repeat, 1).view(pure_obs.shape[0] * num_repeat, pure_obs.shape[1])
        preds = network(torch.cat([pure_obs_temp, goals], dim = -1))
        preds = preds.view(pure_obs.shape[0], num_repeat, -1) # [bs*num_goal, dim] -> [bs, num_goal, dim]
        return preds

    def optimize_discriminator(self, target_states, policy_states, policy_next_states):
        """
        Optimize the discriminator based on the memory and
        target_distribution
        :return:
        """
        self.discriminator_optimizer.zero_grad()
        
        ones = target_states # [bs, dim([ag,dg])] #[g,g]
        zeros = policy_next_states # [bs, dim([ag,dg])] #[s',g]
        zeros_prev = policy_states # [bs, dim([ag,dg])] #[s,g]

        pred_ones = self(ones)
        pred_zeros = self(zeros)
        graph_penalty = self.compute_graph_pen(zeros_prev, zeros)
        min_aim_f_loss = None
        wgan_loss = torch.mean(pred_zeros) + torch.mean(pred_ones * (-1.))                
        loss = wgan_loss + graph_penalty

        loss.backward()
        self.discriminator_optimizer.step()
        return loss.item(), wgan_loss.item(), graph_penalty.item(), min_aim_f_loss


class DiscriminatorEnsemble(nn.Module):
    def __init__(self, n_ensemble, x_dim=1, reward_type='aim', lr = 1e-4, lipschitz_constant=0.1, output_activation= None, device = 'cpu', tanh_constant = 1.0, lambda_coef = 10.0, adam_eps=1e-8, optim = 'adam'):
        super().__init__()
        self.n_ensemble = n_ensemble
        self.adam_eps = adam_eps
        self.optim = optim
        self.discriminator_ensemble = nn.ModuleList([Discriminator(x_dim, reward_type, lr, lipschitz_constant, output_activation, device, tanh_constant, lambda_coef, adam_eps, optim) for i in range(n_ensemble)])
                            
        self.apply(weight_init)

    def forward(self, inputs):
        h = inputs
        outputs = torch.stack([discriminator(h) for discriminator in self.discriminator_ensemble], dim = 1) #[bs, n_ensemble, dim(1)]
        outputs = torch.mean(outputs, dim = 1)  #[bs, 1]
        return outputs

    def std(self,inputs):
        aim_outputs = torch.stack(self.forward(inputs), dim = 1)  # [bs, n_ensemble, 1]
        return torch.std(aim_outputs, dim = 1, keepdim=False) #[bs, 1]

    def reward(self, x: torch.Tensor) -> np.ndarray:
        return np.stack([discriminator.reward(x) for discriminator in self.discriminator_ensemble], axis = 1).mean(axis=1)

    def optimize_discriminator(self, *args, **kwargs):        
        loss_list = []
        wgan_loss_list = []
        graph_penalty_list = []
        # min_aim_f_loss_list = []

        for discriminator in self.discriminator_ensemble:
            loss, wgan_loss, graph_penalty, min_aim_f_loss = discriminator.optimize_discriminator(*args, **kwargs)
            loss_list.append(loss)
            wgan_loss_list.append(wgan_loss)
            graph_penalty_list.append(graph_penalty)
            # min_aim_f_loss_list.append(min_aim_f_loss)
        return torch.stack(loss_list, dim = 0).mean(0), torch.stack(wgan_loss_list, dim = 0).mean(0), torch.stack(graph_penalty_list, dim = 0).mean(0), None
    
    
