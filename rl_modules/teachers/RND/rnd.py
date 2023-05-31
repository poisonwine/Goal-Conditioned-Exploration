import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch import optim

class MlpEncoder(nn.Module):
    def __init__(self, obs_shape, latent_dim):
        super(MlpEncoder, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(obs_shape, 128), 
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(),
            nn.Linear(128, latent_dim), 
            nn.LayerNorm(latent_dim))

    def forward(self, ob):
        x = self.main(ob)

        return x

class RND(object):
    def __init__(self, obs_shape, lr, latent_dim=5):
        self.obs_shape = obs_shape
  
        self.lr = lr
        self.predictor = MlpEncoder(obs_shape=self.obs_shape, latent_dim=latent_dim)
        self.target = MlpEncoder(obs_shape=self.obs_shape, latent_dim=latent_dim)

        self.opt  = optim.Adam(lr=self.lr, params=self.predictor.parameters())
        for p in self.target.parameters():
            p.requires_grad = False
    
    def score_samples(self, obs):
        # (batch, obs_shape)
        obs = np.array(obs).astype(np.float32)
        obs = torch.from_numpy(obs)

        with torch.no_grad():
            src_feats = self.predictor(obs)
            tar_feats = self.target(obs)
            dist = F.mse_loss(src_feats, tar_feats, reduction='none').mean(dim=-1)
            dist = (dist - dist.min())/ (dist.max()-dist.min()+1e-6)
        
        return  dist.cpu().numpy()
    
    def update(self, obs, times=3):
        
        for _ in range(times):
            src_feats = self.predictor.forward(obs)
            tar_feats = self.target.forward(obs)
            loss = F.mse_loss(src_feats, tar_feats)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

