from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from scipy.special import entr
import threading
from abc import ABC, abstractmethod
from rl_modules.network.realnvp import RealNVP
from rl_modules.teachers.RND.rnd import RND
import torch
from torch.utils.data import TensorDataset, DataLoader
from mpi4py import MPI


class RawDensity(ABC):
    @abstractmethod
    def fit(self, n_samples):
        pass

    @abstractmethod
    def normalize_samples(self,samples):
        pass
    
    @abstractmethod
    def evaluate_log_density(self, samples):
        pass
    

    @abstractmethod
    def random_sample(self,batch_size):
        pass

    @abstractmethod
    def load(self, save_folder):
        pass

    @abstractmethod
    def save(self, save_folder):
        pass

    

class KernalDensityEstimator(RawDensity):
    def __init__(self, name, args, logger, buffer, eval_samples=5000, kernel='gaussian', bandwidth=0.1, normalize=True):
        self.step = 0
        self.args = args
        self.kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.normalize = normalize
        self.mean = 0.
        self.std = 1.
        self.fitted_model = None
        self.name = name
        self.logger = logger
        # self.size = buffer_size
        self.buffer = buffer
        self.module_name = kernel
        self.n_kde_samples = eval_samples
        self.kde_samples = None
        self.save_root = os.path.join(self.args.save_dir, self.args.env_name, self.args.alg, 'seed-'+str(self.args.seed))
        self.model_path = os.path.join(self.save_root, 'models')
        if MPI.COMM_WORLD.Get_rank() == 0:
            os.makedirs(self.model_path, exist_ok=True)
        self.save_fre =self.args.save_interval
        # self.lock = threading.Lock()

    def fit(self, n_samples=6000):
        self.kde_samples = self.random_sample(n_samples)
        # self.kde_samples = self.buffer.buffer['candidate_goals'][:self.buffer.current_size].copy()
        if self.normalize:
            self.mean = np.mean(self.kde_samples, axis=0, keepdims=True)
            self.std = np.std(self.kde_samples, axis=0, keepdims=True) + 1e-4
            self.kde_samples = (self.kde_samples - self.mean) / self.std

        self.fitted_model = self.kde.fit(self.kde_samples)
        # if MPI.COMM_WORLD.Get_rank() == 0:
        #     if self.step % self.save_fre == 0:
        #         self.save(self.model_path)
        self.step += 1

        # Scoring samples is a bit expensive, so just use 1000 points
        num_samples = self.n_kde_samples
        s = self.fitted_model.sample(num_samples)
        entropy = - self.fitted_model.score(s) / num_samples + np.log(self.std).sum()
        self.logger.add_scalar('{}_entropy'.format(self.module_name), entropy, self.step)
        
            #print('{}_entropy'.format(self.module_name), entropy, self.time_steps)

    def normalize_samples(self, samples):
        assert self.normalize
        return (samples - self.mean) / self.std

    def evaluate_log_density(self, samples):
        assert self.fitted_model is not None
        if self.normalize:
            samples = self.normalize_samples(samples)
        return self.fitted_model.score_samples(samples)

    def evaluate_elementwise_entropy(self, samples, beta=0.):
        if self.normalize:
            samples = self.normalize_samples(samples)
        log_px = self.fitted_model.score_samples(samples)
        px = np.exp(log_px)
        elem_entropy = entr(px + beta)
        return elem_entropy

    def save(self, save_folder):
        with open(os.path.join(save_folder, self.name + "_density_estimator_"+ str(self.step)+".pkl"), 'wb') as f:
            pickle.dump(self, f)


    def _save_props(self, prop_names, save_folder):
        prop_dict = {prop: self.__dict__[prop] for prop in prop_names}
        with open(os.path.join(save_folder, "density_module.pickle"), 'wb') as f:
            pickle.dump(prop_dict, f)

    def load(self, save_folder):
        with open(os.path.join(save_folder, self.name + "_density_estimator"+ str(self.step)+".pkl"), 'r') as f:
            loaded = pickle.load(f)
        for k, v in loaded.__dict__.items():
            self.__dict__[k] = v

    def random_sample(self, batch_size):
        idx = np.random.randint(self.buffer.current_size, size=batch_size)
        goals = self.buffer.buffer['candidate_goals'][idx].copy()
        return goals

   

    def clear_buffer(self):

        self.mean = 0.
        self.std = 1.
        self.buffer.clear_buffer()



class FlowDensity(RawDensity):
    def __init__(self, name, logger, buffer, lr=1e-3, num_layer_pairs=3, normalize=True, dev='cpu'):
        self.name = name
        self.logger = logger
        self.buffer = buffer
        self.normlize = normalize
        self.lr = lr
        self.num_layer_pairs = num_layer_pairs
        self.sample_mean = 0.
        self.sample_std = 1.
        self.fitted_model = None
        self.input_channel = 3
        self.device = dev

    def fit(self, n_samples=6000):
        # samples = self.buffer.buffer['candidate_goals'][:self.buffer.current_size].copy()
        samples = self.random_sample(n_samples)
        if self.normlize:
            self.sample_mean = np.mean(samples, axis=0, keepdims=True)
            self.sample_std = np.mean(samples, axis=0,keepdims=True) + 1e-4
            samples = (samples - self.sample_mean) / self.sample_std
        if self.fitted_model is None:
            self.init_flow_model(samples)
        
        samples = torch.tensor(samples, dtype=torch.float32).to(torch.device('cpu'))
        self.fitted_model.fit(samples, epochs=5)


    def init_flow_model(self, sample):
        # sample: [batch_size, dim]
        input_size = sample.shape[-1]
        self.input_channel = input_size
        self.fitted_model = RealNVP(input_channel=self.input_channel, lr=self.lr, num_layer_pairs=self.num_layer_pairs, dev=self.device)
    

    def normalize_samples(self, samples):
        assert self.normalize
        return (samples - self.sample_mean) / self.sample_std

    def random_sample(self, batch_size):
        idx = np.random.randint(self.buffer.current_size, size=batch_size)
        goals = self.buffer.buffer['candidate_goals'][idx].copy()
        return goals

    
    def evaluate_log_density(self, samples):
        assert self.fitted_model is not None
        if self.normlize:
            samples = self.normalize_samples(samples)
        samples = np.array(samples, dtype=np.float32)
        return self.fitted_model.score_samples(samples)


    def evaluate_elementwise_entropy(self, samples, beta=1e-3):
        if self.normalize:
            samples = self.normalize_samples(samples)
        log_px = self.fitted_model.score_samples(samples)
        px = np.exp(log_px)
        elem_entropy = entr(px + beta)
        return elem_entropy

    def save(self, save_folder):
        path = os.path.join(save_folder, self.name + '_flow_density.pt')
        if self.fitted_model is not None:
            torch.save({'flow_model': self.fitted_model}, path)
    
    def load(self, save_folder):
        path = os.path.join(save_folder, self.name + '_flow_density.pt')
        if os.path.exists(path):
            self.fitted_model = torch.load(path)

    def clear_buffer(self):

        self.mean = 0.
        self.std = 1.
        self.buffer.clear_buffer()


class RNDDensity(RawDensity):
    def __init__(self, name, logger, buffer, lr, eval_samples=5000, normalize=True):
        self.name = name
        self.logger = logger
        self.buffer = buffer
        self.eval_sampels = eval_samples
        self.normalize = normalize
     
        self.mean = 0.
        self.std = 1.
        self.obs_shape = 3
        self.fitted_model = RND(obs_shape=self.obs_shape, lr=lr,latent_dim=2)
        self.batchsize = 256
        

    def fit(self, n_samples=6000):
        samples = self.random_sample(n_samples)
        # samples = self.buffer.buffer['candidate_goals'][:self.buffer.current_size].copy()
        if self.normalize:
            self.sample_mean = np.mean(samples, axis=0, keepdims=True)
            self.sample_std = np.mean(samples, axis=0,keepdims=True) + 1e-4
            samples = (samples - self.sample_mean) / self.sample_std
        
        samples = torch.from_numpy(samples.astype(np.float32))
        dataset = TensorDataset(samples)
        loader = DataLoader(dataset=dataset, batch_size=self.batchsize)
        for idx, batch_data in enumerate(loader):
            # print(batch_data[0].size)
            self.fitted_model.update(batch_data[0])

 
    def normalize_samples(self,samples):
        assert self.normalize
        return (samples - self.sample_mean) / self.sample_std
    

    def evaluate_log_density(self, samples):
        
        if self.normalize:
            samples = self.normalize_samples(samples)
        scores = self.fitted_model.score_samples(samples)
        log_density = np.log(1-scores + 1e-5)

        return log_density
    

    def random_sample(self,batch_size):
        idx = np.random.randint(self.buffer.current_size, size=batch_size)
        goals = self.buffer.buffer['candidate_goals'][idx].copy()
        return goals


    def load(self, save_folder):
        path = os.path.join(save_folder, self.name + '_flow_density.pt')
        if os.path.exists(path):
            self.fitted_model = torch.load(path)


    def save(self, save_folder):
        path = os.path.join(save_folder, self.name + '_rnd_density.pt')
        if self.fitted_model is not None:
            torch.save({'rnd_model': self.fitted_model}, path)
    

    def clear_buffer(self):

        self.mean = 0.
        self.std = 1.
        self.buffer.clear_buffer()

