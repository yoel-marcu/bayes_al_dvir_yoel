import numpy as np
import pandas as pd
import torch
import gc
import pycls.datasets.utils as ds_utils
from tools.utils import visualize_tsne
import matplotlib.pyplot as plt
###MISP = maximum importance sampling points
torch.cuda.empty_cache()

def compute_norm(x1, x2, device, batch_size=512):
    x1, x2 = x1.unsqueeze(0).to(device), x2.unsqueeze(0).to(device) # 1 x n x d, 1 x n' x d
    dist_matrix = []
    batch_round = x2.shape[1] // batch_size + int(x2.shape[1] % batch_size > 0)
    for i in range(batch_round):
        # distance comparisons are done in batches to reduce memory consumption
        x2_subset = x2[:, i * batch_size: (i + 1) * batch_size]
        dist = torch.cdist(x1, x2_subset, p=2.0) #.to(dtype=torch.float16)

        dist_matrix.append(dist.cpu())
        del dist

    dist_matrix = torch.cat(dist_matrix, dim=-1).squeeze(0)
    return dist_matrix

class RBFKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h=1.0, batch_size=512):
        norm = compute_norm(x1, x2, self.device, batch_size=batch_size)
        k = torch.exp(-1.0 * (norm / h) ** 2)
        return k

class TopHatKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h, batch_size=512):
        x1, x2 = x1.unsqueeze(0).to(self.device), x2.unsqueeze(0).to(self.device) # 1 x n x d, 1 x n' x d
        dist_matrix = []
        batch_round = x2.shape[1] // batch_size + int(x2.shape[1] % batch_size > 0)
        for i in range(batch_round):
            # distance comparisons are done in batches to reduce memory consumption
            x2_subset = x2[:, i * batch_size: (i + 1) * batch_size]
            dist = torch.cdist(x1, x2_subset)
            dist = (dist < h).to(dtype=torch.float32)
            dist_matrix.append(dist.cpu())
            del dist
        dist_matrix = torch.cat(dist_matrix, dim=-1).squeeze(0)
        # k = (dist_matrix < h).to(dtype=torch.float16)
        return dist_matrix


class SoftTopHatKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h, batch_size=512, slope=5):
        x1, x2 = x1.unsqueeze(0).to(self.device), x2.unsqueeze(0).to(self.device) # 1 x n x d, 1 x n' x d
        dist_matrix = []
        batch_round = x2.shape[1] // batch_size + int(x2.shape[1] % batch_size > 0)
        for i in range(batch_round):
            # distance comparisons are done in batches to reduce memory consumption
            x2_subset = x2[:, i * batch_size: (i + 1) * batch_size]
            dist = torch.cdist(x1, x2_subset)
            tophat_mask = (dist < h).to(dtype=torch.float16)
            soft_border_mask = ((dist >= h) & (dist < h + self.soft_border_val)).to(dtype=torch.float16)
            sig = 2 * torch.sigmoid(-slope * (dist - h) / self.soft_border_val)
            dist = tophat_mask + soft_border_mask * sig
            dist_matrix.append(dist.cpu())
            del dist, sig, soft_border_mask, tophat_mask
        dist_matrix = torch.cat(dist_matrix, dim=-1).squeeze(0)
        return dist_matrix

class MISP:
    def __init__(self, cfg, budgetSize, delta=1):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.all_features = ds_utils.load_features(self.ds_name, train=True)
        self.alpha = self.cfg.ALPHA
        self.debug = self.cfg.DEBUG
        self.norm_importance = self.cfg.NORM_IMPORTANCE
        self.own_alpha_weighting = self.cfg.OWN_ALPHA_WEIGHTING
        self.budgetSize = budgetSize
        self.delta = delta
        self.soft_border_val = self.cfg.SOFT_BORDER_VAL if 'SOFT_BORDER_VAL' in self.cfg else 0.15
        kernel_type = self.cfg.KERNEL_TYPE if 'KERNEL_TYPE' in self.cfg else 'rbf'
        if kernel_type == 'soft_tophat':
            self.kernel_fn = SoftTopHatKernel('cuda')
        elif kernel_type == 'tophat':
            self.kernel_fn = TopHatKernel('cuda')
        else:
            self.kernel_fn = RBFKernel('cuda')

        self.V_general = torch.ones(self.all_features.shape[0], device='cuda')
        torch.cuda.empty_cache()

    def init_sampling_loop(self,lset, uset):
        torch.cuda.empty_cache()
        self.set_rel_features(lset, uset)
        self.activeSet = []
        self.K = self.kernel_fn.compute_kernel(
            self.rel_features, self.rel_features, self.delta).to('cuda')
        self.V = self.V_general[self.relevant_indices].to('cuda')


    def set_rel_features(self, lset, uset):
        self.lSet = lset
        self.uSet = uset
        print(lset)
        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        if isinstance(self.all_features, torch.Tensor):
            self.rel_features = self.all_features[self.relevant_indices]
        elif isinstance(self.all_features, np.ndarray):
            self.rel_features = torch.from_numpy(self.all_features[self.relevant_indices])

    def select_samples(self, lset, uset):
        """
        selecting samples using the greedy algorithm.
        iteratively:
        - removes incoming edges to all covered samples
        - selects the sample high the highest out degree (covers most new samples)

        """
        self.init_sampling_loop(lset, uset)

        print(f'Start selecting {self.budgetSize} samples.')
        selected = []

        for i in range(self.budgetSize):
            curr_l_set = np.concatenate((np.arange(len(self.lSet)), selected)).astype(int)
            KV = batched_matmul(self.K, self.V)
            KV[curr_l_set] = -np.inf
            sampled_point = KV.argmax().item()

            self.V = torch.max(self.V * (1 - self.K[sampled_point, :]), torch.zeros_like(self.V))

            assert sampled_point not in selected, 'sample was already selected'

            selected.append(sampled_point)

        assert len(selected) == self.budgetSize, 'added a different number of samples'
        activeSet = self.relevant_indices[selected]

        self.V_general[self.relevant_indices] = self.V
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))
        self.activeSet = activeSet
        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')

        del self.K
        del self.V
        del KV

        return activeSet, remainSet

    def plot_tsne(self):
        labeled_indices = np.array(self.lSet).astype(int)
        sampled_indices = np.array(self.activeSet).astype(int)
        visualize_tsne(labeled_indices, sampled_indices, algo_name='MISP')

def batched_matmul(K, V, chunk_size=1024):
    D, N = K.shape
    result = torch.empty_like(V)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        result[start:end] = K[start:end] @ V
    return result