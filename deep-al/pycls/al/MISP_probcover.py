import numpy as np
import pandas as pd
import torch
import pycls.datasets.utils as ds_utils
from tools.utils import visualize_tsne
import matplotlib.pyplot as plt
###MISP = maximum importance sampling points


def compute_norm(x1, x2, device, batch_size=512):
    x1, x2 = x1.unsqueeze(0).to(device), x2.unsqueeze(0).to(device) # 1 x n x d, 1 x n' x d
    dist_matrix = []
    batch_round = x2.shape[1] // batch_size + int(x2.shape[1] % batch_size > 0)
    for i in range(batch_round):
        # distance comparisons are done in batches to reduce memory consumption
        x2_subset = x2[:, i * batch_size: (i + 1) * batch_size]
        dist = torch.cdist(x1, x2_subset, p=2.0)

        dist_matrix.append(dist.cpu())
        del dist

    dist_matrix = torch.cat(dist_matrix, dim=-1).squeeze(0)
    return dist_matrix

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
            dist_matrix.append(dist.cpu())

        dist_matrix = torch.cat(dist_matrix, dim=-1).squeeze(0)
        k = (dist_matrix < h)
        return k

class MISPC:
    def __init__(self, cfg, lSet, uSet, budgetSize, delta=1):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.all_features = ds_utils.load_features(self.ds_name, train=True)

        self.lSet = lSet
        self.uSet = uSet
        self.budgetSize = budgetSize
        self.delta = delta
        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        print(self.lSet)
        if isinstance(self.all_features, torch.Tensor):
            self.rel_features = self.all_features[self.relevant_indices]
        elif isinstance(self.all_features, np.ndarray):
            self.rel_features = torch.from_numpy(self.all_features[self.relevant_indices])
        else:
            raise NotImplementedError('Unknown type of features')

        print(f'ProbCoverWeighted | Start constructing similarity matrix using delta={self.delta}')

        # self.kernel_fn = RBFKernel('cuda')
        self.kernel_fn = TopHatKernel('cuda')
        self.K = self.kernel_fn.compute_kernel(
            self.rel_features, self.rel_features, self.delta,
            batch_size=1024).to('cuda')
        # print(f"max value in K is {self.K.max()} min is {self.K.min()}")
        self.labeled_mask = torch.ones(self.K.shape[0], dtype=torch.bool)
        self.labeled_mask[np.arange(self.lSet.size).astype(int)] = False

        covered_mask = torch.nonzero(self.K[~self.labeled_mask, :], as_tuple=False).flatten()
        self.labeled_mask[covered_mask] = False
        mask_matrix = torch.outer(self.labeled_mask, self.labeled_mask)
        self.K *= mask_matrix.to('cuda')



        self.K_point_wise = self.K.cpu().sum(dim=1).to(self.K.device)
        self.activeSet = []


    def select_samples(self, alpha=0.5):
        """
        selecting samples using the greedy algorithm.
        iteratively:
        - removes incoming edges to all covered samples
        - selects the sample high the highest out degree (covers most new samples)

        """
        print(f'Start selecting {self.budgetSize} samples.')
        selected = []

        V = torch.ones(self.K.shape[0], dtype=torch.float32, device=self.K.device)
        for i in range(self.budgetSize):
            K_num = self.K.cpu().float()  # cast bool → float
            points_total_importance = V.cpu().float() + alpha * K_num.matmul(V.cpu().float())
            sampled_point = points_total_importance.argmax().item()

            new_covered = torch.nonzero(self.K[sampled_point, :], as_tuple=False).flatten()
            self.K[:, new_covered] = False
            self.K[new_covered, :] = False

            self.K[:, sampled_point] = False
            self.K[sampled_point, :] = False

            assert sampled_point not in selected, 'sample was already selected'

            selected.append(sampled_point)

        assert len(selected) == self.budgetSize, 'added a different number of samples'
        activeSet = self.relevant_indices[selected]
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))
        self.activeSet = activeSet
        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')
        return activeSet, remainSet

    def plot_tsne(self):
        labeled_indices = np.array(self.lSet).astype(int)
        sampled_indices = np.array(self.activeSet).astype(int)
        visualize_tsne(labeled_indices, sampled_indices, algo_name='MISP')