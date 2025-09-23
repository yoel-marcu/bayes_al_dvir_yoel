import numpy as np
import pandas as pd
import torch
import pycls.datasets.utils as ds_utils
from al_utils import cosine_similarity

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

class RBFKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h=1.0, batch_size=512):
        norm = compute_norm(x1, x2, self.device, batch_size=batch_size)
        k = torch.exp(-1.0 * (norm / h) ** 2 / 2)
        return k

class ProbCoverWeighted:
    def __init__(self, cfg, lSet, uSet, budgetSize, delta=1):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.representation_model = self.cfg['DATASET']['REPRESENTATION_MODEL']
        self.all_features = ds_utils.load_features(self.ds_name, representation_model=self.representation_model, train=True,
                                                   normalize=True, project=self.cfg.ACTIVE_LEARNING.USE_COSINE_DIST)
        self.lSet = lSet
        self.uSet = uSet
        self.budgetSize = budgetSize
        self.delta = delta
        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        if isinstance(self.all_features, torch.Tensor):
            self.rel_features = self.all_features[self.relevant_indices]
        elif isinstance(self.all_features, np.ndarray):
            self.rel_features = torch.from_numpy(self.all_features[self.relevant_indices])
        else:
            raise NotImplementedError('Unknown type of features')
        # self.graph_df = self.construct_graph()

        print(f'ProbCoverWeighted | Start constructing similarity matrix using delta={self.delta}')
        self.kernel_fn = RBFKernel('cuda')
        self.kernel_all = self.kernel_fn.compute_kernel(
            self.rel_features, self.rel_features, self.delta,
            batch_size=1024).to('cuda')

    # def construct_graph(self, batch_size=500):
    #     """
    #     creates a directed graph where:
    #     x->y iff l2(x,y) < delta.
    #
    #     represented by a list of edges (a sparse matrix).
    #     stored in a dataframe
    #     """
    #     xs, ys, ds = [], [], []
    #     print(f'Start constructing graph using delta={self.delta}')
    #     # distance computations are done in GPU
    #     cuda_feats = torch.tensor(self.rel_features).cuda()
    #     for i in range(len(self.rel_features) // batch_size):
    #         # distance comparisons are done in batches to reduce memory consumption
    #         cur_feats = cuda_feats[i * batch_size: (i + 1) * batch_size]
    #         cur_feats = cuda_feats[i * batch_size: (i + 1) * batch_size]
    #         if self.cfg.ACTIVE_LEARNING.USE_COSINE_DIST:
    #             dist = (1 - cosine_similarity(cur_feats, cuda_feats))
    #         else:
    #             dist = torch.cdist(cur_feats, cuda_feats)
    #         mask = dist < self.delta
    #         # saving edges using indices list - saves memory.
    #         x, y = mask.nonzero().T
    #         xs.append(x.cpu() + batch_size * i)
    #         ys.append(y.cpu())
    #         ds.append(dist[mask].cpu())
    #
    #     xs = torch.cat(xs).numpy()
    #     ys = torch.cat(ys).numpy()
    #     ds = torch.cat(ds).numpy()
    #     weights = np.exp(-ds ** 2 / 2)
    #
    #     df = pd.DataFrame({'x': xs, 'y': ys, 'd': ds, 'weight': weights})
    #     print(f'Finished constructing graph using delta={self.delta}')
    #     print(f'Graph contains {len(df)} edges.')
    #     return df

    def select_samples(self):
        """
        selecting samples using the greedy algorithm.
        iteratively:
        - removes incoming edges to all covered samples
        - selects the sample high the highest out degree (covers most new samples)

        """
        print(f'Start selecting {self.budgetSize} samples.')
        selected = []
        # removing incoming edges to all covered samples from the existing labeled set
        # edge_from_seen = np.isin(self.graph_df.x, np.arange(len(self.lSet)))
        # covered_samples = self.graph_df.y[edge_from_seen].unique()
        # cur_df = self.graph_df[(~np.isin(self.graph_df.y, covered_samples))]
        curr_density = self.kernel_all.sum(dim=1)

        for i in range(self.budgetSize):
            curr_l_set = np.concatenate((np.arange(len(self.lSet)), selected)).astype(int)
            if len(curr_l_set) > 0:
                # coverage_per_sample = self.kernel_all[:, curr_l_set].sum(dim=1)
                coverage_per_sample = self.kernel_all[:, curr_l_set].max(dim=1).values
            else:
                coverage_per_sample = torch.zeros_like(curr_density)
            uncoverage_per_sample = torch.clamp_min(1 - coverage_per_sample, 0)
            curr_density *= uncoverage_per_sample

            # coverage = len(covered_samples) / len(self.relevant_indices)
            coverage = ((1 - uncoverage_per_sample) / torch.ones_like(curr_density)).mean()

            # selecting the sample with the highest degree
            # weighted_degrees = np.bincount(cur_df.x, minlength=len(self.relevant_indices), weights=cur_df.weights)
            curr_density[curr_l_set] = -1  # Exclude already labeled samples

            print(f'ProbCover Weighted | Iteration is {i}.\tMax degree is {curr_density.max()}.\tCoverage is {coverage:.3f}')

            # Selecting a sample with the highest weighted degree
            cur = curr_density.argmax().item()
            # max_degree = curr_density.max()
            # agrmax = np.where(curr_density == max_degree)[0]
            # cur = np.random.choice(agrmax)
            assert cur not in selected, 'sample was already selected'

            selected.append(cur)

        assert len(selected) == self.budgetSize, 'added a different number of samples'
        activeSet = self.relevant_indices[selected]
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))

        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')
        return activeSet, remainSet