import numpy as np
import torch
import copy
import time
from tqdm import tqdm
import torch.nn.functional as F
import pycls.datasets.utils as ds_utils
from tools.utils import visualize_tsne

# from pycls.utils.metrics import compute_coverage
# from pycls.utils.io import compute_cand_size
# from torch.utils.data import DataLoader


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


class NegNormKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h, batch_size=512):
        dist_matrix = compute_norm(x1, x2, self.device, batch_size=batch_size)
        return -dist_matrix

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

class RBFKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h=1.0, batch_size=512):
        norm = compute_norm(x1, x2, self.device, batch_size=batch_size)
        k = torch.exp(-1.0 * (norm / h) ** 2)
        return k

class StudentTKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h=1.0, batch_size=512, beta=0.5):
        norms = compute_norm(x1, x2, self.device, batch_size=batch_size)
        k = (1 + ((norms / h) ** 2) / beta) ** (-(beta+1)/2)
        return k

class LaplaceKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h=1.0, batch_size=512, beta=1):
        norms = compute_norm(x1, x2, self.device, batch_size=batch_size)
        k = torch.exp(-1 / h * (norms ** beta))
        return k

class CauchyKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h=1.0, batch_size=512):
        norms = compute_norm(x1, x2, self.device, batch_size=batch_size)
        k =  1 / (1 + norms**2)
        return k

class RationalQuadKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h=1.0, batch_size=512, alpha=1.0):
        norms = compute_norm(x1, x2, self.device, batch_size=batch_size)
        k = (1 + norms**2 / (2 * alpha))**(-alpha)
        return k


class MaxHerding:
    def __init__(self, cfg, lSet, uSet, budgetSize,
                 delta=1, kernel="rbf", device="cuda", batch_size=1024):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.device = device
        self.all_features = ds_utils.load_features(self.ds_name, train=True,
                                                   normalized=True)

        self.batch_size = batch_size
        self.lSet = lSet
        self.total_uSet = copy.deepcopy(uSet)
        self.budgetSize = budgetSize
        self.delta = delta
        print(f"MaxHerding | Using {kernel} kernel with sigma = {delta}")

        # subset_size = compute_cand_size(len(self.lSet), self.budgetSize) # 35000 if self.ds_name not in ['IMAGENET', 'IMBALANCED_IMAGENET'] else 40000
        subset_size = 35000 if self.ds_name not in ['IMAGENET', 'IMBALANCED_IMAGENET'] else 40000
        # print(f'Subset size: {subset_size}')
        # # if permute:
        # self.uSet = np.random.permutation(self.total_uSet)[:subset_size]
        # else:
        self.uSet = self.total_uSet

        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        if isinstance(self.all_features, torch.Tensor):
            self.relevant_features = self.all_features[self.relevant_indices]
        elif isinstance(self.all_features, np.ndarray):
            self.relevant_features = torch.from_numpy(self.all_features[self.relevant_indices])
        else:
            raise NotImplementedError('Unknown type of features')

        self.kernel_fn = self.construct_kernel_fn(kernel_name=kernel)
        self.activeSet = []


    def construct_kernel_fn(self, kernel_name):
        if kernel_name == "rbf":
            kernel = RBFKernel('cuda')
        elif kernel_name == "tophat":
            kernel = TopHatKernel('cuda')
        elif kernel_name == "student":
            kernel = StudentTKernel('cuda')
        elif kernel_name == 'negnorm':
            kernel = NegNormKernel('cuda')
        elif kernel_name == "laplace":
            kernel = LaplaceKernel('cuda')
        elif kernel_name == "cauchy":
            kernel = CauchyKernel('cuda')
        elif kernel_name == 'rational':
            kernel = RationalQuadKernel('cuda')
        else:
            raise NotImplementedError(f"{kernel_name} not implemented")
        print(f'Constructed kernel: {kernel_name}')
        return kernel

    def init_sampling_loop(self):
        self.kernel_all = self.kernel_fn.compute_kernel(
            self.relevant_features, self.relevant_features, self.delta,
            batch_size=self.batch_size).to(self.device)  # (l+u) x (l+u)
        print(f"Memory size of kernel: {self.kernel_all.element_size() * self.kernel_all.nelement()}")
        print(self.lSet)
        if len(self.lSet) > 0:
            self.kernel_la = self.kernel_fn.compute_kernel(
                self.relevant_features[:len(self.lSet)], self.relevant_features, self.delta,
                batch_size=self.batch_size).to(self.device)


    def select_samples(self):
        # uncertainties = torch.ones(1, len(self.relevant_indices)).float().to(self.device)

        self.init_sampling_loop()

        start_time = time.time()
        inner_lSet = torch.arange(len(self.lSet)).to(self.device)

        fixed_inner_uSet = torch.arange(len(self.relevant_indices))[len(inner_lSet):].to(self.device)
        inner_uSet_bool = torch.ones_like(fixed_inner_uSet).bool().to(self.device)
        inner_uSet = fixed_inner_uSet[inner_uSet_bool].to(self.device)

        if inner_lSet.shape[0] > 0:
            max_embedding = self.kernel_la.max(dim=0, keepdim=True).values # 1 x N
        else:
            max_embedding = torch.zeros(1, len(inner_lSet) + len(fixed_inner_uSet)).cpu() # 1 x N
            # max_embedding = torch.zeros(1, len(inner_lSet) + len(fixed_inner_uSet)).to(self.device) # 1 x N

        selected = []
        for i in tqdm(range(self.budgetSize), desc="MaxHerding | Selecting samples"):
            num_lSet = len(inner_lSet)
            num_uSet = len(inner_uSet)

            updated_max_embedding = (self.kernel_all.cpu() - max_embedding.cpu()) # N x N
            updated_max_embedding[updated_max_embedding < 0] = 0.

            # mean_max_embedding = (uncertainties * updated_max_embedding).mean(dim=-1) # N
            mean_max_embedding = updated_max_embedding.mean(dim=-1)  # N

            # select a point from u
            mean_max_embedding[inner_lSet.cpu()] = -np.inf
            selected_index = torch.argmax(mean_max_embedding)

            # update lSet and uSet
            inner_lSet = torch.cat((inner_lSet.cpu(), selected_index.view(-1)))
            inner_uSet_bool[selected_index - len(self.lSet)] = False
            inner_uSet = fixed_inner_uSet[inner_uSet_bool]

            max_embedding = updated_max_embedding[selected_index].unsqueeze(0) + max_embedding.cpu()

            if len(set(inner_lSet.cpu().numpy())) != num_lSet + 1:
                print(f'inner_lSet: {len(set(inner_lSet.numpy()))} is not equal to {num_lSet+1}')
                import IPython; IPython.embed()
            if len(set(inner_uSet.cpu().numpy())) != num_uSet - 1:
                print(f'inner_lSet: {len(set(inner_uSet.numpy()))} is not equal to {num_uSet+1}')
                import IPython; IPython.embed()
            assert len(np.intersect1d(inner_lSet.cpu().numpy(), inner_uSet.cpu().numpy())) == 0

            del updated_max_embedding, mean_max_embedding

        selected = inner_lSet[len(self.lSet):].cpu()

        # total_inner_lSet = torch.cat((torch.arange(len(self.lSet)), selected))
        # total_lSet_features = self.relevant_features[total_inner_lSet].to(self.device)
        # coverage = compute_coverage(total_lSet_features, self.relevant_features, self.kernel_fn)
        # print(f'Mean coverage herding: {coverage}')

        assert len(selected) == self.budgetSize, 'added a different number of samples'
        activeSet = self.relevant_indices[selected].reshape(-1)
        remainSet = np.array(sorted(list(set(self.total_uSet) - set(activeSet))))
        self.activeSet = activeSet
        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')
        print(f'Time: {np.round(time.time() - start_time, 4)}sec')

        return activeSet, remainSet

    def plot_tsne(self):
        labeled_indices = np.array(self.lSet).astype(int)
        sampled_indices = np.array(self.activeSet).astype(int)
        visualize_tsne(labeled_indices, sampled_indices, algo_name='Max Herding')