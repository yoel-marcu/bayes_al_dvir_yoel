import numpy as np
import torch
import pycls.datasets.utils as ds_utils
from tools.utils import visualize_tsne
import time
torch.cuda.empty_cache()

## S: For suggestions

# TODO's:
# 1. Remove assertions
# 2. Understand batched_diffs, batched_diffs_weighted & implement better batched_diffs_weighted 

### Possible changes (Software design XD):
# 1. C matrix induced by lset: move from init to a separate function.
# 2. Refactor select_samples to smaller functions.


def compute_norm(x1: torch.Tensor, x2: torch.Tensor, device, batch_size=512, pin_cpu=True):
    """
    Compute Gram matrix in batches to reduce GPU memory consumption.
    """
    x1, x2 = x1.to(device), x2.to(device)
    n, m = x1.shape[0], x2.shape[0]
    # Allocate the final matrix on CPU and pin it so not uneccessary transfers are done:
    dist_cpu = torch.empty(size=(n,m), device='cpu', pin_memory=pin_cpu)

    # Compute the distances where x2 is batched:
    for start in range(0, m, batch_size):
        end = min((start+batch_size), m)
        dist = torch.cdist(x1, x2[start:end,:])
        dist_cpu[:, start:end].copy_(dist, non_blocking=True) # Copy batch from GPU to pre-allocated CPu memory. Use DMA for efficiency.
        del dist

    dist_matrix = torch.cat(dist_matrix, dim=-1).squeeze(0)
    return dist_matrix


class RBFKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h=1.0, batch_size=512):
        """
        Compute the RBF (Radial Basis Function) kernel between two sets of data points.
        
        This method calculates the Gaussian kernel (RBF kernel) using the formula:
        k(x1, x2) = exp(-(||x1 - x2||^2 / h^2))
    
        Returns:
            torch.Tensor: Kernel matrix of shape (n1, n2) containing pairwise 
                          kernel values between points in x1 and x2.
                          The kernel values are in the range (0, 1]
        """
        norm = compute_norm(x1, x2, self.device, batch_size=batch_size)
        k = torch.exp(-1.0 * (norm / h) ** 2)
        return k


class TopHatKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h, batch_size=512):
        x1, x2 = x1.unsqueeze(0).to(self.device), x2.unsqueeze(0).to(self.device)
        dist_matrix = []
        batch_round = x2.shape[1] // batch_size + int(x2.shape[1] % batch_size > 0)
        for i in range(batch_round):
            # distance comparisons are done in batches to reduce memory consumption
            x2_subset = x2[:, i * batch_size: (i + 1) * batch_size]
            dist = torch.cdist(x1, x2_subset)
            dist = (dist < h).to(dtype=torch.float16)
            dist_matrix.append(dist.cpu())
            del dist
            
        dist_matrix = torch.cat(dist_matrix, dim=-1).squeeze(0)
        return dist_matrix


class BAYES_MISP:
    
    """
    Implementation of Dirichlet based AL algorithm.
    """
    
    def __init__(self, cfg, budgetSize, train_labels, lset, delta=1):
        # general settings
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.debug = self.cfg.DEBUG
        self.budgetSize = budgetSize
        # {'margin', 'max', no entropy and some other functions for some reason(?)}
        self.diff_method = self.cfg.DIFF_METHOD if 'DIFF_METHOD' in self.cfg else 'abs_diff' # Peakedness measure
        
        # kernel settings
        self.delta = delta # Kernel bandwidth
        kernel_type = self.cfg.KERNEL_TYPE if 'KERNEL_TYPE' in self.cfg else 'rbf'
        if kernel_type == 'tophat':
            self.kernel_fn = TopHatKernel('cuda')
        else:
            self.kernel_fn = RBFKernel('cuda')

        # features & labels
        self.all_features = ds_utils.load_features(self.ds_name, train=True) # (datapoints, features)
        self.train_labels_general = np.array(train_labels) # Labels for oracle querying
        unique_labels = np.unique(self.train_labels_general) # Possible labels in the entire dataset
        self.num_of_classes = np.unique(self.train_labels_general).size # Number of classes 'M'
        self.chosen_labels_count = torch.zeros(self.num_of_classes).to('cuda') # Counting how many samples were chosen from each class
        
        # C matrix
        self.alpha = self.cfg.ALPHA # Init prior
        # TODO: Transpose for consistency with theoretical formulation?
        self.C_general = torch.full((self.all_features.shape[0], unique_labels.size), self.alpha, device='cuda', dtype=torch.float16) # Initialise C when L = {}

        # C matrix induced by lset. TODO: Understand logic - is it the same as we know?
        if lset is not None and lset.size > 0:
            # Compute kernel on entire dataset:
            temp_K = self.kernel_fn.compute_kernel(
                torch.from_numpy(self.all_features), torch.from_numpy(self.all_features), self.delta).to('cuda')
            # For each class, which indices already belong to it.
            class_indices = {label: np.where(self.train_labels_general[lset.astype(int)] == label)[0] for label in unique_labels}

            for label in unique_labels:
                curr_labels_sim = temp_K[class_indices[label]]
                self.C_general[:, label] = torch.max(curr_labels_sim, axis=0).values
            del temp_K, curr_labels_sim, class_indices
        torch.cuda.empty_cache()


    def init_sampling_loop(self, lset, uset):
        """
        Initialize variables for the sampling loop.
        """
        torch.cuda.empty_cache()
        self.set_rel_features(lset, uset)
        self.active_set = []
        self.K = self.kernel_fn.compute_kernel(
            self.rel_features, self.rel_features, self.delta).to('cuda')
        # order C matrix & train_labels to have labeled samples first, then unlabeled samples
        # C contains the reordered indices but C_general is in the original state
        self.C = self.C_general[self.relevant_indices].to('cuda')
        self.train_labels = self.train_labels_general[self.relevant_indices]


    def set_rel_features(self, lset, uset):
        """
        Reorder X to have labeled samples first, then unlabeled samples.
        """
        self.lSet = lset # Indices
        self.uSet = uset
        
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
        selected_idxs = []
        for i in range(self.budgetSize):

            # Update local labeled set indices:
            curr_l_set = np.concatenate((np.arange(len(self.lSet)), selected_idxs)).astype(int) ## S: Can just append instead

            # Normalise the C matrix in order to get expected distribution.
            C_sum = torch.sum(self.C, dim=1, keepdim=True)
            norm_C = self.C / C_sum

            # Compute peakedness score:
            if self.diff_method == 'margin':
                vals, inds = torch.topk(self.C, k=2, dim=1) # Using self.C instead of norm_C as in 'max'(?)
                old_margin = vals[:, 0] - vals[:, 1]
                point_total_contribution = batched_diffs(self.K, old_margin, self.alpha, self.num_of_classes, diff_method="margin")
                
            elif self.diff_method == 'max':
                max_vals, indices = torch.max(norm_C, dim=1)
                point_total_contribution = batched_diffs(self.K, max_vals, self.alpha, self.num_of_classes, diff_method="max")

            elif self.diff_method == 'weighted_max':
                start = time.time()
                point_total_contribution = batched_diffs_weighted(self.K, self.C, self.alpha, self.num_of_classes, diff_method="weighted_max")
                print(time.time() - start)
            else:
                point_total_contribution = batched_diffs(self.K, self.C, diff_method=self.diff_method)
            
            # TODO: In Batched_diffs watch what happens to C in rows (~ theoretically cols) of labeled samples (negative control??)
            point_total_contribution[curr_l_set] = -np.inf # Prevent re-sampling already labeled samples
            sampled_point_idx = point_total_contribution.argmax().item()
            chosen_label = self.train_labels[sampled_point_idx].item() # get label from oracle

            self.chosen_labels_count[chosen_label] += 1
            self.C[:, chosen_label] += self.K[sampled_point_idx] # Update rule for C

            assert sampled_point_idx not in selected_idxs, 'sample was already selected'
            selected_idxs.append(sampled_point_idx)

        assert len(selected_idxs) == self.budgetSize, 'added a different number of samples'
        active_set_indices = self.relevant_indices[selected_idxs] # original indices of selected samples (relevant_indices[new_idx] = original idx)

        # Update C_general to reflect changes made to C considering the different ordering of samples
        self.C_general[self.relevant_indices] = self.C
        remaining_unlabeled_idxs = np.array(sorted(list(set(self.uSet) - set(active_set_indices))))
        
        self.active_set_indices = active_set_indices
        print(f'Finished the selection of {len(active_set_indices)} samples.')
        print(f'Active set indices are {active_set_indices}')

        del self.K
        del self.C

        return active_set_indices, remaining_unlabeled_idxs 


    def plot_tsne(self):
        labeled_indices = np.array(self.lSet).astype(int)
        sampled_indices = np.array(self.active_set_idxs).astype(int)
        visualize_tsne(labeled_indices, sampled_indices, algo_name='MISP')


def batched_diffs(K, C, alpha, number_of_classes, chunk_size=1024, diff_method="abs_diff"):
    D, N = K.shape # K is general. In our setting D = N
    result = torch.empty(D).to(device=C.device)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        if diff_method == "abs_diff":
            result[start:end] = torch.sum(torch.maximum(K[start:end] - C, torch.zeros_like(K[start:end]).to(device=C.device)), dim=1)
        elif diff_method == "max":
            result[start:end] = torch.sum(
                torch.maximum(((K[start:end] + alpha) / (K[start:end] + alpha * number_of_classes)) - C, torch.zeros_like(K[start:end]).to(device=C.device)), dim=1)
        elif diff_method == 'margin':
            result[start:end] = torch.sum(
                torch.maximum((K[start:end] / (K[start:end] + alpha * number_of_classes)) - C, torch.zeros_like(K[start:end]).to(device=C.device)), dim=1)
        else:
            raise ValueError(f"Unknown diff method: {diff_method}")
    return result


def batched_diffs_weighted(K, C, alpha, number_of_classes, chunk_size=32, diff_method="abs_diff"):
    D, N = K.shape
    result = torch.empty(D).to(device=C.device)
    max_C = torch.max(C, axis=1, keepdim=True).values
    sum_C = torch.sum(C, axis=1, keepdim=True)
    old_max = (max_C.squeeze() / sum_C.squeeze()).unsqueeze(1)
    norm_C = (C / sum_C).unsqueeze(1)
    num_iterations = N // 100
    K = K.unsqueeze(2)
    # timing each iteration
    for i in range(0, num_iterations, chunk_size):
        if diff_method == "weighted_max":
            end = i + chunk_size
            K_batched = K[i:end]
            future_sum = K_batched + sum_C
            modified_C = C + K[i:end]
            new_maxes = torch.maximum(max_C, modified_C)
            f_vecs = new_maxes / future_sum

            point_diff = f_vecs - old_max
            point_diff.clamp_(min=0)
            weighted_point_diff = torch.bmm(norm_C[i:end], point_diff.permute(0, 2, 1))
            result[i:end] = torch.nansum(weighted_point_diff, dim=2)[:,0]
        else:
            raise ValueError(f"Unknown diff method: {diff_method}")
    return result