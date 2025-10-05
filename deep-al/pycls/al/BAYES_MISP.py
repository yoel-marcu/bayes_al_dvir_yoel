import numpy as np
import torch
import pycls.datasets.utils as ds_utils
from tools.utils import visualize_tsne
import time
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
            dist = (dist < h).to(dtype=torch.float16)
            dist_matrix.append(dist.cpu())
            del dist
        dist_matrix = torch.cat(dist_matrix, dim=-1).squeeze(0)
        # k = (dist_matrix < h).to(dtype=torch.float16)
        return dist_matrix


class BAYES_MISP:
    def __init__(self, cfg, budgetSize, train_labels, lset, delta=1):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.all_features = ds_utils.load_features(self.ds_name, train=True)
        self.alpha = self.cfg.ALPHA
        self.debug = self.cfg.DEBUG
        self.confidence_method = self.cfg.CONFIDENCE_METHOD if 'CONFIDENCE_METHOD' in self.cfg else 'max'
        self.budgetSize = budgetSize
        self.delta = delta
        self.diff_method = self.cfg.DIFF_METHOD if 'DIFF_METHOD' in self.cfg else 'abs_diff'
        kernel_type = self.cfg.KERNEL_TYPE if 'KERNEL_TYPE' in self.cfg else 'rbf'
        if kernel_type == 'tophat':
            self.kernel_fn = TopHatKernel('cuda')
        else:
            self.kernel_fn = RBFKernel('cuda')

        self.train_labels_general = np.array(train_labels)
        unique_labels = np.unique(self.train_labels_general)
        # TODO: Transpose?
        self.C_general = torch.full((self.all_features.shape[0], unique_labels.size), self.alpha, device='cuda', dtype=torch.float16) # Initialise the C matrix with constant alpha
        self.num_of_classes = np.unique(self.train_labels_general).size
        self.chosen_labels_num = torch.zeros(self.num_of_classes).to('cuda')

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

    def init_sampling_loop(self,lset, uset):
        torch.cuda.empty_cache()
        self.set_rel_features(lset, uset)
        self.activeSet = []
        self.K = self.kernel_fn.compute_kernel(
            self.rel_features, self.rel_features, self.delta).to('cuda')
        self.C = self.C_general[self.relevant_indices].to('cuda')
        self.train_labels = self.train_labels_general[self.relevant_indices]

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

            # Update local labeled set:
            curr_l_set = np.concatenate((np.arange(len(self.lSet)), selected)).astype(int)

            # Normalise the C matrix in order to get expected distribution.
            C_sum = torch.sum(self.C, dim=1, keepdim=True)
            norm_C = self.C / C_sum

            # Compute peakedness score:
            if self.diff_method == 'margin':
                vals, inds = torch.topk(self.C, k=2, dim=1)

                old_margin = vals[:, 0] - vals[:, 1]

                point_total_contribution = batched_diffs(self.K, old_margin, self.alpha, self.num_of_classes, diff_method="margin")
            elif self.diff_method == 'max':
                max_vals, indices = torch.max(norm_C, dim=1)
                point_total_contribution = batched_diffs(self.K, max_vals, self.alpha, self.num_of_classes, diff_method="max")

            elif self.diff_method == 'weighted_max':
                # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                #     with record_function("model_inference"):
                #         point_total_contribution = batched_diffs_weighted(self.K, self.C, self.alpha, self.num_of_classes, diff_method="weighted_max")
                start = time.time()
                point_total_contribution = batched_diffs_weighted(self.K, self.C, self.alpha, self.num_of_classes, diff_method="weighted_max")
                print(time.time() - start)
                # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            else:
                point_total_contribution = batched_diffs(self.K, self.C, diff_method=self.diff_method)
            point_total_contribution[curr_l_set] = -np.inf # TODO: In Batched_diffs watch what happens to C in rows (~ theoretically cols) of labeled samples (negative control??)
            sampled_point = point_total_contribution.argmax().item()
            chosen_label = self.train_labels[sampled_point].item()

            self.chosen_labels_num[chosen_label] += 1

            self.C[:, chosen_label] += self.K[sampled_point]

            assert sampled_point not in selected, 'sample was already selected'
            selected.append(sampled_point)



        if False:
            name = "prob_method_v1"
            np.save(f"/cs/labs/daphna/itai.david/py_repos/TypiClust/vectors_debug/0708/{name}.npy", self.K[selected].cpu())

        assert len(selected) == self.budgetSize, 'added a different number of samples'
        activeSet = self.relevant_indices[selected]

        self.C_general[self.relevant_indices] = self.C
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))
        self.activeSet = activeSet
        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')

        del self.K
        del self.C

        return activeSet, remainSet

    def plot_tsne(self):
        labeled_indices = np.array(self.lSet).astype(int)
        sampled_indices = np.array(self.activeSet).astype(int)
        visualize_tsne(labeled_indices, sampled_indices, algo_name='MISP')

def batched_diffs(K, C, alpha, number_of_classes, chunk_size=1024, diff_method="abs_diff"):
    D, N = K.shape # Shouldn't K be (N, N)??
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
# @torch.compile(backend="cudagraphs")
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