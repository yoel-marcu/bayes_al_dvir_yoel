import numpy as np
import torch
import pycls.datasets.utils as ds_utils
from tools.utils import visualize_tsne
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

class RBFKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h=1.0, batch_size=512):
        norm = compute_norm(x1, x2, self.device, batch_size=batch_size)
        k = torch.exp(-1.0 * (norm / h) ** 2 / 2)
        return k

class MISP_PLUS:
    def __init__(self, cfg, lSet, uSet, budgetSize, train_data, delta=1):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.all_features = ds_utils.load_features(self.ds_name, train=True)
        self.alpha = self.cfg.ALPHA
        self.debug = self.cfg.DEBUG
        self.norm_importance = self.cfg.NORM_IMPORTANCE
        self.own_alpha_weighting = self.cfg.OWN_ALPHA_WEIGHTING
        self.lSet = lSet
        self.uSet = uSet
        self.budgetSize = budgetSize
        self.delta = delta
        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        self.labels = np.array(train_data.targets)[ self.relevant_indices]
        print(self.lSet)
        if isinstance(self.all_features, torch.Tensor):
            self.rel_features = self.all_features[self.relevant_indices]
        elif isinstance(self.all_features, np.ndarray):
            self.rel_features = torch.from_numpy(self.all_features[self.relevant_indices])
        else:
            raise NotImplementedError('Unknown type of features')

        print(f'ProbCoverWeighted | Start constructing similarity matrix using delta={self.delta}')

        self.kernel_fn = RBFKernel('cuda')
        self.K = self.kernel_fn.compute_kernel(
            self.rel_features, self.rel_features, self.delta,
            batch_size=1024).to('cuda')
        self.K_point_wise = self.K.sum(dim=1)
        self.K.fill_diagonal_(0)
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


        for i in range(self.budgetSize):
            curr_l_set = np.concatenate((np.arange(len(self.lSet)), selected)).astype(int)
            current_labels = self.labels[curr_l_set]

            mean_distance_to_each_label = {}
            if len(curr_l_set) > 0:
                unique_labels_in_selection = np.unique(current_labels)
                for label_val in unique_labels_in_selection:
                    indices_for_this_label = curr_l_set[current_labels == label_val]
                    if len(indices_for_this_label) > 0:
                        similarities_to_this_label_group = self.K[:, indices_for_this_label]
                        mean_similarity_of_all_points_to_this_label = similarities_to_this_label_group.mean(
                            dim=1)

                        mean_distance_to_each_label[int(label_val)] = mean_similarity_of_all_points_to_this_label

                if len(mean_distance_to_each_label) < 2:
                    distance_between_top2_mean_similarities = list(mean_distance_to_each_label.values())[0]
                else:
                    sorted_labels = sorted(mean_distance_to_each_label.keys())

                    combined_mean_similarities = torch.stack([
                        mean_distance_to_each_label[label] for label in sorted_labels
                    ], dim=1)

                    top_2_mean_similarities, _ = torch.topk(combined_mean_similarities, k=2, dim=1, largest=True,
                                                            sorted=True)

                    closest_mean_similarity = top_2_mean_similarities[:, 0]
                    second_closest_mean_similarity = top_2_mean_similarities[:, 1]

                    distance_between_top2_mean_similarities = closest_mean_similarity - second_closest_mean_similarity

                V = distance_between_top2_mean_similarities
            else:
                V = torch.ones_like(self.K_point_wise)

            torch.cuda.empty_cache()

            if self.debug:
                points_total_importance = V.cpu() + self.alpha * (self.K.cpu() @ V.cpu() / self.K.shape[0])
            else:
                # points_total_importance = V + self.alpha * self.K @ V
                KV = batched_matmul(self.K, V)
                KV = (KV / self.K.shape[0])

                if self.norm_importance:
                    ## norm to 0 ->1
                    KV = KV - KV.min()
                    KV = KV / KV.max()

                    V = V - V.min()
                    V = V / V.max()

                own_factor = 1 - self.alpha if self.own_alpha_weighting else 1

                points_total_importance = own_factor * V + self.alpha * KV

            points_total_importance[curr_l_set] = -np.inf

            sampled_point = points_total_importance.argmax().item()

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

def batched_matmul(K, V, chunk_size=1024):
    D, N = K.shape
    result = torch.empty_like(V)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        result[start:end] = K[start:end] @ V
    return result