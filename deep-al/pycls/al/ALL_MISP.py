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

class ALL_MISP:
    def __init__(self, cfg, budgetSize, train_labels, lset, delta=1):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.all_features = ds_utils.load_features(self.ds_name, train=True)
        self.alpha = self.cfg.ALPHA
        self.debug = self.cfg.DEBUG
        self.norm_importance = self.cfg.NORM_IMPORTANCE
        self.confidence_method = self.cfg.CONFIDENCE_METHOD if 'CONFIDENCE_METHOD' in self.cfg else 'max'
        self.budgetSize = budgetSize
        self.delta = delta
        self.soft_border_val = self.cfg.SOFT_BORDER_VAL if 'SOFT_BORDER_VAL' in self.cfg else 0.15
        self.diff_method = self.cfg.DIFF_METHOD if 'DIFF_METHOD' in self.cfg else 'abs_diff'
        kernel_type = self.cfg.KERNEL_TYPE if 'KERNEL_TYPE' in self.cfg else 'rbf'
        if kernel_type == 'soft_tophat':
            self.kernel_fn = SoftTopHatKernel('cuda')
        elif kernel_type == 'tophat':
            self.kernel_fn = TopHatKernel('cuda')
        else:
            self.kernel_fn = RBFKernel('cuda')

        self.train_labels_general = np.array(train_labels)
        unique_labels = np.unique(self.train_labels_general)
        self.C_general = torch.zeros((self.all_features.shape[0], unique_labels.size), device='cuda')
        self.chosen_labels_num = torch.zeros(np.unique(self.train_labels_general).size).to('cuda')
        if lset is not None and lset.size > 0:
            temp_K = self.kernel_fn.compute_kernel(
                torch.from_numpy(self.all_features), torch.from_numpy(self.all_features), self.delta).to('cuda')
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

            curr_l_set = np.concatenate((np.arange(len(self.lSet)), selected)).astype(int)
            num_classes = self.chosen_labels_num.size(0)
            class_sim_max = torch.zeros((num_classes, num_classes), device='cuda')
            class_sim_mean = torch.zeros((num_classes, num_classes), device='cuda')
            class_indices = {label: np.where(self.train_labels[curr_l_set] == label)[0] for label in range(num_classes)}

            for c1 in range(num_classes):
                for c2 in range(c1, num_classes):
                    indices_c1 = class_indices.get(c1, [])
                    indices_c2 = class_indices.get(c2, [])

                    if not len(indices_c1) or not len(indices_c2):
                        continue

                    sim_submatrix = self.K[indices_c1, :][:, indices_c2]

                    if c1 == c2:
                        if len(indices_c1) > 1:
                            # Exclude diagonal for intra-class similarity
                            non_diagonal_mask = ~torch.eye(len(indices_c1), dtype=torch.bool,
                                                           device=sim_submatrix.device)
                            if non_diagonal_mask.any():
                                class_sim_mean[c1, c1] = sim_submatrix[non_diagonal_mask].mean()
                                class_sim_max[c1, c1] = sim_submatrix[non_diagonal_mask].max()
                    else:
                        mean_val = sim_submatrix.mean()
                        max_val = sim_submatrix.max()
                        class_sim_mean[c1, c2] = mean_val
                        class_sim_mean[c2, c1] = mean_val
                        class_sim_max[c1, c2] = max_val
                        class_sim_max[c2, c1] = max_val


            if self.diff_method == '2_closest_diff':
                # C_softmax = torch.nn.functional.softmax(C_sum, dim=1)
                # vals, inds = torch.topk(C_softmax, k=2, dim=1)
                # C_diff = (vals[:, 0] - vals[:, 1])

                # vals, inds = torch.topk(C_sum, k=2, dim=1)
                # C_sum_per_point = torch.sum(C_sum, dim=1)
                # C_diff = (vals[:, 0] - vals[:, 1]) / (C_sum_per_point + 1e-8)

                vals, inds = torch.topk(self.C, k=2, dim=1)
                C_sum_per_point = torch.sum(self.C, dim=1)
                C_diff = (vals[:, 0] - vals[:, 1]) / (C_sum_per_point + 1e-8) if self.confidence_method == 'margin' else vals[:, 0] / (C_sum_per_point + 1e-8)

                point_total_contribution = batched_diffs(self.K, C_diff, diff_method="abs_diff")
            elif self.diff_method == '1_closest_diff':
                C_diff = torch.max(self.C, dim=1).values
                point_total_contribution = batched_diffs(self.K, C_diff, diff_method="abs_diff")

            elif self.diff_method == 'combine_uncert_type':
                # vals, inds = torch.topk(self.C, k=2, dim=1)
                # C_sum_per_point = torch.sum(self.C, dim=1)
                # C_diff = (vals[:, 0] - vals[:, 1]) / (C_sum_per_point + 1e-8)
                #
                #
                # C1 = vals[:, 0] / (self.chosen_labels_num[inds[:, 0]] + 1e-8)
                # C2 = vals[:, 1] / (self.chosen_labels_num[inds[:, 1]] + 1e-8)
                # J = 2 * (C1 * C2) / (C1 + C2 + 1e-8)
                # margin = C1 - C2
                # U = (1 - margin) * J
                # point_total_contribution = batched_diffs(self.K, C1, diff_method="abs_diff", U=U)

                vals_new, inds_new = torch.topk(self.C / (self.chosen_labels_num + 1e-8) , k=2, dim=1)
                C1_new = vals_new[:, 0]
                C2_new = vals_new[:, 1]

                J_new = 2 * (C1_new * C2_new) / (C1_new + C2_new + 1e-8)
                margin_new = C1_new - C2_new
                U_new = (1 - margin_new) * J_new
                point_total_contribution = batched_diffs(self.K, C1_new, diff_method="combine_uncert_type", U=U_new)

            elif self.diff_method == 'combine_uncert_type_U0':
                vals_new, inds_new = torch.topk(self.C / (self.chosen_labels_num + 1e-8) , k=2, dim=1)
                C1_new = vals_new[:, 0]
                point_total_contribution = batched_diffs(self.K, C1_new, diff_method="abs_diff")

            elif self.diff_method == 'combine_uncert_type_U2':
                vals_new, inds_new = torch.topk(self.C / (self.chosen_labels_num + 1e-8) , k=2, dim=1)
                C1_new = vals_new[:, 0]
                C2_new = vals_new[:, 1]

                J_new = 2 * (C1_new * C2_new) / (C1_new + C2_new + 1e-8)
                margin_new = C1_new - C2_new
                U_new = (1 - margin_new) * J_new
                point_total_contribution = batched_diffs(self.K, C1_new, diff_method="combine_uncert_type", U=U_new**2)


            elif self.diff_method == 'combine_uncert_type_J2':
                vals_new, inds_new = torch.topk(self.C / (self.chosen_labels_num + 1e-8) , k=2, dim=1)
                C1_new = vals_new[:, 0]
                C2_new = vals_new[:, 1]

                J_new = 2 * (C1_new * C2_new) / (C1_new + C2_new + 1e-8)
                margin_new = C1_new - C2_new
                U_new = (1 - margin_new) * J_new**2
                point_total_contribution = batched_diffs(self.K, C1_new, diff_method="combine_uncert_type", U=U_new)

            elif self.diff_method == 'combine_uncert_type_outer':
                vals_new, inds_new = torch.topk(self.C, k=2, dim=1)
                C1_new = vals_new[:, 0]
                C2_new = vals_new[:, 1]

                J_new = 2 * (C1_new * C2_new) / (C1_new + C2_new + 1e-8)
                margin_new = C1_new - C2_new
                U_new = (1 - margin_new) * J_new
                point_total_contribution_from_others = batched_diffs(self.K, C1_new, diff_method="abs_diff")

                corr_factor = (class_sim_max[inds_new[:, 0], inds_new[:, 1]] - class_sim_mean[inds_new[:, 0], inds_new[:, 1]])
                point_total_contribution = point_total_contribution_from_others + corr_factor * U_new

            elif self.diff_method == 'combine_uncert_type_outer_mean':
                vals_new, inds_new = torch.topk(self.C / (self.chosen_labels_num + 1e-8), k=2, dim=1)
                C1_new = vals_new[:, 0]
                C2_new = vals_new[:, 1]

                J_new = 2 * (C1_new * C2_new) / (C1_new + C2_new + 1e-8)
                margin_new = C1_new - C2_new
                U_new = (1 - margin_new) * J_new
                point_total_contribution_from_others = batched_diffs(self.K, C1_new, diff_method="abs_diff")

                corr_factor = (class_sim_max[inds_new[:, 0], inds_new[:, 1]] - class_sim_mean[inds_new[:, 0], inds_new[:, 1]])
                point_total_contribution = point_total_contribution_from_others + corr_factor * U_new

            elif self.diff_method == 'prob_method_v1':
                if i > 0:
                    vals_new, inds_new = torch.topk(self.C / i, k=2, dim=1)
                    C1_new = vals_new[:, 0]
                    C2_new = vals_new[:, 1]
                else:
                    C1_new = C2_new = torch.zeros(self.C.size(0)).to(device=self.C.device)
                J_new = 2 * (C1_new * C2_new) / (C1_new + C2_new + 1e-8)
                margin_new = C1_new - C2_new
                U_new = (1 - margin_new) * J_new
                point_total_contribution = batched_diffs(self.K, C1_new, diff_method="abs_diff")

            else:
                point_total_contribution = batched_diffs(self.K, self.C, diff_method=self.diff_method)
            point_total_contribution[curr_l_set] = -np.inf
            sampled_point = point_total_contribution.argmax().item()
            chosen_label = self.train_labels[sampled_point].item()

            self.chosen_labels_num[chosen_label] += 1

            self.C[:, chosen_label] += self.K[sampled_point]
            # self.C[:, chosen_label] = torch.maximum(self.K[sampled_point], self.C[:, chosen_label])
            if self.diff_method == 'prob_method_v1':
                c_prob = (1 - self.K[sampled_point]) / (self.K.size(0) - 1)
                c_prob_mask = torch.ones(self.C.size(1), dtype=torch.bool)
                c_prob_mask[chosen_label] = False
                self.C[:, c_prob_mask] += c_prob.repeat(99, 1).T



            assert sampled_point not in selected, 'sample was already selected'
            selected.append(sampled_point)

            # vals, inds = torch.topk(C_sum[l], k=2, dim=1)


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

def batched_diffs(K, C, chunk_size=1024, diff_method="abs_diff", U=None):
    D, N = K.shape
    result = torch.empty(D).to(device=C.device)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        if diff_method == "abs_diff":
            result[start:end] = torch.sum(torch.maximum(K[start:end] - C, torch.zeros_like(K[start:end]).to(device=C.device)), dim=1)
        elif diff_method == "max":
            # find places K > C
            pos_mask = K[start:end] > C
            temp_K = K[start:end]
            temp_K[~pos_mask] = 0
            result[start:end] = torch.sum(temp_K, dim=1)
            del pos_mask, temp_K
        elif diff_method == "combine_uncert_type":
            if U is None:
                raise ValueError("U must be provided for combine_uncert_type method")

            result[start:end] = torch.sum(
                torch.maximum((K[start:end] - C) + (K[start:end] * U), torch.zeros_like(K[start:end]).to(device=C.device)), dim=1)
        else:
            raise ValueError(f"Unknown diff method: {diff_method}")
    # del vals, inds
    return result