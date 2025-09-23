def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
import os
import sys
add_path(os.path.abspath('..'))

from tqdm import tqdm
import pycls.datasets.utils as ds_utils
import numpy as np
import torch
import seaborn as sns
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import time
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



all_features = torch.from_numpy(ds_utils.load_features('CIFAR100', train=True))
train_labels = np.load("/cs/labs/daphna/itai.david/py_repos/TypiClust/data/for_class_corr/cifar100_train_labels.npy")
kernel_fn = RBFKernel('cuda')
K = kernel_fn.compute_kernel(all_features, all_features, 1).to('cuda')

exp_path = "/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR100/resnet18/2025_8_14/CIFAR100_all_misp_from_features_2025_8_14_104059_635817/"



episodes_num = 32

def get_max_mean_classes_corr(activeset):
    num_classes = 100
    class_sim_max = torch.zeros((num_classes, num_classes), device='cuda')
    class_sim_mean = torch.zeros((num_classes, num_classes), device='cuda')
    class_indices = {label: np.where(train_labels[activeset] == label)[0] for label in range(num_classes)}

    for c1 in range(num_classes):
        for c2 in range(c1, num_classes):
            indices_c1 = class_indices.get(c1, [])
            indices_c2 = class_indices.get(c2, [])

            if not len(indices_c1) or not len(indices_c2):
                continue

            sim_submatrix = K[indices_c1, :][:, indices_c2]

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
    return class_sim_max, class_sim_mean

max_matrices = []
mean_matrices = []
acc_active_set = np.array([], dtype=int)
for ep in tqdm(range(episodes_num)):
    cur_path = exp_path + f"episode_{ep}/" + 'activeSet.npy'
    cur_active_set = np.load(cur_path)
    acc_active_set = np.concatenate((acc_active_set, cur_active_set), dtype=int)

    cur_max_mat, cur_mean_mat = get_max_mean_classes_corr(acc_active_set)
    max_matrices.append(cur_max_mat.cpu().numpy())
    mean_matrices.append(cur_mean_mat.cpu().numpy())



#visualize part

fig, ax = plt.subplots()

def update(frame, corr_matrices):
    ax.clear()
    sns.heatmap(corr_matrices[frame], vmin=-1, vmax=1, cmap="coolwarm", ax=ax, cbar=False)
    ax.set_title(f"Time {frame}")

wrap_max_update = lambda frame : update(frame, max_matrices)
wrap_mean_update = lambda frame : update(frame, mean_matrices)

max_ani = animation.FuncAnimation(fig, wrap_max_update, frames=len(max_matrices))
# max_ani.save("correlation_dynamics.gif", dpi=100, writer="imagemagick")
# HTML(max_ani.to_jshtml())
plt.show()



# Example: corr_matrices is a list/array of shape (32, 100, 100)
for t, mat in enumerate(mean_matrices):
    plt.clf()  # clear figure
    sorted_map = np.sort(mat, axis=1)
    sns.heatmap(sorted_map, vmin=-1, vmax=1, cmap="coolwarm", cbar=False)
    plt.title(f"Time {t}")
    plt.pause(0.3)  # seconds between frames

plt.show()


max_mean_values = []
max_std_values = []
for i in range(len(max_matrices)):
    max_mat_copy = max_matrices[i].copy()
    only_valid_matching = mat_copy  > 0
    max_mat_copy[~only_valid_matching] = np.nan

    mean_mat_copy = mean_matrices[i].copy()
    only_valid_matching = mean_mat_copy > 0
    mean_mat_copy[~only_valid_matching] = np.nan

    dist_from_center = max_mat_copy - mean_mat_copy
    class_max_values = np.nanmean(dist_from_center, axis=1)



    max_mean_values.append(np.nanmean(class_max_values))
    max_std_values.append(np.nanstd(class_max_values))

max_mean_values = np.array(max_mean_values)
max_std_values = np.array(max_std_values)

plt.close()
plt.plot(max_mean_values, label='Mean of max correlations')
plt.fill_between(range(len(max_mean_values)), max_mean_values - max_std_values, max_mean_values + max_std_values,
                 alpha=0.2, label='Std of max correlations')
plt.title("Mean and Std of Max Correlation per Class over Episodes")
plt.xlabel("Episode")
plt.ylabel("Correlation")
plt.legend()
plt.show()

# plt.plot(max_mean_values)
# plt.show()