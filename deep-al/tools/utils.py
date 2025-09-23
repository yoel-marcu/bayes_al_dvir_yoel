import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import sys
import os
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.abspath('..'))

import pycls.datasets.utils as ds_utils


def visualize_tsne(
    labeled_indices: np.ndarray = None,
    sampled_indices: np.ndarray = None,
    dataset_name: str = 'CIFAR10',
    algo_name: str = 'default',
):
    """
    Reduces embeddings to 2D using t-SNE and plots them with labels. Optionally subsamples for speed.

    :param embeddings: NumPy array of shape (N, D)
    :param labels: NumPy array of shape (N,)
    :param perplexity: t-SNE perplexity parameter
    :param random_state: Random seed for reproducibility
    :param max_samples: Number of samples to use for visualization
    :param save_path: Optional path to save the plot
    :param include_indices: Optional array of indices to always include and highlight
    """

    embeddings_path = f"/cs/labs/daphna/itai.david/py_repos/TypiClust/results/t-sne/{dataset_name}_t-sne_embeddings.npy"
    labels_path = f"/cs/labs/daphna/itai.david/py_repos/TypiClust/results/t-sne/{dataset_name}_t-sne_labels.npy"
    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path)
    num_classes = len(np.unique(labels))

    labeled_embeddings = embeddings[labeled_indices]
    labeled_labels = labels[labeled_indices]

    sampled_embeddings = embeddings[sampled_indices]
    sampled_labels = labels[sampled_indices]

    plt.figure(figsize=(10, 10))
    palette = sns.color_palette("hsv", num_classes)

    # Plot all points normally
    sns.scatterplot(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        hue=labels,
        palette=palette,
        legend="full",
        s=15,
        alpha=0.7
    )

    # Overlay included indices with different marker
    if len(labeled_indices) > 0:
        sns.scatterplot(
            x=labeled_embeddings[:, 0],
            y=labeled_embeddings[:, 1],
            hue=labeled_labels,
            palette=palette,
            marker='X',
            s=90,
            edgecolor='gray',
            linewidth=0.7,
            legend=False  # Avoid duplicate legend entries
        )
    if len(sampled_indices) > 0:
        sns.scatterplot(
            x=sampled_embeddings[:, 0],
            y=sampled_embeddings[:, 1],
            hue=sampled_labels,
            palette=palette,
            marker='X',
            s=110,
            edgecolor='black',
            linewidth=1,
            legend=False  # Avoid duplicate legend entries
        )

    plt.title(f"t-SNE Visualization of Embeddings by {algo_name}")
    plt.axis('off')

    # if save_path:
    #
    #     print(f"Plot saved to {save_path}")
    # else:
    num_labeled_samples = len(labeled_indices)
    save_full_path = f'/cs/labs/daphna/itai.david/py_repos/TypiClust/results/t-sne-visual/{algo_name}_{dataset_name}_labeled_{num_labeled_samples}.png'
    plt.savefig(save_full_path, bbox_inches='tight')
    plt.show()

    plt.close()


# def save_embeddings_to_file(embeddings: np.ndarray, labels: np.ndarray, dataset_name: str):
#     """
#     Saves embeddings and labels to a text file.
#
#     :param embeddings: NumPy array of shape (N, D)
#     :param labels: NumPy array of shape (N,)
#     :param file_path: Path to save the embeddings
#     """
#     assert embeddings.shape[0] == labels.shape[0], "Embeddings and labels must have same length"
#     embed_output_path = f"/cs/labs/daphna/itai.david/py_repos/TypiClust/results/t-sne/{dataset_name}_t-sne_embeddings.npy"
#     label_output_path = f"/cs/labs/daphna/itai.david/py_repos/TypiClust/results/t-sne/{dataset_name}_t-sne_labels.npy"
#     tsne = TSNE(n_components=2,
#                 perplexity=30,  # Experiment with values like 20, 30, 50
#                 learning_rate=200,  # Experiment with values like 100, 200, 500, 1000
#                 n_iter=1000,  # Number of iterations, 1000-5000 is common
#                 verbose=1,  # Show progress
#                 random_state=42,  # For reproducibility
#                 init='pca',  # Initialize with PCA results
#                 n_jobs=-1  # Use all available CPU cores
#                 )
#     tsne_results = tsne.fit_transform(embeddings)
#     # Save the embeddings
#     np.save(embed_output_path, tsne_results)
#     print(f"t-SNE embeddings saved to {embed_output_path}")
#
#     # Save the corresponding labels
#     np.save(label_output_path, labels)
#     print(f"Labels saved to {label_output_path}")
#
#
# if __name__ == "__main__":
#     # Example usage
#     all_features = ds_utils.load_features('CIFAR10', train=True)
#     # embeddings = np.random.rand(1000, 128)  # Example embeddings
#     labels = np.zeros_like(all_features)  # Example labels (10 classes)
#
#     # visualize_tsne(embeddings, labels, max_samples=500, include_indices=np.array([0, 1, 2]))
#     save_embeddings_to_file(all_features, labels, "CIFAR10")


# path = "/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR10/resnet18/CIFAR10_resnet18_2025_6_10_171959_564799/plot_episode_yvalues.npy"
#
# x = np.load(path)
# print(x)