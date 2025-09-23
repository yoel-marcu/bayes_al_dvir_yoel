import numpy as np

# DATASET_FEATURES_DICT = {
#     'train':
#         {
#             'CIFAR10':'/cs/labs/daphna/itai.david/py_repos/TypiClust/results/cifar-10/pretext/features_seed{seed}.npy',
#             'CIFAR100':'../../scan/results/cifar-100/pretext/features_seed{seed}.npy',
#             'TINYIMAGENET': '../../scan/results/tiny-imagenet/pretext/features_seed{seed}.npy',
#             'IMAGENET50': '../../dino/runs/trainfeat.pth',
#             'IMAGENET100': '../../dino/runs/trainfeat.pth',
#             'IMAGENET200': '../../dino/runs/trainfeat.pth',
#         },
#     'test':
#         {
#             'CIFAR10': '/cs/labs/daphna/itai.david/py_repos/TypiClust/results/cifar-10/pretext/features_seed{seed}.npy',
#             'CIFAR100': '../../scan/results/cifar-100/pretext/test_features_seed{seed}.npy',
#             'TINYIMAGENET': '../../scan/results/tiny-imagenet/pretext/test_features_seed{seed}.npy',
#             'IMAGENET50': '../../dino/runs/testfeat.pth',
#             'IMAGENET100': '../../dino/runs/testfeat.pth',
#             'IMAGENET200': '../../dino/runs/testfeat.pth',
#         }
# }

DATASET_FEATURES_DICT = {
    'train':
        {
            'CIFAR10':'/cs/labs/daphna/itai.david/representations_bank/cifar-10_simclr/pretext/features_seed1.npy',
            'CIFAR100': '/cs/labs/daphna/itai.david/representations_bank/cifar-100_simclr/cifar100_simclr_train.npy',
            'TINYIMAGENET': '/cs/labs/daphna/itai.david/representations_bank/tiny-imagenet_simclr/pretext/features_seed1.npy',
            'IMAGENET50': '/cs/labs/daphna/itai.david/representations_bank/imagenet_dinov2/train_features.npy',
            'IMAGENET100': '/cs/labs/daphna/itai.david/representations_bank/imagenet_dinov2/train_features.npy',
            'IMAGENET200': '/cs/labs/daphna/itai.david/representations_bank/imagenet_dinov2/train_features.npy',
            'IMAGENET': '/cs/labs/daphna/itai.david/representations_bank/imagenet_dinov2/train_features.npy'
        },
    'test':
        {
            'CIFAR10': '/cs/labs/daphna/itai.david/representations_bank/cifar-10_simclr/pretext/test_features_seed1.npy',
            'CIFAR100': '/cs/labs/daphna/itai.david/representations_bank/cifar-100_simclr/cifar100_simclr_test.npy',
            'TINYIMAGENET': '/cs/labs/daphna/itai.david/representations_bank/tiny-imagenet_simclr/pretext/test_features_seed1.npy',
            'IMAGENET50': '/cs/labs/daphna/itai.david/representations_bank/imagenet_dinov2/val_features.npy',
            'IMAGENET100': '/cs/labs/daphna/itai.david/representations_bank/imagenet_dinov2/val_features.npy',
            'IMAGENET200': '/cs/labs/daphna/itai.david/representations_bank/imagenet_dinov2/val_features.npy',
            'IMAGENET': '/cs/labs/daphna/itai.david/representations_bank/imagenet_dinov2/val_features.npy'
        }
}

def load_features(ds_name, seed=1, train=True, normalized=True):
    " load pretrained features for a dataset "
    split = "train" if train else "test"
    fname = DATASET_FEATURES_DICT[split][ds_name].format(seed=seed)
    if fname.endswith('.npy'):
        features = np.load(fname)
    elif fname.endswith('.pth'):
        features = torch.load(fname)
    else:
        raise Exception("Unsupported filetype")
    if normalized:
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
    return features