# Getting Started

## Setup

Clone the repository
```
git clone https://github.com/avihu111/TypiClust
cd TypiClust
```

Create an environment using
```
conda create --name typiclust --file typiclust-env.txt
conda activate typiclust
pip install pyyaml easydict termcolor tqdm simplejson yacs
```

If that fails, it might be due to incompatible CUDA version.
In that case, try installing by
```
conda create --name typiclust python=3.7
conda activate typiclust
conda install pytorch torchvision torchaudio cudatoolkit=<CUDA_VERSION> -c pytorch
conda install matplotlib scipy scikit-learn pandas
conda install -c conda-forge faiss-gpu
pip install pyyaml easydict termcolor tqdm simplejson yacs
```
Select the GPU to be used by running
```
CUDA_VISIBLE_DEVICES=0
```

## Representation Learning
Both TypiClust variants and ProbCover rely on representation learning. 
To train CIFAR-10 on simclr please run
```
cd scan
python simclr.py --config_env configs/env.yml --config_exp configs/pretext/simclr_cifar10.yml
cd ..
```
When this finishes, the file `./results/cifar-10/pretext/features_seed1.npy` should exist.

To save time, you can download the features of CIFAR-10/100 from here:

| Dataset          | Download link |
|------------------|---------------| 
|CIFAR10           | [Download](https://drive.google.com/file/d/1Le1ZuZOpfxBfxL3nnNahZcCt-lLWLQSB/view?usp=sharing)  |
|CIFAR100          | [Download](https://drive.google.com/file/d/1o2nz_SKLdcaTCB9XVA44qCTVSUSmktUb/view?usp=sharing)  |


and locate the files here:`./results/cifar-10/pretext/features_seed1.npy`.

## TypiClust - K-Means variant
To select samples according to TypiClust (K-Means) where the `initial_size=0` and the `budget=100` please run 
```
cd deep-al/tools
python train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al typiclust_rp --exp-name auto --initial_size 0 --budget 100
cd ../../
```


## TypiClust - SCAN variant
In this section we select `budget=10` samples without an initial set. 
We first must run SCAN clustering algorithm, as TypiClust uses its features cluster assignments.
Please repeat the following command  `for k in [10, 20, 30, 40, 50, 60]`

```
cd scan
python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_cifar10.yml --num_clusters k
cd .. 
```
You can use other representations and change the path in the file `deep-al/pycls/datasets/utils/features.py`.
Then, you can run the active learning experiment by running
```
cd deep-al/tools
python train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al typiclust_dc --exp-name auto --initial_size 0 --budget 10
cd ../../
```

## ProbCover
Example ProbCover script
```
cd deep-al/tools
python train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al probcover --exp-name auto --initial_size 0 --budget 10 --delta 0.75
cd ../../
```

## DCoM
Example DCoM script:
```
cd deep-al/tools
python train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al dcom --exp-name auto --initial_size 0 --budget 10 --initial_delta 0.75
cd ../../
```

You can add the `a_logistic` and `k_logistic` parameters to the run using `--a_logistic` and `--k_logistic`.
