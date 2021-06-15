# Implementation of Reparametrized Gradient Perturbation (RGP)

See our paper "Large Scale Private Learning via Low-rank Reparametrization" in ICML 2021 for more details.


# Environment
This code is tested on Linux system with CUDA version 11.0

To run the source code, please first install the following packages:

python>=3.6
numpy>=1.15
torch>=1.3
torchvision>=0.4
scipy
six

# Example commands

An example command that trains a WRN28-10 model with (8, 1e-5)-DP:

    CUDA_VISIBLE_DEVICES=0 python cifar_train.py  --eps 8 --delta 1e-5 --rank 16 --lr 0.5 --clipping 1 --batchsize 1000 --n_epoch 400 --width 10

Training for one epoch takes ~120 seconds on a single Tesla V100 GPU. This command achieves ~69% test accuracy on CIFAR10. 

You can also try different choices of model width:

    CUDA_VISIBLE_DEVICES=0 python cifar_train.py  --eps 8 --delta 1e-5 --rank 16 --lr 1.0 --clipping 1 --batchsize 1000 --n_epoch 400 --width 4

# Citation

```
@inproceedings{yu2021private,
  title={Large Scale Private Learning via Low-rank Reparametrization},
  author={Yu, Da and Zhang, Huishuai and Chen, Wei and Yin, Jian and Liu, Tie-Yan},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2021},
}
```
