This folder contains the code for ICML submission 'Private Learning at Scale via Low-rank Reparametrization'.

The implementation of reparametrized convolutional layer can be found in the 'LrkConv2d' module in models/resnet_cifar.py.

An example command that trains a WRN28-10 model with (8, 1e-5)-DP:

    CUDA_VISIBLE_DEVICES=0 python cifar_train.py  --eps 8 --delta 1e-5 --rank 16 --lr 0.5 --clipping 1 --batchsize 1000 --n_epoch 400

Training for one epoch takes ~120 seconds on a single Tesla V100 GPU. This command achieves ~69% test accuracy on CIFAR10. 