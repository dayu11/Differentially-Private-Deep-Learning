# Implementation of Gradient Embeding Perturbation (GEP)

See our paper "Do not Let Privacy Overbill Utility: Gradient Embedding Perturbation for Private Learning" in ICLR 2021 for more details.
https://openreview.net/forum?id=7aogOj_VYO0

# Environment
This code is tested on Linux system with CUDA version 11.0

To run the source code, please first install the following packages:

    python>=3.6
    numpy>=1.15
    torch>=1.3
    torchvision>=0.4
    scipy
    six
    backpack-for-pytorch

# Example commands

Private learning with GEP:
This code supports CIFAR10 and extended SVHN datasets. The following command trains a ResNet20 model with GEP using a 1000-dimensional anchor subspace and 2000 auxiliary samples from ImageNet.

    CUDA_VISIBLE_DEVICES=0 python main.py  --private --rgp --clip0 5 --clip1 2 --num_bases 1000 --aux_dataset imagenet --aux_data_size 2000 --sess cifar10_GEP_default

The default command evenly divides the parameters into three groups. This parameter grouping trick significantly reduces the memory cost. 

You can also train with Biased-GEP with the '--rgp' flag removed.

    CUDA_VISIBLE_DEVICES=0 python main.py  --private --clip0 5 --num_bases 1000 --aux_dataset imagenet --aux_data_size 2000 --sess cifar10_BGEP_default

The default dataset is CIFAR10, you can train with SVHN using the following command.

    CUDA_VISIBLE_DEVICES=0 python main.py  --private --rgp --dataset 'svhn' --n_epoch 10 --clip0 5 --clip1 2 --num_bases 1000 --aux_dataset imagenet --aux_data_size 2000 --sess svhn_GEP_default

We divide the parameters into multiple groups to reduce the computational/memory cost (see the discussion in Section 3.1 and Appendix B). The number of groups is controlled by the argument '--num_groups'. The default choice is 3. Increasing '--num_groups' reduces both computational and memory costs, but leads to slightly worse performance.

    CUDA_VISIBLE_DEVICES=0 python main.py  --private --rgp --num_groups 10 --clip0 5 --clip1 2 --num_bases 1000 --aux_dataset imagenet --aux_data_size 2000 --sess cifar10_GEP_default

The default choice of auxiliary dataset is ImageNet. We downsample the samples from ImageNet and store them in the file 'imagenet_examples_2000'. You can also use samples from CIFAR100 as the auxiliary dataset, which yields similar performance.

    CUDA_VISIBLE_DEVICES=0 python main.py  --private --rgp  --clip0 5 --clip1 2 --num_bases 1000 --aux_dataset cifar100 --aux_data_size 2000 --sess cifar10_GEP_default

# Citation

```
@inproceedings{yu2021do,
  title={Do Not Let Privacy Overbill Utility: Gradient Embedding Perturbation for Private Learning},
  author={Yu, Da and Zhang, Huishuai and Chen, Wei and  Liu, Tie-Yan},
 year      = {2021},
 booktitle = {International Conference on Learning Representations (ICLR)}
}
```
