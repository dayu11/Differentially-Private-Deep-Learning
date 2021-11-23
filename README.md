# Differentially-Private-Deep-Learning

## Update 11/18/2021

Update the results of fine-tuning RoBERTa-large with large batchsize and full precision.

## Update 09/01/2021

Our code for fine-tuning BERT models with differential privacy now supports loading official RoBERTa checkpoints.

## Readme

This repo provides some example code to help you get started with differentially private deep learning. 

Our implementation uses **Pytorch**. We cover several algorithms including *Differentially Private SGD* [[1]](https://arxiv.org/abs/1607.00133), *Gradient Embedding Perturbation* [[2]](https://openreview.net/forum?id=7aogOj_VYO0), and *Reparametrized Gradient Perturbation* [[3]](https://arxiv.org/abs/2106.09352).

In the *vision* folder, we implement the algorithms in [1,2,3] to train deep ResNets on benchmark vision datasets.

In the *language* folder, we implement the algorithm in [3] to fine-tune BERT models on four tasks from the [GLUE](https://gluebenchmark.com/) benchrmark.


## References


\[1\]: **Deep learning with differential privacy.** Martin Abadi, Andy Chu, Ian Goodfellow, H Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang.   In ACM SIGSAC Conference on Computer and Communications Security, 2016.

\[2\]: **Do Not Let Privacy Overbill Utility: Gradient Embedding Perturbation for Private Learning.**  Da Yu, Huishuai Zhang, Wei Chen, and Tie-Yan Liu.  In International Conference on Learning Representations (ICLR), 2021.

\[3\]: **Large Scale Private Learning via Low-rank Reparametrization.** Da  Yu,  Huishuai  Zhang,  Wei  Chen,  Jian  Yin,  and  Tie-Yan  Liu.    In International Conference on Machine Learning (ICML), 2021. 
