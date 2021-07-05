# Differentially-Private-Deep-Learning

This repo provides some example code to help you get started with differentially private deep learning. 

Our implementation uses **Pytorch**. We cover several algorithms including *Differentially Private SGD* [[1]](https://arxiv.org/abs/1607.00133), *Gradient Embedding Perturbation* [[2]](https://openreview.net/forum?id=7aogOj_VYO0), and *Reparametrized Gradient Perturbation* [[3]](https://arxiv.org/abs/2106.09352).

In the *vision* folder, we train deep ResNets from scratch on benchmark vision datasets such as the CIFAR-10 dataset. 

In the *language* folder, we fine-tune BERT models on four tasks from the [GLUE](https://gluebenchmark.com/) benchrmark.


## References


\[1\]: **Deep learning with differential privacy.** Martin Abadi, Andy Chu, Ian Goodfellow, H Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang.   In ACM SIGSAC Conference on Computer and Communications Security, 2016.

\[2\]: **Do Not Let Privacy Overbill Utility: Gradient Embedding Perturbation for Private Learning.**  Da Yu, Huishuai Zhang, Wei Chen, and Tie-Yan Liu.  In International Conference on Learning Representations (ICLR), 2021.

\[3\]: **Large Scale Private Learning via Low-rank Reparametrization.** Da  Yu,  Huishuai  Zhang,  Wei  Chen,  Jian  Yin,  and  Tie-Yan  Liu.    In International Conference on Machine Learning (ICML), 2021. 
