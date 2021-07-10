# Reparametrized Gradeint Perturbation (RGP) on Language Tasks
In this folder, we apply the RGP algorithm to fine-tune the BERT model on four downstream tasks from the GLUE benchmark.


# Environment
This code is tested on Linux system with CUDA version 11.0

To run the source code, please first install the following packages:

```
python>=3.6
numpy>=1.15
torch>=1.8
scipy
six
apex
```

From my experience, installing apex via anaconda is a convenience appraoch. https://anaconda.org/conda-forge/nvidia-apex


# Some Implementation Details

The major modifcations are made in the following files:
```
bert_code/fairseq/models/roberta/model.py
bert_code/fairseq/modules/multihead_attention.py
bert_code/fairseq/modules/transformer_sentence_encoder_layer.py
bert_code/fariseq/lrk_utils.py
bert_code/fariseq/trainer.py
```
The reparametrized forward process is implemented in 'multihead_attention.py' and 'transformer_sentence_encoder_layer.py'. Codes for computing individual gradients and running power iterations are in 'lrk_utils.py'. The codes for collecting and processing individual gradients are in 'trainer.py'.

I provide a 'bert_code/run_exp.py' file to help you run experiments in an easier way. You can adjust most of the hyperparameters in that file.




# Example Commands

A pre-trained model is available at: https://drive.google.com/file/d/1xK4JaldIpOBmSaTmiCQVi9ef0D7YL0E2/view?usp=sharing. This model is pre-trained for 100k updates on the public data in https://github.com/google-research/bert. You can also load the parameters of your own pre-trained model.

Here is an example command to fine-tune the model on SST-2 dataset:
```
python run_exp.py --ckpt_dir path_to_checkpoint --batch_size 1000 --epoch 50 --gpu_id 0 --seed 0  --lr 3e-4 --eps 8 --delta 1e-5 --clip 10 --rank 1 --epoch 50 --sess debug_sst2 --to_console
```

You can also try other tasks such as MNLI, QNLI, QQP.



# Citation

```
@inproceedings{yu2021private,
  title={Large Scale Private Learning via Low-rank Reparametrization},
  author={Yu, Da and Zhang, Huishuai and Chen, Wei and Yin, Jian and Liu, Tie-Yan},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2021},
}
```

