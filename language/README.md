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
prv_accountant
```

From my experience, installing apex via anaconda is a convenience appraoch. https://anaconda.org/conda-forge/nvidia-apex

To install prv_accountant, go to the 'prv_accountant' folder and run 
```
pip install --editable .
```

Then go to the `bert_code` folder and install fairseq by running:

```
pip install --editable .
```

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


Official pre-trained checkpoints of RoBERTa are available at: https://github.com/pytorch/fairseq/tree/master/examples/roberta. 

I pre-process the GLUE data following the instructions at https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.glue.md. The processed data are in the `glue_data` folder. 


Here is an example command to fine-tune the model on SST-2 dataset:
```
python run_exp.py --ckpt_dir path_to_checkpoint --batch_size 2000 --gpu_id 0 --seed 0 --accountant prv --lr 3e-4 --eps 8 --delta 1e-5 --clip 1 --rank 1 --epoch 30 --sess debug_sst2 --to_console
```

You can also try other tasks such as MNLI, QNLI, and QQP. 

# Results on MNLI, QNLI, QQP, and SST-2

The privacy bound is <img src="https://render.githubusercontent.com/render/math?math=(8 , 1e^{-5})">-differential privacy. We run experiments with `roberta.large` models. The batchsizes for MNLI, QQP, QNLI, and SST-2 are 14000, 12000, 4000, 2000, respectively. Experiments are run with full precision. The learning rate is chosen from \[1e-4, 3e-4\]. Other hyperparameters are the same as those in the example command.  The results (test accuracy in %) are in the table below.

| Model    | 	MNLI | QNLI | QQP | SST-2 |    Average |                                                                                         
| -----------    |-----------    |  -----------  | ----------- | ----------- |----------- |
| roberta.large   | 87.9		|    91.1     |  88.1    | 94.2 |  90.34   |




# Citation

```
@inproceedings{yu2021private,
  title={Large Scale Private Learning via Low-rank Reparametrization},
  author={Yu, Da and Zhang, Huishuai and Chen, Wei and Yin, Jian and Liu, Tie-Yan},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2021},
}
```

