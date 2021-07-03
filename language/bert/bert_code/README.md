# MC-BERT

Implementation for the paper "MC-BERT: Efficient Language Pre-Training via a Meta Controller",  which has been submitted to ICML'2020.

## Brief Introduction
This repo is built for the experimental codes in our paper, containing all the model implementation, data preprocessing, and parameter settings. Here we thank the authors of the codebase, [fairseq](https://github.com/pytorch/fairseq), and our repo is upgraded from it. So more details and usages on fairseq please see the original repo.

## Upgrades

#### General
1. Add metrics for down-stream tasks. Except for the original accuracy metric, we add pearson_spearman, F1, MCC for various down-stream tasks in GLUE, see the `critetions/sentence_prediction.py` for sentence prediction.
2. Checkpoints saving related settings, see the checkpoints utils file. Support transformer-v2 and fix v1 setting based on the original repo, see `modules/transformer_sentence_encoder.py`.

#### ELECTRA
1. Add generator and the other logics, in the model definition file, see the new folder `electra` in `models`.
2. Add a new dataset, with the same as MC-BERT, named `mask_tokens_dataset2.py`.
3. Define a new loss in criterions, see `electra.py` in `criterions`; a new task, see `electra.py` in `tasks`.

#### MC-BERT
1. Add the meta controller and the other logics, in the model definition file, see the new folder `mcbert` in `models`.
2. Define a new loss in criterions, see `mcbert.py` in `criterions`;  a new task, see `mcbert.py` in `tasks`.

## Requirements and Installation

More details see [fairseq](https://github.com/pytorch/fairseq). Berifly,

* [PyTorch](http://pytorch.org/) version >= 1.2.0
* Python version >= 3.5
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library with the `--cuda_ext` option

**Installing from source**

To install MC-BERT from source and develop locally:
```bash
git clone https://github.com/MC-BERT/MC-BERT
cd MC-BERT
pip install --editable .
```

## Getting Started

### Overall Usage
The [full documentation](https://fairseq.readthedocs.io/) of fairseq contains instructions for getting started, training new models and extending fairseq with new model types and tasks.

### Data Pre-Processing

#### Pretraining Data
We follow a couple of consecutive pre-processing steps: segmenting documents into sentences by Spacy, normalizing, lower-casing, and tokenizing the texts by Moses decoder, and finally, applying byte pair encoding (BPE) with setting the vocabulary size |V| as 32,678. The preprocess code refers to `preprocess/pretrain/process.sh`.

#### Down-Stream Data
Follow the procedure as the above one, we process the GLUE by `preprocess/glue/process.sh`.

When reproducing, please modify some related file paths.

### Pre-Training Usage

#### ELECTRA
For pre-training ELECTRA model, you can refer to the following:
```bash
#!/usr/bin/env bash
EXEC_ID=electra-50L
DATA_DIR=../data-bin/wiki_book_32768
TOTAL_UPDATES=1000000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0001          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=8        # Number of sequences per batch (batch size)
UPDATE_FREQ=4          # Increase the batch size 16x
SEED=100

python train.py ${DATA_DIR} --fp16 --num-workers 4 --ddp-backend=no_c10d \
       --task electra --criterion electra \
       --arch electra --sample-break-mode complete --tokens-per-sample ${TOKENS_PER_SAMPLE} \
       --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
       --lr-scheduler polynomial_decay --lr ${PEAK_LR} --warmup-updates ${WARMUP_UPDATES} --total-num-update ${TOTAL_UPDATES} \
       --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
       --max-sentences ${MAX_SENTENCES} --update-freq ${UPDATE_FREQ} --seed ${SEED} \
       --loss-lamdba 50.0 --mask-prob 0.15 \
       --embedding-normalize --generator-size-divider 3 \
       --max-update ${TOTAL_UPDATES} --log-format simple --log-interval 100 --tensorboard-logdir ../tsb_log/electra-${EXEC_ID} \
       --distributed-world-size 8 --distributed-rank 0 --distributed-init-method "tcp://xxx.xxx.xxx.xxx:8080" \
       --keep-updates-list 20000 50000 100000 200000 \
       --save-interval-updates 10000 --keep-interval-updates 5 --no-epoch-checkpoints --skip-invalid-size-inputs-valid-test \
       --save-dir ../saved_cp/electra-${EXEC_ID}
```
#### MC-BERT
For pretrainning MC-BERT model, you can refer to the following:
```bash
#!/usr/bin/env bash
EXEC_ID=mcbert-10C-50L
DATA_DIR=../data-bin/wiki_book_32768
TOTAL_UPDATES=1000000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0001          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=8        # Number of sequences per batch (batch size)
UPDATE_FREQ=4          # Increase the batch size 16x
SEED=100

python train.py ${DATA_DIR} --fp16 --num-workers 4 --ddp-backend=no_c10d \
       --task mcbert --criterion mcbert \
       --arch mcbert_base --sample-break-mode complete --tokens-per-sample ${TOKENS_PER_SAMPLE} \
       --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
       --lr-scheduler polynomial_decay --lr ${PEAK_LR} --warmup-updates ${WARMUP_UPDATES} --total-num-update ${TOTAL_UPDATES} \
       --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
       --max-sentences ${MAX_SENTENCES} --update-freq ${UPDATE_FREQ} --seed ${SEED} \
       --loss-lambda 50.0 --mask-prob 0.15 --class-num 10 \
       --embedding-normalize --mc-size-divider 3 \
       --max-update ${TOTAL_UPDATES} --log-format simple --log-interval 100 --tensorboard-logdir ../tsb_log/mcbert-${EXEC_ID} \
       --distributed-world-size 8 --distributed-rank 0 --distributed-init-method "tcp://xxx.xxx.xxx.xxx:8080" \
       --keep-updates-list 20000 50000 100000 200000 \
       --save-interval-updates 10000 --keep-interval-updates 5 --no-epoch-checkpoints --skip-invalid-size-inputs-valid-test \
       --save-dir ../saved_cp/mcbert-${EXEC_ID}
```

### Fine-tuning
After setting hyperparameters, you can fine-tune the model by:

```bash
python train.py $DATA_PATH/${PROBLEM}-bin \
       --restore-file $BERT_MODEL_PATH \
       --max-positions 512 \
       --max-sentences $SENT_PER_GPU \
       --max-tokens 4400 \
       --task sentence_prediction \
       --reset-optimizer --reset-dataloader --reset-meters \
       --required-batch-size-multiple 1 \
       --init-token 0 --separator-token 2 \
       --arch $ARCH \
       --criterion sentence_prediction \
       --num-classes $N_CLASSES \
       --dropout 0.1 --attention-dropout 0.1 \
       --weight-decay $WEIGHT_DECAY --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
       --clip-norm 0.0 \
       --lr-scheduler polynomial_decay --lr $LR --total-num-update $N_UPDATES --warmup-updates $WARMUP_UPDATES\
       --max-epoch $N_EPOCH --seed $SEED --save-dir $OUTPUT_PATH --no-progress-bar --log-interval 100 --no-epoch-checkpoints --no-last-checkpoints --no-best-checkpoints \
       --find-unused-parameters --skip-invalid-size-inputs-valid-test --truncate-sequence --embedding-normalize \
       --tensorboard-logdir $TENSORBOARD_LOG/${PROBLEM}/${N_EPOCH}-${BATCH_SZ}-${LR}-${WEIGHT_DECAY}-$SEED \
       --best-checkpoint-metric $METRIC --maximize-best-checkpoint-metric
```

More detailed setting and results are given in the supplementary files. Thanks for your visiting, if you have any questions, please new an issue.
