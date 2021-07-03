# run by `python profiling.py -a mcbert .`

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from itertools import chain

from apex.contrib.multihead_attn import SelfMultiheadAttn
from fairseq import utils, optim, options
from fairseq.modules import TransformerSentenceEncoder, TransformerSentenceEncoderLayer, MultiheadAttention, LayerNorm
from apex.optimizers import FusedAdam

import torch.cuda.profiler as profiler
import pyprof2.pyprof2 as pyprof2
pyprof2.init()

if not torch.cuda.is_available():
    raise NotImplementedError('Running on CPU is not supported')
torch.cuda.set_device(0)

torch.manual_seed(111)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(111)

warmup_trials = 2
trials = 10
batch_size = 16
accumulate_steps = 16
seq_len = 512
hidden_dim = 768
ffn_embedding_dim = 3072
num_layer = 12
num_head = 12
vocab_size = 30000

class MLM(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the unmasked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x

def run_bert_benchmark(encoder, title):
    encoder = encoder.half()
    encoder.train()
    head = MLM(hidden_dim, vocab_size, "gelu", encoder.embed_tokens.weight).cuda().half()
    params = list(
        filter(
            lambda p: p.requires_grad,
            chain(encoder.parameters(), head.parameters()),
        )
    )
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)
    args.optimizer = "adam"
    args.fp16 = True
    opt = optim.FP16Optimizer.build_optimizer(args, params)
    with torch.autograd.profiler.emit_nvtx():
        inputs_seq = torch.randint(0, vocab_size, (batch_size, seq_len), device=torch.device("cuda")).requires_grad_(False)
        mask_pos = torch.cuda.FloatTensor(batch_size, seq_len).uniform_() > 0.85
        label = inputs_seq[mask_pos]
        target_trial = int(trials / 2) + warmup_trials
        for trial in range(0, trials + warmup_trials) :

            if trial == target_trial:
                profiler.start()

            layer_inputs  = inputs_seq
            evt_idx       = trial - warmup_trials
            opt.zero_grad()
            for _ in range(accumulate_steps):
                x = encoder.forward(layer_inputs, last_state_only=True)[0][0]
                x = head(x, mask_pos)
                loss = F.nll_loss(
                    F.log_softmax(
                        x.view(-1, x.size(-1)),
                        dim=-1,
                        dtype=torch.float32,
                    ),
                    label.view(-1),
                    reduction='sum',
                    ignore_index=0,
                )
                loss.backward()
            opt.step()

            if trial == target_trial:
                profiler.stop()

    print(f"{title} data collected.")


def run_bert():
    encoder = TransformerSentenceEncoder(
            padding_idx=0,
            vocab_size=vocab_size,
            num_encoder_layers=num_layer,
            num_attention_heads=num_head,
            max_seq_len=seq_len,
            num_segments = 0,
            embedding_dim=hidden_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            activation_fn="gelu",
            ).cuda()
    run_bert_benchmark(encoder, "BERT-base (with apex)")

def run_bert_nonapex():
    encoder = TransformerSentenceEncoder(
            padding_idx=0,
            vocab_size=vocab_size,
            num_encoder_layers=num_layer,
            num_attention_heads=num_head,
            max_seq_len=seq_len,
            num_segments = 0,
            embedding_dim=hidden_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            activation_fn="gelu",
            use_apex=False
            ).cuda()
    run_bert_benchmark(encoder, "BERT-base (no apex)")


def run_component_benchmark(name, generate_model_and_data, forward, backward):
    model, data = generate_model_and_data()
    target_trial = int(trials / 2) + warmup_trials

    with torch.autograd.profiler.emit_nvtx():
        for trial in range(0, trials + warmup_trials) :

            if trial == target_trial:
                profiler.start()

            for _ in range(accumulate_steps):
                output = forward(model, data)
                backward(output, data)

            if trial == target_trial:
                profiler.stop()

    print(f"{name}: data collected")


def run_single_transfomer():
    def model_and_data():
        encoder = TransformerSentenceEncoderLayer(
                num_attention_heads=12,
                activation_fn="gelu",
                embedding_dim=hidden_dim,
                ffn_embedding_dim=ffn_embedding_dim
                ).cuda()
        encoder = encoder.half()
        encoder.train()
        inputs = torch.randn(seq_len, batch_size, hidden_dim, dtype=torch.float16, device=torch.device("cuda")).requires_grad_(True)
        grads = torch.randn_like(inputs)
        return encoder, (inputs, grads)
    def forward(model, data):
        return model.forward(data[0])
    def backward(output, data):
        return output.backward(data[1])
    run_component_benchmark("Transformer (with apex)", model_and_data, forward, backward)

def run_single_transfomer_nonapex():
    def model_and_data():
        encoder = TransformerSentenceEncoderLayer(
                num_attention_heads=12,
                activation_fn="gelu",
                embedding_dim=hidden_dim,
                ffn_embedding_dim=ffn_embedding_dim,
                use_apex=False
                ).cuda()
        encoder = encoder.half()
        encoder.train()
        inputs = torch.randn(seq_len, batch_size, hidden_dim, dtype=torch.float16, device=torch.device("cuda")).requires_grad_(True)
        grads = torch.randn_like(inputs)
        return encoder, (inputs, grads)
    def forward(model, data):
        return model.forward(data[0])
    def backward(output, data):
        return output.backward(data[1])
    run_component_benchmark("Transformer (no apex)", model_and_data, forward, backward)

def run_attention():
    def model_and_data():
        encoder = SelfMultiheadAttn(hidden_dim, 12, dropout=0.1, bias=False,  impl='fast').cuda()
        encoder = encoder.half()
        inputs = torch.randn(seq_len, batch_size, hidden_dim, dtype=torch.float16, device=torch.device("cuda")).requires_grad_(True)
        grads = torch.randn_like(inputs)
        return encoder, (inputs, grads)
    def forward(model, data):
        return  model.forward(data[0], data[0], data[0], is_training=True)[0]
    def backward(output, data):
        return output.backward(data[1])
    run_component_benchmark("Apex Attention", model_and_data, forward, backward)

def run_attention_builtin():
    def model_and_data():
        encoder = MultiheadAttention(hidden_dim, 12, dropout=0.1).cuda()
        encoder = encoder.half()
        encoder.train()
        inputs = torch.randn(seq_len, batch_size, hidden_dim, dtype=torch.float16, device=torch.device("cuda")).requires_grad_(True)
        grads = torch.randn_like(inputs)
        return encoder, (inputs, grads)
    def forward(model, data):
        return  model.forward(data[0], data[0], data[0])[0]
    def backward(output, data):
        return output.backward(data[1])
    run_component_benchmark("Built-in Attention", model_and_data, forward, backward)

parser = argparse.ArgumentParser(description='Benchmark target')
parser.add_argument('--bench_target', type=str)
args, unk = parser.parse_known_args()
target = args.bench_target

# monkey-patch to remove the args that fairseq doesn't like
import sys
sys.argv = [sys.argv[0]] + unk

if target == 'bert': run_bert()
elif target == 'bert_nonapex': run_bert_nonapex()
elif target == 'single_transformer': run_single_transfomer()
elif target == 'single_transformer_nonapex': run_single_transfomer_nonapex()
elif target == 'attention': run_attention()
elif target == 'attention_builtin': run_attention_builtin()
else: raise Exception('benchmark target unrecognized')

