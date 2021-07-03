# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('mcbert')
class McbertLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
        mask_idx = self.task.dictionary.index('<mask>')
        masked_tokens = sample['net_input']['src_tokens'].eq(mask_idx)
        
        not_pad_tokens = sample['target'].ne(self.padding_idx)
        
        sample_size = masked_tokens.int().sum().item()

        # (Rare case) When all tokens are masked, the model results in empty
        # tensor and gives CUDA error.
        if sample_size == 0:
            masked_tokens = None
        if masked_tokens is not None:
            gen_logits, nota_output, kmlm_output, replace_tokens, _ = model(
                **sample['net_input'],
                masked_tokens=masked_tokens,
                targets=sample['target']
            )
            targets = model.get_targets(sample, [gen_logits])

            targets = targets[masked_tokens]

            gen_loss = F.nll_loss(
                F.log_softmax(
                    gen_logits.view(-1, gen_logits.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                ),
                targets.view(-1),
                reduction='sum',
                ignore_index=self.padding_idx,
            )
            nota_target = (~replace_tokens).float()
            nota_loss = F.binary_cross_entropy_with_logits(nota_output.float().view(-1),
                nota_target.view(-1), reduction='mean')
            if kmlm_output is not None:
                kmlm_loss = F.nll_loss(
                    F.log_softmax(
                        kmlm_output,
                        dim=-1,
                        dtype=torch.float32,
                    ),
                    torch.zeros(kmlm_output.size(0), dtype=torch.long, device=kmlm_output.device, requires_grad=False),
                    reduction='mean',
                )
            else:
                kmlm_loss = torch.zeros(1, dtype=torch.float, device=sample['target'].device, requires_grad=True)

            loss = gen_loss + self.args.nota_loss_weight * nota_loss * sample_size \
                + self.args.mlm_loss_weight * kmlm_loss * sample_size
            logging_output = {
                'loss': utils.item(loss.data) if reduce else loss.data,
                'nll_loss': utils.item(gen_loss.data) if reduce else gen_loss.data,
                'ntokens': sample['ntokens'],
                'nsentences': sample['nsentences'],
                'sample_size': sample_size
            }
            logging_output.update(
                nota_loss=(nota_loss * not_pad_tokens.sum()).item() 
            )
            logging_output.update(
                kmlm_loss=(kmlm_loss * replace_tokens.sum()).item()
            )
            logging_output.update(
                replace_rate=replace_tokens.sum().item()
            )
        else:
            print("Skip the empty batch")
            loss = torch.zeros(1, dtype=torch.float, device=sample['target'].device, requires_grad=True)
            logging_output = {
                'loss': 0,
                'nll_loss': 0,
                'ntokens': sample['ntokens'],
                'nsentences': sample['nsentences'],
                'sample_size': sample_size
            }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        nota_loss = sum(log.get('nota_loss', 0) for log in logging_outputs)
        kmlm_loss = sum(log.get('kmlm_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        replace_tokens = sum(log.get('replace_rate', 0) for log in logging_outputs)
        sample_size = max(sample_size, 1e-10)
        replace_tokens = max(replace_tokens, 1e-10)
        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / sample_size / math.log(2) if ntokens > 0 else 0.,
            'nota_loss': nota_loss / ntokens,
            'kmlm_loss': kmlm_loss / replace_tokens,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'replace_rate': replace_tokens / ntokens,
        }
        
        return agg_output
