#!/bin/bash

logfile=$1
logfile_name=${logfile%.*}

echo "Processing log file $logfile"...

# sample train log line:
# | epoch 001:     10 / 31621 loss=15.180, nll_loss=15.180, ppl=37113.65, wps=132197, ups=0, wpb=127729.455, bsz=256.000, num_updates=11, lr=1.1e-07, gnorm=4.208, clip=0.000, oom=0.000, loss_scale=128.000, wall=704, train_wall=19
echo -e "num_updates\tloss\tnll_loss\tppl\twps\tups\twpb\tbsz\tlr\tgnorm\tclip\toom\tloss_scale\twall\ttrain_wall" > $logfile_name.train.tsv
grep -P '\| epoch \d\d\d:' $logfile | \
awk '{print $14, $7, $8, $9, $10, $11, $12, $13, $15, $16, $17, $18, $19, $20, $21}' | \
sed 's/\w\w*=//g' | \
sed 's/,\s*/\t/g' >> $logfile_name.train.tsv

# sample valid log line:
# | epoch 001 | valid on 'valid' subset | loss 8.185 | nll_loss 8.185 | ppl 291.06 | num_updates 10000
echo -e "num_updates\tloss\tnll_loss\tppl" > $logfile_name.valid.tsv
grep -P '\| epoch \d\d\d \| valid on' $logfile | \
awk '{print $20, $11, $14, $17}' | \
sed 's/\s\s*/\t/g' >> $logfile_name.valid.tsv
