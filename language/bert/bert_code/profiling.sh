#!/bin/bash

export PATH=/usr/local/cuda/bin:$PATH
targets="bert bert_nonapex single_transformer single_transformer_nonapex attention attention_builtin"
for t in $targets
do
  echo "Collecting profiling data for target $t..."
  nvprof -f -o profiling/$t.sql --profile-from-start off -- python profiling.py -a mcbert data --bench_target $t
  python pyprof2/pyprof2/parse/parse.py profiling/$t.sql > profiling/$t.dict
  python pyprof2/pyprof2/prof/prof.py --csv profiling/$t.dict > profiling/$t.csv
done
