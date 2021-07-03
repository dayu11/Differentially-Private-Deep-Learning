set title 'Corpus word frequency distribution'
set xlabel 'Words (milli-tile)'
set ylabel 'Frequency'
set xrange [0:1000]
set yrange [1:100000]
set logscale y
set parametric
plot 'train.unigram.stats.tsv' using 1:3:2 with labels point pt 3 rotate left font ',2'
set trange [1:100000]
replot 8, t
replot 30, t
replot 150, t
replot 465, t
