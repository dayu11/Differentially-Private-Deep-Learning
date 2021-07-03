#!/usr/bin/env bash

# fail fast
set -e

# TODO use az to retrieve key vault connection string
# TODO use azcopy to download data

DATA_DIR=/media/data
OLDWIKI_RAW=$DATA_DIR/fastbert_pretrain_golddata/raw_data/enwiki.txt
WIKI_RAW=$DATA_DIR/fastbert_pretrain_diydata_20200426/enwiki/enwiki-20200401-pages-articles.txt
SMALLBOOK_RAW="$DATA_DIR/fastbert_pretrain_diydata_20200426/bookcorpus/books_large_p1.txt $DATA_DIR/fastbert_pretrain_diydata_20200426/bookcorpus/books_large_p2.txt"
BOOK_RAW="$DATA_DIR/fastbert_pretrain_golddata/raw_data/book_corpus_epub.txt $DATA_DIR/fastbert_pretrain_golddata/raw_data/book_corpus_txt.txt"
TEXT_RAW="$WIKI_RAW $BOOK_RAW"

# !!NOTE the BOOK_RAW corpus still has invalid utf8 codepoints (e.g. 0x3F) -- it's likely the single quote symbol.
# so, an extra step: sed "s/\x3f/'/g" bookcorpus.cleaned.txt > bookcorpus.cleaned.0x3f_to_quote.txt

cat $TEXT_RAW | \
python ../common/remove_non_utf8_chars.py | \
python ../common/precleanup_english.py | \
perl ../common/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl en | \
perl ../common/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl | \
python filter_and_cleanup_lines.py | \
python replace_patterns.py | \
# dedup, and append each line (which is a document in wiki, and a paragraph in bookcorpus) with eos token </s>
awk '!x[$0]++ { print $0, "</s>" }'> $DATA_DIR/old_corpus.cleaned.txt

g++ -O3 -std=c++17 calc_wordfreq.cpp -o calc_wordfreq
# calc word freq, replace low-freq words with <unk>
./calc_wordfreq $DATA_DIR/old_corpus.cleaned.txt $DATA_DIR/wordfreq.txt $DATA_DIR/old_corpus.filtered.txt 0.21

python split.py $DATA_DIR/old_corpus.filtered.txt $DATA_DIR/old_corpus 13088055

# the moses decoder splits our notations.
# we need to protect these tokens.
MOSESDECODER_PROTECTED_PATTERNS=$DATA_DIR/mosesdecoder_protpat
cat > $MOSESDECODER_PROTECTED_PATTERNS << EOF
<url>
<unk>
<email>
<file>
</s>
EOF

cat $DATA_DIR/old_corpus.valid.txt | \
python segment_sentence.py | \
../common/mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 1 -no-escape -l en -protected ${MOSESDECODER_PROTECTED_PATTERNS} \
> $DATA_DIR/old_corpus.valid.tok

rm $DATA_DIR/old_corpus.valid.txt

# can do it parallelly
for i in 0 1 2 3 4
do
cat $DATA_DIR/old_corpus.train.txt.${i} | \
python segment_sentence.py | \
../common/mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 1 -no-escape -l en -protected ${MOSESDECODER_PROTECTED_PATTERNS} \
> $DATA_DIR/old_corpus.train.tok.${i}

rm $DATA_DIR/old_corpus.train.txt.${i}
done

rm $DATA_DIR/corpus.train.tok ||:
for i in 0 1 2 3 4
do 
  cat $DATA_DIR/old_corpus.train.tok.${i} >> $DATA_DIR/corpus.train.tok
  rm $DATA_DIR/old_corpus.train.tok.${i}
done

mv $DATA_DIR/old_corpus.valid.tok $DATA_DIR/corpus.valid.tok

g++ -O3 ../../fastbpe/fastBPE/main.cc -pthread -o fastbpe
./fastbpe learnbpe 32640 $DATA_DIR/corpus.train.tok > $DATA_DIR/bpe-code

cat $DATA_DIR/corpus.train.tok | \
python concat_short_sentences.py | \
python ../common/length_filter_by_char.py 20 1000000 > $DATA_DIR/corpus.train.tok.tmp
./fastbpe applybpe \
  $DATA_DIR/corpus.train.tok.bpe \
  $DATA_DIR/corpus.train.tok.tmp \
  $DATA_DIR/bpe-code

cat $DATA_DIR/corpus.valid.tok | \
python concat_short_sentences.py | \
python ../common/length_filter_by_char.py 20 1000000 > $DATA_DIR/corpus.valid.tok.tmp
./fastbpe applybpe \
  $DATA_DIR/corpus.valid.tok.bpe \
  $DATA_DIR/corpus.valid.tok.tmp \
  $DATA_DIR/bpe-code

# cd ../..

python ../../preprocess.py \
--only-source \
--nwordssrc 32768 \
--trainpref $DATA_DIR/corpus.train.tok.bpe \
--validpref $DATA_DIR/corpus.valid.tok.bpe \
--destdir $DATA_DIR/data-bin/wiki_book_32768 \
--workers 24
