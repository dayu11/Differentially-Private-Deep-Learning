#include <algorithm>
#include <functional>
#include <iostream>
#include <set>
#include <string>
#include <thread>

#include "common.h"

using namespace std;

// intermediate working set data used in parallel processing
struct result_t {
  uint64_t sumlen = 0;
  uint64_t sumn = 0;
  uint64_t sumn_0 = 0;
  uint64_t sumn_1 = 0;
  uint64_t sumn_5 = 0;
  uint64_t sumn_20 = 0;
  uint64_t sumn_100 = 0;
};

void compute_sentence_len(dataset_t& data) {
  vector<result_t> results(8);
  work_parallel<result_t>(data, results, [](auto idx, auto* pdata, auto ntok, result_t& result) {
      ++result.sumn;
      result.sumlen += ntok;
      if(ntok == 0) {
        ++result.sumn_0;
      } else if (ntok == 1) {
        ++result.sumn_1;
      } else if (ntok <= 5) {
        ++result.sumn_5;
      } else if (ntok <= 20) {
        ++result.sumn_20;
      } else if (ntok <= 100) {
        ++result.sumn_100;
      }
  });

  result_t total;
  for(auto &partial: results) {
    total.sumlen += partial.sumlen;
    total.sumn += partial.sumn;
    total.sumn_0 += partial.sumn_0;
    total.sumn_1 += partial.sumn_1;
    total.sumn_5 += partial.sumn_5;
    total.sumn_20 += partial.sumn_20;
    total.sumn_100 += partial.sumn_100;
  }

  cout 
    << "Total nr. tokens               = " << total.sumlen << endl
    << "Total nr. sentences            = " << total.sumn << endl
    << "Total nr. sentences (empty)    = " << total.sumn_0 << endl
    << "Total nr. sentences (1tok)     = " << total.sumn_1 << endl
    << "Total nr. sentences (<=5tok)   = " << total.sumn_5 << endl
    << "Total nr. sentences (<=20tok)  = " << total.sumn_20 << endl
    << "Total nr. sentences (<=100tok) = " << total.sumn_100 << endl
    << "Average nr.tokens per sent.    = " << total.sumlen / double(total.sumn) << endl
  ;
}

int main(int argc, const char *argv[]) {
  if (argc != 4) {
    exit(-1);
  }

  const char *dic_fpath = argv[1];
  const char *idx_fpath = argv[2];
  const char *bin_fpath = argv[3];
  const char *fprefix = argv[4];

  cerr << "dict file: " << dic_fpath << endl
       << "index file: " << idx_fpath << endl
       << "data file: " << bin_fpath << endl;

  int fd_idx, fd_bin;
  struct stat64 stat_idx, stat_bin;
  void *pidx, *pbin;
  auto dict = load_dict(dic_fpath);

  if (-1 == (fd_idx = open(idx_fpath, O_RDONLY | O_LARGEFILE))) {
    die("fd_idx");
  }
  if (-1 == (fd_bin = open(bin_fpath, O_RDONLY | O_LARGEFILE))) {
    die("fd_bin");
  }
  if (fstat64(fd_idx, &stat_idx)) {
    die("stat_idx");
  }
  if (fstat64(fd_bin, &stat_bin)) {
    die("stat_bin");
  }

  if (nullptr == (pidx = mmap(nullptr, stat_idx.st_size, PROT_READ, MAP_PRIVATE,
                              fd_idx, 0))) {
    die("pidx");
  }
  if (nullptr == (pbin = mmap(nullptr, stat_bin.st_size, PROT_READ, MAP_PRIVATE,
                              fd_bin, 0))) {
    die("pbin");
  }

  index_file_t *pindex_file = static_cast<index_file_t *>(pidx);

  assert(!strcmp(pindex_file->magic_header, "MMIDIDX"));
  assert(pindex_file->version == 1);
  assert(pindex_file->dtype_code == DT_UINT16);

  cerr << "index file length: " << stat_idx.st_size << endl
       << "data file length: " << stat_bin.st_size << endl
       << "index version: " << pindex_file->version << endl
       << "index count: " << pindex_file->data_cnt << endl
       << "index data type: " << (int)pindex_file->dtype_code << endl;

  uint64_t freq_low = 10;
  uint64_t freq_hi = 100000;

  dataset_t dataset{
      pindex_file->data_cnt,
      pindex_file->sizes,
      reinterpret_cast<uint64_t *>(pindex_file->sizes + pindex_file->data_cnt),
      reinterpret_cast<uint8_t *>(pbin),
      std::move(dict),
      fprefix,
      freq_low,
      freq_hi,
  };

  compute_sentence_len(dataset);

  close(fd_bin);
  close(fd_idx);
  return EXIT_SUCCESS;
}
