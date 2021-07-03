#pragma once

#include <unordered_map>
#include <vector>
#include <cassert>
#include <iostream>
#include <functional>
#include <thread>

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>


using namespace std;

enum dtype_t {
  DT_UINT8 = 1,
  DT_INT8 = 2,
  DT_INT16 = 3,
  DT_INT32 = 4,
  DT_INT64 = 5,
  DT_FLOAT = 6,
  DT_DOUBLE = 7,
  DT_UINT16 = 8,
};

struct __attribute__((packed)) index_file_t {
  char magic_header[9];
  uint64_t version;
  uint8_t dtype_code;
  uint64_t data_cnt;
  uint32_t sizes[0];
  uint64_t offsets[0];
};

struct dict_entry_t {
  string token;
  bool terminal;

  dict_entry_t(const char *token, bool t) : token(token), terminal(t) {}
};

using wordfreq_t = vector<pair<string, uint64_t>>;
using wordmap_t = unordered_map<string, uint64_t>;

struct graph_t {
  unordered_map<uint64_t, vector<uint64_t>> G;

  void insert(uint64_t v, uint64_t u) {
    auto it = G.find(v);
    if (it == G.end()) {
      G[v] = {u};
    } else {
      G[v].push_back(u);
    }
  }

  vector<uint64_t> &get(uint64_t v) {
    assert(false);
  }

  void clear() { G.clear(); }

  void merge(graph_t &other) {
    for (const auto &[wid, svec] : other.G) {
      const auto &[it, ins] = G.insert({wid, svec});
      if (!ins) {
        it->second.insert(it->second.end(), svec.begin(), svec.end());
      }
    }
  }
};

struct dataset_t {
  // static data loaded from disk
  uint64_t cnt;
  uint32_t *psizes;
  uint64_t *poffsets;
  uint8_t *pbin;
  vector<dict_entry_t> dict;
  // configuration data
  const char *fprefix;
  uint64_t freq_low;
  uint64_t freq_hi;
  // processed runtime data
  uint64_t unigram_filter_first;
  uint64_t unigram_filter_last;  // inclusive
  wordfreq_t unigram_freq;
  wordmap_t unigram_map;
  graph_t graph;
};

void die(const char *msg) {
  perror(msg);
  exit(EXIT_FAILURE);
}

vector<dict_entry_t> load_dict(const char *dict_path) {
  vector<dict_entry_t> ret;
  FILE *fp = fopen(dict_path, "r");
  if (!fp) {
    die("load_dict");
  }
  char word_buf[128];
  uint64_t freq_buf;

  ret.emplace_back("<s>", true);
  ret.emplace_back("<pad>", true);
  ret.emplace_back("</s>", true);
  ret.emplace_back("<unk>", true);
  while (2 == fscanf(fp, "%s %lu", word_buf, &freq_buf)) {
    int len = strlen(word_buf);
    bool cont = len > 2 && word_buf[len - 1] == '@' && word_buf[len - 2] == '@';
    if (cont) {
      word_buf[len - 2] = '\0';
    }
    ret.emplace_back(word_buf, !cont);
  }
  fclose(fp);
  cerr << "load_dict OK." << endl;
  return ret;
}

template<typename T>
void work_parallel(
    const dataset_t &data, vector<T> &thread_results,
    std::function<void(uint64_t, uint16_t *, uint32_t, T &)> proc) {
  int nthread = thread_results.size();
  vector<thread> threads;

  for (int tidx = 0; tidx < nthread; ++tidx) {
    threads.emplace_back(
        [&](int thread_id) {
          const auto work_from = data.cnt / nthread * thread_id;
          const auto work_to = (thread_id == nthread - 1)
                                   ? data.cnt
                                   : data.cnt / nthread * (thread_id + 1);

          this_thread::sleep_for(chrono::milliseconds(thread_id * 50));
          cerr << "thread #" << thread_id << " starting, working from "
               << work_from << " to " << work_to << endl;

          for (auto i = work_from; i < work_to; ++i) {
            uint16_t *pdata =
                reinterpret_cast<uint16_t *>(data.pbin + data.poffsets[i]);
            auto ntoken = data.psizes[i];
            proc(i, pdata, ntoken, thread_results[thread_id]);
          }

          cerr << "thread #" << thread_id << " finished." << endl;
        },
        tidx);
  }

  for (int tidx = 0; tidx < nthread; ++tidx) {
    threads[tidx].join();
  }
}
