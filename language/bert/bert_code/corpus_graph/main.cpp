#include <algorithm>
#include <functional>
#include <iostream>
#include <set>
#include <string>
#include <thread>

#include "common.h"

using namespace std;

constexpr auto c_unigram_file = ".unigram.txt";
constexpr auto c_graph_file = ".graph.bin";

// intermediate working set data used in parallel processing
struct result_t {
  wordmap_t unigram_freq;
  graph_t graph;
};

void usage(const char *prg) {
  cerr << prg << ": construct corpus graph." << endl
       << endl
       << "usage:" << endl
       << prg << " dic_file idx_file bin_file prefix" << endl
       << endl
       << prg << "The prefix will be used to generate the output files."
       << endl;
  exit(EXIT_FAILURE);
}

bool load_unigram(dataset_t &data) {
  string unigram_fname = data.fprefix;
  unigram_fname.append(c_unigram_file);
  FILE *fp = fopen(unigram_fname.data(), "r");
  bool ok = (fp != nullptr);
  // yeah, there are lines as long as 5551 chars.. wc -L and you'll see.
  char wordbuf[6000];
  uint64_t freqbuf;
  data.unigram_freq.clear();
  if (ok) {
    while (2 == fscanf(fp, "%s %lu", wordbuf, &freqbuf)) {
      data.unigram_freq.emplace_back(wordbuf, freqbuf);
    }
    cerr << "unigram file loaded." << endl;
  } else {
    cerr << "unigram file not exist." << endl;
  }
  if (fp) {
    fclose(fp);
  }
  return ok;
}

void process_unigram(dataset_t &data, vector<result_t> &thread_results) {
  // unigram
  work_parallel<result_t>(
      data, thread_results,
      [&](auto _, auto *pdata, auto ntoken, auto &result) {
        string curword;
        for (uint32_t j = 0; j < ntoken; ++j) {
          const auto &entry = data.dict[pdata[j]];
          curword.append(entry.token);
          if (entry.terminal) {
            const auto &[iter, ins] = result.unigram_freq.insert({curword, 1});
            if (!ins) {
              ++iter->second;
            }
            curword.clear();
          }
        }
      });

  wordmap_t unigram_freq;

  cerr << "aggregating results" << endl;

  for (auto &partial : thread_results) {
    // merge unigram
    for (const auto &[k, v] : partial.unigram_freq) {
      const auto &[iter, ins] = unigram_freq.insert({k, v});
      if (!ins) {
        iter->second += v;
      }
    }
    partial.unigram_freq.clear();
  }

  cerr << "dumping results to a vec..." << endl;
  auto compFunctor = [](const pair<string, uint64_t> &a,
                        const pair<string, uint64_t> &b) {
    return a.second > b.second || (a.second == b.second && a.first < b.first);
  };

  set<pair<string, uint64_t>, decltype(compFunctor)> sorted_unigram_freq(
      unigram_freq.begin(), unigram_freq.end(), compFunctor);
  unigram_freq.clear();
  data.unigram_freq =
      wordfreq_t(sorted_unigram_freq.begin(), sorted_unigram_freq.end());
  sorted_unigram_freq.clear();

  string unigram_fname = data.fprefix;
  unigram_fname.append(c_unigram_file);
  FILE *fp = fopen(unigram_fname.data(), "w");
  if (!fp) {
    die("unigram write file");
  }

  cerr << "saving unigram data..." << endl;
  for (const auto &[k, v] : data.unigram_freq) {
    fprintf(fp, "%s %lu\n", k.data(), v);
  }
  fclose(fp);
}

void unigram_stats(wordfreq_t &unigram, const dataset_t &dataset) {
  string stats_name = dataset.fprefix;
  stats_name.append(".unigram.stats.tsv");
  FILE *fp = fopen(stats_name.data(), "w");
  if (!fp) {
    die("unigram_stats");
  }

  int percentile = -1;
  for (int i = 0; i < unigram.size(); ++i) {
    int cur_percentile = i * 1000 / unigram.size();
    if (cur_percentile != percentile) {
      percentile = cur_percentile;
      fprintf(fp, "%d\t%s\t%lu\n", percentile, unigram[i].first.data(),
              unigram[i].second);
    }
  }

  fclose(fp);
  cerr << "unigram stats written to " << stats_name << endl;
}

bool load_graph(dataset_t &data) {
  string unigram_fname = data.fprefix;
  unigram_fname.append(c_graph_file);
  FILE *fp = fopen(unigram_fname.data(), "rb");
  bool ok = (fp != nullptr);
  uint64_t buf[4096];
  graph_t graph;
  constexpr auto maxedges = sizeof(buf) / sizeof(uint64_t) / 2;
  if (ok) {
    cerr << "loading corpus graph..." << endl;
    while (true) {
      auto readedge = fread(buf, sizeof(uint64_t) * 2, maxedges, fp);
      if (readedge < maxedges) {
        break;
      }
      for (auto i = 0; i < readedge; ++i) {
        graph.insert(buf[i * 2], buf[i * 2 + 1]);
      }
    }
    cerr << "graph file loaded." << endl;
  } else {
    cerr << "graph file not exist." << endl;
  }
  if (fp) {
    fclose(fp);
  }
  return ok;
}

void process_graph(dataset_t &data, vector<result_t> &thread_results) {
  cerr << "processing corpus graph..." << endl;
  work_parallel<result_t>(data, thread_results,
                [&](auto sentence_id, auto *pdata, auto ntoken, auto &result) {
                  string curword;
                  set<string> processed_words;
                  for (uint32_t j = 0; j < ntoken; ++j) {
                    const auto &entry = data.dict[pdata[j]];
                    curword.append(entry.token);
                    if (entry.terminal) {
                      // the word won't be present in the map if it's filtered
                      auto wordit = data.unigram_map.find(curword);
                      if (wordit != data.unigram_map.end()) {
                        // only insert the first occurrence of a word in a sentence
                        const auto& [it, ins] = processed_words.insert(curword);
                        if(ins) {
                          result.graph.insert(wordit->second, sentence_id);
                        }
                      }
                      curword.clear();
                    }
                  }
                });

  data.graph.clear();
  cerr << "aggregating results..." << endl;

  for (auto &partial : thread_results) {
    // merge graph
    data.graph.merge(partial.graph);
    partial.graph.clear();
  }

  string fname = data.fprefix;
  fname.append(c_graph_file);
  FILE *fp = fopen(fname.data(), "wb");
  if (!fp) {
    die("unigram write file");
  }

  cerr << "saving graph data..." << endl;
  uint64_t buf[2];
  for (const auto &[wid, svec] : data.graph.G) {
    for (const auto sid : svec) {
      buf[0] = wid;
      buf[1] = sid;
      fwrite(buf, sizeof(buf), 1, fp);
    }
  }
  fclose(fp);
}

void filter_unigram(dataset_t &data) {
  wordfreq_t filtered;
  uint64_t idx = 0;
  bool first = true;
  for (const auto &[word, freq] : data.unigram_freq) {
    if (freq < data.freq_hi && freq > data.freq_low) {
      filtered.push_back({word, freq});
      if (first) {
        first = false;
        data.unigram_filter_first = idx;
      } else {
        data.unigram_filter_last = idx;
      }
    }
    ++idx;
  }
  cerr << "filter_unigram: reduced unigram size from "
       << data.unigram_freq.size() << " to " << filtered.size() << endl;
  data.unigram_freq = filtered;
}

void process_ngram(dataset_t &data) {

  int nthread = thread::hardware_concurrency();
  vector<result_t> thread_results(nthread);

  cerr << "parallelism capacity: " << nthread << endl;

  bool ok = false;
  if (!load_unigram(data)) {
    process_unigram(data, thread_results);
  }

  filter_unigram(data);

  // unigram_stats(unigram_freq, data);
  data.unigram_map.clear();
  uint64_t unigram_id = data.unigram_filter_first;
  for (const auto &[k, _] : data.unigram_freq) {
    data.unigram_map[k] = unigram_id++;
  }

  if (!load_graph(data)) {
    process_graph(data, thread_results);
  }

  uint64_t edge_count = 0;
  for(const auto& [w, svec]: data.graph.G) {
    edge_count += svec.size();
  }

  cerr
      << "unigram max freq = " << data.unigram_freq.front().second << endl
      << "unigram min freq = " << data.unigram_freq.back().second << endl
      << "graph edge count = " << edge_count << endl
  ;
}

/// Input: a sequence of subword tokens.
///
void query_graph(vector<uint64_t> tokens) {}

int main(int argc, const char *argv[]) {
  if (argc != 5) {
    usage(argv[0]);
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

  process_ngram(dataset);

  close(fd_bin);
  close(fd_idx);
  return EXIT_SUCCESS;
}
