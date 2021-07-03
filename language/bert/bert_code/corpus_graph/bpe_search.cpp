#include "common.h"

int search(const char *buf, const vector<dict_entry_t> &dict) {
  int result = 0;
  auto blen = strlen(buf);

  for (const auto &entry : dict) {
    auto len = entry.token.length();
    if (len > blen || (len == blen && !entry.terminal) || (len != blen && entry.terminal)) {
      continue;
    }
    if (strncmp(buf, entry.token.data(), len)) {
      continue;
    }
    // this token is O.K. for BPE construction.
    if (len == blen) {
      ++result;
    } else {
      result += search(buf+len, dict);
    }
  }

  return result;
}

int main() {
  FILE *unigram = fopen("train.unigram.txt", "r");
  FILE *out = fopen("unigram.bpe.search.txt", "w");

  char buf[6000];
  int freq;

  auto dict = load_dict("/home/yatli/data/wiki_book_32768/dict.txt");

  while (2 == fscanf(unigram, "%s %d", buf, &freq)) {
    int search_result = search(buf, dict);
    fprintf(out, "%s %d %d\n", buf, freq, search_result);
  }

  fclose(unigram);
  fclose(out);
  return 0;
}
