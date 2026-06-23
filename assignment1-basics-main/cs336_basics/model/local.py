# step2_local.py
import pickle
import time
import os
from collections import defaultdict


def merge_word(ids, pair, new_id):
    a, b = pair
    out = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)


def train_bpe_from_word_freq(word_freq_path, vocab_size, special_tokens=None):
    special_tokens = special_tokens or []

    # 加载
    print(f"Loading {word_freq_path}...")
    with open(word_freq_path, "rb") as f:
        word_freq = pickle.load(f)
    print(f"Loaded {len(word_freq):,} unique words")

    # vocab init
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    # words
    words = list(word_freq.items())
    del word_freq

    pair_counts = defaultdict(int)
    pair_index = defaultdict(set)

    print("Building initial pair statistics...")
    for idx, (ids, freq) in enumerate(words):
        for i in range(len(ids) - 1):
            p = (ids[i], ids[i + 1])
            pair_counts[p] += freq
            pair_index[p].add(idx)
    print(f"Initial pairs: {len(pair_counts):,}")

    # main loop
    merges = []
    vocab_cache = {}

    total_merges = vocab_size - next_id
    start_time = time.time()

    while next_id < vocab_size and pair_counts:
        # =====================================================
        # 尾部批量优化：最后10%，每次处理200个pair
        # =====================================================
        if next_id > vocab_size * 0.9:
            batch_size = min(200, len(pair_counts), vocab_size - next_id)
            if batch_size > 1:
                sorted_pairs = sorted(
                    pair_counts.items(),
                    key=lambda x: x[1], reverse=True
                )[:batch_size]

                for (a, b), _ in sorted_pairs:
                    if next_id >= vocab_size:
                        break
                    new_id = next_id
                    next_id += 1
                    vocab[new_id] = vocab[a] + vocab[b]
                    merges.append((vocab[a], vocab[b]))

                    if (a, b) in pair_index:
                        for idx in pair_index[(a, b)]:
                            ids, freq = words[idx]
                            words[idx] = (merge_word(ids, (a, b), new_id), freq)

                # 重建索引
                pair_counts.clear()
                pair_index.clear()
                for idx, (ids, freq) in enumerate(words):
                    for i in range(len(ids) - 1):
                        p = (ids[i], ids[i + 1])
                        pair_counts[p] += freq
                        pair_index[p].add(idx)

                elapsed = time.time() - start_time
                progress = (next_id - 256) / total_merges * 100
                print(f"[BATCH] Progress: {progress:.1f}% ({next_id}/{vocab_size}), "
                      f"Elapsed: {elapsed:.0f}s")
                continue

        # =====================================================
        # 前期：你原来的优化逻辑
        # =====================================================
        best_pair = max(pair_counts.items(),
                        key=lambda x: (x[1],
                                       vocab_cache.setdefault(x[0][0], vocab[x[0][0]]),
                                       vocab_cache.setdefault(x[0][1], vocab[x[0][1]]))
                        )[0]

        a, b = best_pair
        new_id = next_id
        next_id += 1
        vocab[new_id] = vocab[a] + vocab[b]
        merges.append((vocab[a], vocab[b]))

        affected = list(pair_index[best_pair])
        pair_index.pop(best_pair, None)
        pair_counts.pop(best_pair, None)

        delta = defaultdict(int)

        for idx in affected:
            old_ids, freq = words[idx]
            new_ids = merge_word(old_ids, best_pair, new_id)
            words[idx] = (new_ids, freq)

            for i in range(len(old_ids) - 1):
                p = (old_ids[i], old_ids[i + 1])
                delta[p] -= freq
                pair_index[p].discard(idx)

            for i in range(len(new_ids) - 1):
                p = (new_ids[i], new_ids[i + 1])
                delta[p] += freq
                pair_index[p].add(idx)

        for p, d in delta.items():
            if d == 0:
                continue
            if p not in pair_counts:
                if d > 0:
                    pair_counts[p] = d
            else:
                pair_counts[p] += d
                if pair_counts[p] <= 0:
                    pair_counts.pop(p, None)
                    pair_index.pop(p, None)

        if next_id % 500 == 0:
            empty = [p for p, s in pair_index.items() if not s]
            for p in empty:
                pair_index.pop(p, None)
                pair_counts.pop(p, None)

            elapsed = time.time() - start_time
            progress = (next_id - 256) / total_merges * 100
            eta = elapsed / max(next_id - 256, 1) * total_merges - elapsed
            print(f"Progress: {progress:.1f}% ({next_id}/{vocab_size}), "
                  f"Elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.0f}s ({total_time / 60:.1f} min)")

    return vocab, merges


def main():
    word_freq_path = "workspace/owt_word_freq.pkl"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]

    vocab, merges = train_bpe_from_word_freq(
        word_freq_path=word_freq_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )

    # 保存
    with open("workspace/owt_bpe_vocab_32000.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("workspace/owt_bpe_merges_32000.pkl", "wb") as f:
        pickle.dump(merges, f)

    longest_id, longest_bytes = max(vocab.items(), key=lambda kv: len(kv[1]))
    longest_str = longest_bytes.decode("utf-8", errors="replace")

    print(f"\nSaved vocab and merges")
    print(f"Vocab size: {len(vocab)}, Merges: {len(merges)}")
    print(f"Longest token id={longest_id}, bytes_len={len(longest_bytes)}")
    print(f"Longest token: {repr(longest_str)}")


if __name__ == "__main__":
    main()