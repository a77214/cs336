from collections import defaultdict, Counter
from multiprocessing import Pool
import regex
import os
import re

# =========================================================
# regex pattern
# =========================================================
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_COMPILED = None


# =========================================================
# worker init
# =========================================================
def init_worker():
    global PAT_COMPILED
    PAT_COMPILED = regex.compile(PAT)


# =========================================================
# find chunk boundaries
# =========================================================
def find_chunk_boundaries(input_path: str, num_chunks: int):
    """返回: [0, b1, b2, ..., filesize]"""
    filesize = os.path.getsize(input_path)
    if num_chunks <= 1:
        return [0, filesize]
    chunk_size = filesize // num_chunks
    boundaries = [0]
    with open(input_path, "rb") as f:
        for i in range(1, num_chunks):
            pos = i * chunk_size
            f.seek(pos)
            # UTF8 对齐
            while True:
                b = f.read(1)
                if not b:
                    break
                byte = b[0]
                if (byte & 0b11000000) != 0b10000000:
                    break
            # 找到下一个空白字符
            while True:
                pos_now = f.tell()
                b = f.read(1)
                if not b:
                    break
                try:
                    ch = b.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                if ch.isspace():
                    boundaries.append(pos_now)
                    break
            else:
                boundaries.append(f.tell())
        boundaries.append(filesize)
    return boundaries


# =========================================================
# chunk processor
# =========================================================
def process_chunk(args):
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    counter = Counter()
    if special_tokens:
        special_token_set = set(special_tokens)
        pattern = (
                "(" +
                "|".join(
                    re.escape(t)
                    for t in sorted(special_tokens, key=len, reverse=True)
                )
                + ")"
        )
        parts = re.split(pattern, text)
        for part in parts:
            if part in special_token_set:
                continue
            for w in PAT_COMPILED.findall(part):
                counter[tuple(w.encode("utf-8"))] += 1
    else:
        for w in PAT_COMPILED.findall(text):
            counter[tuple(w.encode("utf-8"))] += 1
    return counter


# =========================================================
# multiprocessing word freq builder
# =========================================================
def build_parallel_word_freq(input_path, boundaries, special_tokens, num_processes):
    tasks = [
        (input_path, s, e, special_tokens)
        for s, e in zip(boundaries[:-1], boundaries[1:])
    ]
    merged = Counter()
    with Pool(processes=num_processes, initializer=init_worker) as pool:
        for c in pool.map(process_chunk, tasks):
            merged.update(c)
    return merged


# =========================================================
# build pair stats
# =========================================================
def build_pair_stats(sequences, freqs):
    """高效构建pair统计"""
    pair_counts = defaultdict(int)
    pair_to_indices = defaultdict(set)

    for idx, (seq, freq) in enumerate(zip(sequences, freqs)):
        for i in range(len(seq) - 1):
            p = (seq[i], seq[i + 1])
            pair_counts[p] += freq
            pair_to_indices[p].add(idx)

    return pair_counts, pair_to_indices


# =========================================================
# train bpe (memory efficient)
# =========================================================
def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens=None,
        num_processes=4,
        boundaries=None
):
    special_tokens = special_tokens or []

    # =====================================================
    # vocab init
    # =====================================================
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    # =====================================================
    # word freq
    # =====================================================
    if num_processes > 1:
        if boundaries is None:
            boundaries = find_chunk_boundaries(input_path, num_processes)
        word_freq = build_parallel_word_freq(input_path, boundaries, special_tokens, num_processes)
    else:
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        word_freq = Counter()
        if special_tokens:
            special_token_set = set(special_tokens)
            pattern = "(" + "|".join(
                re.escape(t)
                for t in sorted(special_tokens, key=len, reverse=True)
            ) + ")"
            parts = re.split(pattern, text)
            for part in parts:
                if part in special_token_set:
                    continue
                for w in regex.findall(PAT, part):
                    word_freq[tuple(w.encode("utf-8"))] += 1
        else:
            for w in regex.findall(PAT, text):
                word_freq[tuple(w.encode("utf-8"))] += 1

    if not word_freq:
        return vocab, []

    # =====================================================
    # 转换为可变列表
    # =====================================================
    word_sequences = []
    word_freqs = []

    for ids_tuple, freq in word_freq.items():
        word_sequences.append(list(ids_tuple))
        word_freqs.append(freq)

    # 清理不再需要的word_freq
    del word_freq

    # 构建初始pair统计
    pair_counts, pair_to_indices = build_pair_stats(word_sequences, word_freqs)

    merges = []

    # =====================================================
    # main loop
    # =====================================================
    while next_id < vocab_size and pair_counts:
        # 找最佳pair
        best_pair = max(
            pair_counts.items(),
            key=lambda x: (x[1], vocab[x[0][0]], vocab[x[0][1]])
        )[0]

        a, b = best_pair
        new_id = next_id
        next_id += 1
        vocab[new_id] = vocab[a] + vocab[b]
        merges.append((vocab[a], vocab[b]))

        # 获取受影响的序列索引
        affected_indices = pair_to_indices.get(best_pair, set())

        # 清理旧pair
        pair_to_indices.pop(best_pair, None)
        pair_counts.pop(best_pair, None)

        # 增量更新
        delta = defaultdict(int)

        for idx in affected_indices:
            old_seq = word_sequences[idx]
            freq = word_freqs[idx]

            # 减去旧pairs
            for i in range(len(old_seq) - 1):
                p = (old_seq[i], old_seq[i + 1])
                delta[p] -= freq
                if p in pair_to_indices:
                    pair_to_indices[p].discard(idx)
                    if not pair_to_indices[p]:
                        del pair_to_indices[p]

            # 合并序列
            new_seq = []
            i = 0
            while i < len(old_seq):
                if (i + 1 < len(old_seq) and
                        old_seq[i] == a and old_seq[i + 1] == b):
                    new_seq.append(new_id)
                    i += 2
                else:
                    new_seq.append(old_seq[i])
                    i += 1

            word_sequences[idx] = new_seq

            # 加上新pairs
            for i in range(len(new_seq) - 1):
                p = (new_seq[i], new_seq[i + 1])
                delta[p] += freq
                pair_to_indices[p].add(idx)

        # 批量更新pair_counts
        for p, d in delta.items():
            if p not in pair_counts:
                pair_counts[p] = 0
            pair_counts[p] += d
            if pair_counts[p] <= 0:
                pair_counts.pop(p, None)

    return vocab, merges