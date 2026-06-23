from collections import defaultdict, Counter
from multiprocessing import Pool
import regex
import os
import re
import gc
import json

# =========================================================
# regex pattern
# =========================================================
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_COMPILED = None


def init_worker():
    global PAT_COMPILED
    PAT_COMPILED = regex.compile(PAT)


def find_chunk_boundaries(input_path: str, num_chunks: int):
    filesize = os.path.getsize(input_path)
    if num_chunks <= 1:
        return [0, filesize]
    chunk_size = filesize // num_chunks
    boundaries = [0]
    with open(input_path, "rb") as f:
        for i in range(1, num_chunks):
            pos = i * chunk_size
            f.seek(pos)
            while True:
                b = f.read(1)
                if not b:
                    break
                byte = b[0]
                if (byte & 0b11000000) != 0b10000000:
                    break
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
# 核心：将文本编码为初始字节ID序列（流式）
# =========================================================
def encode_text_to_initial_ids(input_path, special_tokens, chunk_size=5 * 1024 * 1024):
    """
    流式读取文件，将文本分割为word，每个word编码为字节ID列表
    返回: list of lists of ints
    """
    all_sequences = []
    special_token_set = set(special_tokens) if special_tokens else set()

    if special_tokens:
        pattern = "(" + "|".join(
            re.escape(t) for t in sorted(special_tokens, key=len, reverse=True)
        ) + ")"

    with open(input_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            # 对齐到词边界（非chunk末尾时）
            if len(chunk) == chunk_size:
                # 回退到最后一个空白字符
                while chunk and not chunk[-1:].decode("utf-8", errors="ignore").isspace():
                    chunk = chunk[:-1]
                    if not chunk:
                        break
                if chunk:
                    f.seek(f.tell() - (chunk_size - len(chunk)))
                else:
                    f.seek(f.tell() - chunk_size + 1)
                    chunk = f.read(chunk_size)

            try:
                text = chunk.decode("utf-8", errors="ignore")
            except:
                continue

            # 分词并编码
            if special_tokens:
                parts = re.split(pattern, text)
                for part in parts:
                    if part in special_token_set:
                        continue
                    for w in regex.findall(PAT, part):
                        ids = list(w.encode("utf-8"))
                        if ids:
                            all_sequences.append(ids)
            else:
                for w in regex.findall(PAT, text):
                    ids = list(w.encode("utf-8"))
                    if ids:
                        all_sequences.append(ids)

            del text, chunk

    return all_sequences


# =========================================================
# 将序列列表保存到磁盘（JSON格式，逐行存储）
# =========================================================
def save_sequences(sequences, path):
    """保存序列到磁盘，每个序列占一行"""
    with open(path, "w") as f:
        for seq in sequences:
            f.write(json.dumps(seq) + "\n")


def load_sequences(path):
    """从磁盘加载序列"""
    sequences = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                sequences.append(json.loads(line))
    return sequences


# =========================================================
# 扫描磁盘上的序列，统计pair频率
# =========================================================
def scan_pairs_from_disk(sequences_path, batch_size=100000):
    """
    从磁盘分批读取序列，统计pair频率
    返回: pair_counts, pair_to_indices
    """
    pair_counts = defaultdict(int)
    # pair_to_indices 只存储批次号+批次内偏移，而不是全局索引
    # 格式: {pair: [(batch_idx, offset), ...]}
    pair_to_indices = defaultdict(list)

    batch_idx = 0

    with open(sequences_path, "r") as f:
        batch_sequences = []
        for line in f:
            line = line.strip()
            if line:
                batch_sequences.append(json.loads(line))

            if len(batch_sequences) >= batch_size:
                # 处理当前批次
                for offset, seq in enumerate(batch_sequences):
                    for i in range(len(seq) - 1):
                        p = (seq[i], seq[i + 1])
                        pair_counts[p] += 1
                        pair_to_indices[p].append((batch_idx, offset))

                batch_sequences = []
                batch_idx += 1
                gc.collect()

        # 处理最后一批
        if batch_sequences:
            for offset, seq in enumerate(batch_sequences):
                for i in range(len(seq) - 1):
                    p = (seq[i], seq[i + 1])
                    pair_counts[p] += 1
                    pair_to_indices[p].append((batch_idx, offset))

    return pair_counts, pair_to_indices, batch_idx + 1


# =========================================================
# 更新磁盘上的序列（应用一个merge）
# =========================================================
def apply_merge_to_disk(sequences_path, best_pair, new_id, pair_to_indices, batch_size=100000):
    """
    在磁盘上应用一个merge操作
    best_pair: (a, b)
    new_id: 新token ID
    """
    a, b = best_pair
    affected = pair_to_indices.get(best_pair, [])

    if not affected:
        return

    # 按批次分组
    batches = defaultdict(set)
    for batch_idx, offset in affected:
        batches[batch_idx].add(offset)

    # 创建临时文件
    temp_path = sequences_path + ".tmp"

    with open(sequences_path, "r") as fin, open(temp_path, "w") as fout:
        current_batch = 0
        batch_lines = []

        for line in fin:
            line = line.strip()
            if not line:
                fout.write("\n")
                continue

            seq = json.loads(line)

            if current_batch in batches and len(batch_lines) in batches[current_batch]:
                # 应用merge
                new_seq = []
                i = 0
                while i < len(seq):
                    if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                        new_seq.append(new_id)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                seq = new_seq

            batch_lines.append(seq)
            fout.write(json.dumps(seq) + "\n")

            if len(batch_lines) >= batch_size:
                batch_lines = []
                current_batch += 1

    # 替换原文件
    os.replace(temp_path, sequences_path)


# =========================================================
# 主训练函数
# =========================================================
def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens=None,
        num_processes=4,
        boundaries=None,
        sequences_cache_dir="bpe_cache"
):
    """
    基于磁盘的BPE训练，用时间换空间，完全保证merge质量
    """
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
    # 第一步：将文本编码为初始ID序列，保存到磁盘
    # =====================================================
    os.makedirs(sequences_cache_dir, exist_ok=True)
    sequences_path = os.path.join(sequences_cache_dir, "sequences.jsonl")

    print("Step 1: Encoding text to initial byte sequences...")

    # 先获取word频率（用于初始编码）
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
                re.escape(t) for t in sorted(special_tokens, key=len, reverse=True)
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
        del text
        gc.collect()

    # 保存初始序列到磁盘
    print("Saving initial sequences to disk...")
    all_sequences = [list(ids) for ids in word_freq.keys()]
    save_sequences(all_sequences, sequences_path)

    del word_freq, all_sequences
    gc.collect()

    print(f"Initial sequences saved to {sequences_path}")

    # =====================================================
    # 第二步：迭代BPE
    # =====================================================
    merges = []
    scan_interval = 100  # 每100次merge重新扫描一次（可调整）

    print("Step 2: Starting BPE iterations...")

    # 初始扫描
    pair_counts, pair_to_indices, num_batches = scan_pairs_from_disk(sequences_path)
    print(f"Initial pairs: {len(pair_counts)}, batches: {num_batches}")

    iteration = 0

    while next_id < vocab_size and pair_counts:
        if next_id % 100 == 0:
            print(f"Merge {next_id - 256}/{vocab_size - 256}, pairs: {len(pair_counts)}")

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

        # 应用merge到磁盘
        apply_merge_to_disk(sequences_path, best_pair, new_id, pair_to_indices)

        iteration += 1

        # 定期重新扫描以更新pair统计
        if iteration % scan_interval == 0:
            print(f"  Rescanning at iteration {iteration}...")
            del pair_counts, pair_to_indices
            gc.collect()
            pair_counts, pair_to_indices, num_batches = scan_pairs_from_disk(sequences_path)
        else:
            # 增量更新pair_counts（只更新受影响的pair）
            # 获取受影响的批次和偏移
            affected = pair_to_indices.get(best_pair, [])

            # 移除旧pair
            old_freq = pair_counts.pop(best_pair, 0)
            del pair_to_indices[best_pair]

            # 需要重新读取受影响的行来更新统计
            batches = defaultdict(set)
            for batch_idx, offset in affected:
                batches[batch_idx].add(offset)

            # 读取并更新
            with open(sequences_path, "r") as f:
                current_batch = 0
                batch_lines = []
                line_idx = 0

                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    seq = json.loads(line)

                    if current_batch in batches and line_idx in batches[current_batch]:
                        # 这个序列被修改了，更新pair统计
                        for i in range(len(seq) - 1):
                            p = (seq[i], seq[i + 1])
                            pair_counts[p] += 1
                            pair_to_indices[p].append((current_batch, line_idx))

                    line_idx += 1
                    if line_idx >= 100000:  # batch_size
                        current_batch += 1
                        line_idx = 0

    # 清理缓存
    import shutil
    if os.path.exists(sequences_cache_dir):
        shutil.rmtree(sequences_cache_dir)

    return vocab, merges