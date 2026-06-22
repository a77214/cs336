import os
import time
import pickle
import psutil
from collections import Counter
from multiprocessing import Pool
from typing import Optional
from cs336_basics.model.train_bpe import train_bpe, build_word_freq_serial


def find_chunk_boundaries(file, num_chunks: int, split_token: bytes):
    """
    查找按特殊token边界对齐的chunk边界。
    返回字节偏移列表 [0, ... , file_size]。
    """
    file.seek(0, os.SEEK_END)
    file_size = file.tell()

    if num_chunks <= 1:
        return [0, file_size]

    # 估算每个chunk的大小
    chunk_size = file_size // num_chunks

    # 定位到每个chunk的近似结束位置，然后调整到最近的特殊token
    boundaries = [0]
    for i in range(1, num_chunks):
        pos = i * chunk_size
        file.seek(pos)

        # 向前读取，找到下一个特殊token
        # 分块读取以处理大文件
        buffer_size = 8192
        found = False
        while pos < file_size:
            file.seek(pos)
            data = file.read(min(buffer_size, file_size - pos))
            if not data:
                break

            idx = data.find(split_token)
            if idx != -1:
                pos += idx + len(split_token)
                found = True
                break
            pos += len(data)

        # 如果没找到特殊token，使用近似位置
        if not found:
            pos = i * chunk_size

        boundaries.append(pos)

    boundaries.append(file_size)
    return boundaries


def init_worker():
    """初始化worker进程（此用例中为空操作）。"""
    pass


def process_chunk(args):
    """
    处理一个文本chunk并返回词频统计。
    """
    input_path, start, end, special_tokens = args
    return build_word_freq_serial(input_path, special_tokens, start=start, end=end)


def build_word_freq_parallel(
        input_path: str | os.PathLike,
        special_tokens: list[str],
        num_processes: int,
        *,
        num_chunks: int | None = None
) -> dict[tuple[bytes, ...], int]:
    """
    使用多进程构建词频统计。
    Chunk边界对齐到特殊token边界。
    """
    if num_processes <= 1 or not special_tokens:
        return build_word_freq_serial(input_path, special_tokens)

    if num_chunks is None:
        num_chunks = max(num_processes * 32, num_processes)

    split_special_token = special_tokens[0].encode("utf-8")  # 例如 b"<|endoftext|>"
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, split_special_token)

    tasks = [(str(input_path), s, e, special_tokens) for s, e in zip(boundaries[:-1], boundaries[1:])]

    merged = Counter()
    with Pool(processes=num_processes, initializer=init_worker, maxtasksperchild=8) as pool:
        for partial in pool.imap_unordered(process_chunk, tasks, chunksize=1):
            merged.update(partial)

    return dict(merged)


def main():
    input_path = "data/owt_train.txt"
    output_dir = "workspace"
    os.makedirs(output_dir, exist_ok=True)

    vocab_size = 32_000
    special_tokens = ["<|endoftext|>"]
    num_processes = 4

    # 并行构建词频统计
    print("正在构建词频统计...")
    word_freq = build_word_freq_parallel(
        input_path,
        special_tokens,
        num_processes,
        num_chunks=num_processes * 32
    )
    print(f"发现 {len(word_freq)} 个独特的词类型")

    proc = psutil.Process(os.getpid())

    t0 = time.perf_counter()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_processes=num_processes,
        word_freq=word_freq  # 传入预先计算好的词频
    )
    t1 = time.perf_counter()

    rss_gb = proc.memory_info().rss / (1024 ** 3)

    vocab_path = os.path.join(output_dir, "owt_bpe_vocab_32000.pkl")
    merges_path = os.path.join(output_dir, "owt_bpe_merges_32000.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    longest_id, longest_bytes = max(vocab.items(), key=lambda kv: len(kv[1]))
    longest_str = longest_bytes.decode("utf-8", errors="replace")

    print(f"已保存词表 -> {vocab_path}")
    print(f"已保存合并记录 -> {merges_path}")
    print(f"耗时: {(t1 - t0):.2f}秒")
    print(f"RSS (近似): {rss_gb:.2f} GB")
    print(f"最长token id={longest_id}, 字节长度={len(longest_bytes)}")
    print(f"最长token (解码后): {repr(longest_str)}")


if __name__ == "__main__":
    main()