import os
import time
import pickle
import psutil
from collections import Counter
from multiprocessing import Pool
from typing import Iterable, Iterator, Any


# ============================================================
# 添加并行词频统计函数
# ============================================================

def find_chunk_boundaries(file, num_chunks, split_token):
    """找到文件分割边界（对齐到特殊 token）"""
    file.seek(0, 2)
    file_size = file.tell()
    chunk_size = file_size // num_chunks
    boundaries = [0]

    for i in range(1, num_chunks):
        pos = i * chunk_size
        file.seek(pos)
        file.readline()  # 跳到下一行
        boundaries.append(file.tell())

    boundaries.append(file_size)
    return boundaries


def process_chunk(args):
    """处理单个 chunk 的词频统计"""
    input_path, start, end, special_tokens = args
    # 这里需要你的实际分词逻辑
    # 返回 Counter 对象
    local_counter = Counter()

    with open(input_path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
        # 处理数据...
        # 这里调用你的分词函数

    return local_counter


def init_worker():
    """初始化 worker 进程"""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def build_word_freq_parallel(
        input_path: str,
        special_tokens: list[str],
        num_processes: int,
        *,
        num_chunks: int | None = None
) -> dict[tuple[bytes, ...], int]:
    """
    Build word frequency statistics using multiprocessing.
    Chunk boundaries are aligned to special-token boundaries.
    """
    if num_processes <= 1 or not special_tokens:
        # 需要导入串行版本
        # return build_word_freq_serial(input_path, special_tokens)
        pass

    if num_chunks is None:
        num_chunks = max(num_processes * 32, num_processes)

    split_special_token = special_tokens[0].encode("utf-8")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, split_special_token)

    tasks = [(str(input_path), s, e, special_tokens) for s, e in zip(boundaries[:-1], boundaries[1:])]

    merged = Counter()
    with Pool(processes=num_processes, initializer=init_worker, maxtasksperchild=8) as pool:
        for partial in pool.imap_unordered(process_chunk, tasks, chunksize=1):
            merged.update(partial)

    return dict(merged)


# ============================================================
# 主函数（修改版）
# ============================================================

def main():
    input_path = "data/owt_train.txt"
    output_dir = "workspace"
    os.makedirs(output_dir, exist_ok=True)

    vocab_size = 32_000
    special_tokens = ["<|endoftext|>"]
    num_processes = 2  # 降低进程数减少内存

    proc = psutil.Process(os.getpid())

    def log_memory(msg=""):
        mem = psutil.virtual_memory()
        rss = proc.memory_info().rss / 1024 ** 3
        print(f"{msg} RSS: {rss:.2f}GB, 系统: {mem.used / 1024 ** 3:.2f}GB/{mem.total / 1024 ** 3:.2f}GB")

    log_memory("训练前")

    t0 = time.perf_counter()

    # ============================================================
    # 使用并行词频统计（而不是直接调用 train_bpe）
    # ============================================================
    # 如果 train_bpe 内部调用 build_word_freq_parallel，则直接调用 train_bpe
    # 否则需要手动构建词频然后训练

    try:
        from cs336_basics.model.train_bpe import train_bpe

        # 如果 train_bpe 支持直接调用
        vocab, merges = train_bpe(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            num_processes=num_processes
        )
    except (ImportError, TypeError):
        # 如果不支持，手动构建
        print("train_bpe 不支持并行，使用手动构建...")

        # 构建词频
        word_freq = build_word_freq_parallel(
            input_path=input_path,
            special_tokens=special_tokens,
            num_processes=num_processes,
            num_chunks=num_processes * 8  # 减少 chunk 数
        )

        # 然后使用词频训练（这里需要你的 train_bpe 实现）
        # vocab, merges = train_bpe_from_freq(word_freq, vocab_size, special_tokens)
        print(f"词频统计完成: {len(word_freq)} 个词")

    t1 = time.perf_counter()
    log_memory("训练后")

    # 保存结果
    vocab_path = os.path.join(output_dir, "owt_bpe_vocab_32000.pkl")
    merges_path = os.path.join(output_dir, "owt_bpe_merges_32000.pkl")

    # 如果有结果则保存
    try:
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)
        with open(merges_path, "wb") as f:
            pickle.dump(merges, f)
        print(f"保存完成: {vocab_path}")
    except NameError:
        print("没有生成 vocab/merges，只进行了词频统计")

    print(f" 耗时: {(t1 - t0):.2f}s")


if __name__ == "__main__":
    main()