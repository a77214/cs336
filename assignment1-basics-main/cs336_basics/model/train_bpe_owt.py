import os
import time
import pickle
import psutil
import gc
from cs336_basics.model.train_bpe import train_bpe


def main():
    input_path = "data/owt_train.txt"
    output_dir = "workspace"
    os.makedirs(output_dir, exist_ok=True)

    vocab_size = 32_000
    special_tokens = ["<|endoftext|>"]

    # 内存阈值设置
    MEMORY_LIMIT_GB = 30  # 根据你的机器调整
    MAX_TRAINING_LINES = 5_000_000  # 限制行数，防止OOM

    proc = psutil.Process(os.getpid())

    # 检查可用内存
    available_mem = psutil.virtual_memory().available / (1024 ** 3)
    if available_mem < MEMORY_LIMIT_GB:
        print(f"Warning: Only {available_mem:.2f}GB available, might be insufficient")
        # 减少数据量
        MAX_TRAINING_LINES = min(MAX_TRAINING_LINES, 1_000_000)

    t0 = time.perf_counter()

    try:
        vocab, merges = train_bpe(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            num_processes=4,
            max_lines=MAX_TRAINING_LINES  # 如果train_bpe支持
        )
    except MemoryError:
        print("MemoryError occurred! Trying with reduced settings...")
        # 降级方案：单进程 + 更少数据
        vocab, merges = train_bpe(
            input_path=input_path,
            vocab_size=vocab_size // 2,  # 减小vocab
            special_tokens=special_tokens,
            num_processes=1,
            max_lines=MAX_TRAINING_LINES // 2
        )

    t1 = time.perf_counter()

    # 强制清理
    gc.collect()

    rss_gb = proc.memory_info().rss / (1024 ** 3)

    vocab_path = os.path.join(output_dir, "owt_bpe_vocab_32000.pkl")
    merges_path = os.path.join(output_dir, "owt_bpe_merges_32000.pkl")

    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    longest_id, longest_bytes = max(vocab.items(), key=lambda kv: len(kv[1]))
    longest_str = longest_bytes.decode("utf-8", errors="replace")

    print(f"Saved vocab -> {vocab_path}")
    print(f"Saved merges -> {merges_path}")
    print(f"Elapsed: {(t1 - t0):.2f}s")
    print(f"RSS (approx): {rss_gb:.2f} GB")
    print(f"Longest token id={longest_id}, bytes_len={len(longest_bytes)}")
    print(f"Longest token (decoded): {repr(longest_str)}")


if __name__ == "__main__":
    main()