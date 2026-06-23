# step1_cloud.py
import pickle
import os
import time
from collections import Counter
from multiprocessing import Pool
import regex
import re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_COMPILED = None


def init_worker():
    global PAT_COMPILED
    PAT_COMPILED = regex.compile(PAT)


def find_chunk_boundaries(input_path, num_chunks):
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
        pattern = "(" + "|".join(re.escape(t) for t in sorted(special_tokens, key=len, reverse=True)) + ")"
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
    tasks = [(input_path, s, e, special_tokens) for s, e in zip(boundaries[:-1], boundaries[1:])]
    merged = Counter()
    with Pool(processes=num_processes, initializer=init_worker) as pool:
        for c in pool.map(process_chunk, tasks):
            merged.update(c)
    return merged


def main():
    input_path = "data/owt_train.txt"
    output_dir = "workspace"
    os.makedirs(output_dir, exist_ok=True)

    special_tokens = ["<|endoftext|>"]
    num_processes = 8

    print(f"Processing {input_path} with {num_processes} processes...")
    t0 = time.time()

    boundaries = find_chunk_boundaries(input_path, num_processes)
    print(f"Chunks: {len(boundaries) - 1}")

    word_freq = build_parallel_word_freq(input_path, boundaries, special_tokens, num_processes)

    t1 = time.time()
    print(f"Done in {t1 - t0:.1f}s, unique words: {len(word_freq):,}")

    # 保存
    output_path = os.path.join(output_dir, "owt_word_freq.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(dict(word_freq), f)
    print(f"Saved to {output_path}")
    print(f"Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()