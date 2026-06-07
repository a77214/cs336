from collections import defaultdict
import regex

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def get_stats(words):
    """
    统计所有 pair 频率

    words:
        [
            ((1,2,3), freq),
            ...
        ]
    """
    stats = defaultdict(int)
    for ids, freq in words:
        for i in range(len(ids) - 1):
            stats[(ids[i], ids[i + 1])] += freq

    return stats


def merge_word(ids, pair, new_id):
    """
    对单个 word 做 merge
    """
    i = 0
    new_ids = []
    while i < len(ids):
        if (
                i < len(ids) - 1
                and ids[i] == pair[0]
                and ids[i + 1] == pair[1]
        ):
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return tuple(new_ids)


def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str] = None

):
    """
    GPT-style BPE 增量更新实现
    """

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")

    if special_tokens is None:
        special_tokens = []

    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size too small")

    # =========================================================
    # 初始化 vocab
    # =========================================================
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    special_token_to_id = {}

    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        special_token_to_id[token] = next_id
        next_id += 1

    # =========================================================
    # 读取文本
    # =========================================================
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    if not text:
        return vocab, []

    # =========================================================
    # special token split
    # =========================================================
    if special_tokens:
        pattern = "(" + "|".join(
            regex.escape(t) for t in special_tokens
        ) + ")"

        chunks = regex.split(pattern, text)

    else:
        chunks = [text]

    # =========================================================
    # 构建 word frequency
    # =========================================================
    word_freq = defaultdict(int)

    for chunk in chunks:

        # special token 不参与 merge 训练
        if chunk in special_token_to_id:
            continue

        for word in regex.findall(PAT, chunk):
            ids = tuple(word.encode("utf-8"))

            word_freq[ids] += 1

    # =========================================================
    # words:
    # [
    #   ((1,2,3), freq),
    # ]
    # =========================================================
    words = list(word_freq.items())

    # =========================================================
    # 初始 pair 统计
    # =========================================================
    pair_counts = get_stats(words)

    merges = []

    # =========================================================
    # 主循环
    # =========================================================
    while next_id < vocab_size:

        if not pair_counts:
            break

        # -----------------------------------------------------
        # 删除无效 pair
        # -----------------------------------------------------
        dead_pairs = [
            p
            for p, c in pair_counts.items()
            if c <= 0
        ]

        for p in dead_pairs:
            del pair_counts[p]

        if not pair_counts:
            break

        # -----------------------------------------------------
        # 找最优 pair
        #
        # 频率优先
        # bytes lexicographic tie-break
        # -----------------------------------------------------
        max_count = max(pair_counts.values())

        candidates = [
            p
            for p, c in pair_counts.items()
            if c == max_count
        ]

        best_pair = max(
            candidates,
            key=lambda p: (
                vocab[p[0]],
                vocab[p[1]]
            )
        )

        best_count = pair_counts[best_pair]

        if best_count <= 0:
            break

        # -----------------------------------------------------
        # 创建新 token
        # -----------------------------------------------------
        new_id = next_id
        next_id += 1

        vocab[new_id] = (
                vocab[best_pair[0]]
                + vocab[best_pair[1]]
        )

        # merges 必须存 bytes
        merges.append((
            vocab[best_pair[0]],
            vocab[best_pair[1]]
        ))

        # =====================================================
        # 增量更新
        # =====================================================
        delta = defaultdict(int)

        new_words = []

        for ids, freq in words:

            # 快速跳过
            if best_pair not in zip(ids, ids[1:]):
                new_words.append((ids, freq))
                continue

            old_ids = ids

            # merge
            new_ids = merge_word(
                old_ids,
                best_pair,
                new_id
            )

            # -------------------------------------------------
            # 删除旧 pair 贡献
            # -------------------------------------------------
            for i in range(len(old_ids) - 1):
                delta[(old_ids[i], old_ids[i + 1])] -= freq

            # -------------------------------------------------
            # 增加新 pair 贡献
            # -------------------------------------------------
            for i in range(len(new_ids) - 1):
                delta[(new_ids[i], new_ids[i + 1])] += freq

            new_words.append((new_ids, freq))

        # =====================================================
        # 应用增量更新
        # =====================================================
        for pair, change in delta.items():

            pair_counts[pair] += change

            if pair_counts[pair] <= 0:
                del pair_counts[pair]

        words = new_words

    return vocab, merges