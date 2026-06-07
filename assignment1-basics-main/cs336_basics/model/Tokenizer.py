from typing import Iterable, Iterator
import regex
import re
import pickle

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        # bytes -> token id
        self.bytes_to_id = {
            token_bytes: token_id
            for token_id, token_bytes in vocab.items()
        }
        # GPT2 byte vocabulary
        self.byte_vocab = {}
        for token_id, token_bytes in vocab.items():
            if len(token_bytes) == 1:
                self.byte_vocab[token_bytes[0]] = token_id
        self.special_tokens = sorted(
            special_tokens or [],
            key=len,
            reverse=True,
        )
        # append special tokens if missing
        next_id = max(vocab.keys()) + 1
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.bytes_to_id:
                self.vocab[next_id] = token_bytes
                self.bytes_to_id[token_bytes] = next_id
                next_id += 1
        # merge rank table
        self.merge_ranks = {}
        for rank, (left, right) in enumerate(merges):
            if (
                left in self.bytes_to_id
                and right in self.bytes_to_id
            ):
                left_id = self.bytes_to_id[left]
                right_id = self.bytes_to_id[right]
                self.merge_ranks[(left_id, right_id)] = rank

    # =====================================================
    # from files
    # =====================================================
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens=None,
    ):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens,
        )

    # =====================================================
    # encode one regex piece
    # =====================================================
    def _encode_piece(
        self,
        piece: bytes,
    ) -> list[int]:
        ids = [
            self.byte_vocab[b]
            for b in piece
        ]
        while len(ids) > 1:
            best_rank = None
            best_pos = None
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pos = i
            if best_pos is None:
                break
            a = ids[best_pos]
            b = ids[best_pos + 1]
            merged_bytes = self.vocab[a] + self.vocab[b]
            new_id = self.bytes_to_id[merged_bytes]
            ids = ids[:best_pos] + [new_id] + ids[best_pos + 2:]
        return ids

    # =====================================================
    # encode
    # =====================================================
    def encode(
        self,
        text: str,
    ) -> list[int]:
        result = []
        if self.special_tokens:
            pattern = (
                "("
                + "|".join(
                    re.escape(tok)
                    for tok in self.special_tokens
                )
                + ")"
            )
            chunks = [
                x
                for x in re.split(pattern, text)
                if x
            ]
        else:
            chunks = [text]
        special_set = set(self.special_tokens)
        for chunk in chunks:
            # special token
            if chunk in special_set:
                result.append(
                    self.bytes_to_id[
                        chunk.encode("utf-8")
                    ]
                )
                continue
            # normal text
            for piece in regex.findall(
                PAT,
                chunk,
            ):
                result.extend(
                    self._encode_piece(
                        piece.encode("utf-8")
                    )
                )
        return result

    # =====================================================
    # encode iterable
    # =====================================================
    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    # =====================================================
    # decode
    # =====================================================
    def decode(
        self,
        ids: list[int],
    ) -> str:
        data = b"".join(
            self.vocab[token_id]
            for token_id in ids
        )
        return data.decode(
            "utf-8",
            errors="replace",
        )