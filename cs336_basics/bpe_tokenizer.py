import pickle
import regex as re
from typing import Dict, List, Tuple, Iterable, Iterator, Optional


PAT = re.compile(r"""'(?:[smdt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


class Tokenizer:
    """BPE 分词器：将文本编码为 token id, 并将 token id 解码为文本。"""

    def __init__(
        self, 
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[bytes]] = None,
    ):
        """
        构造分词器

        Args:
            vocab: 词汇表, token id 到字节序列的映射
            merges: BPE 合并规则表
            special_tokens: 特殊 token 列表
        """
        self.vocab = Dict(vocab)
        self.merges = List(merges)
        self.special_tokens = List(special_tokens) if special_tokens else []

        # 构建字节序列到 token id 的映射
        self.byte_to_rank: Dict[bytes, int] = {}
        for token_id, token_bytes in self.vocab.items():
            self.byte_to_rank[token_bytes] = token_id

        # 处理特殊token，如果不在词汇表中要添加
        self.special_token_ids: Dict[str, int] = {}
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes in self.byte_to_rank:
                self.special_token_ids[token] = self.byte_to_rank[token_bytes]
            else:
                new_id = max(self.vocab.keys()) + 1 if self.vocab else 0
                self.vocab[new_id] = token_bytes
                self.byte_to_rank[token_bytes] = new_id
                self.special_token_ids[token] = new_id

        # 构建合并优先级映射
        self.merge_rank: Dict[Tuple[bytes, bytes], int] = {}
        for i, (left, right) in enumerate(self.merges):
            self.merge_rank[(left, right)] = i

    @classmethod
    def from_files(
        cls,
        vocab_path: str,
        merges_path: str,
        special_tokens: Optional[List[str]] = None,
    ) -> "Tokenizer":
        """
        从文件中加载分词器

        Args:
            vocab_path: 词汇表文件路径
            merges_path: 合并规则文件路径
            special_tokens: 特殊token列表
        """
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_path, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def _encode_bytees(self, word_bytes: bytes) -> List[int]:
        """
        对字节串应用 BPE 编码
        """
        tokens: List[bytes] = [bytes([b]) for b in word_bytes]

        while len(tokens) > 1:
            best_rank = float('inf')
            best_idx = -1

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_rank:
                    rank = self.merge_rank[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_idx = i
            
            # 没有可合并的对，结束循环
            if best_idx == -1:
                break

            merged = tokens[best_idx] + tokens[best_idx + 1]
            tokens[:best_idx] + [merged] + tokens[best_idx + 2:]

        result: List[int] = []
        for token in tokens:
            if token in self.byte_to_rank:
                result.append(self.byte_to_rank[token])
            else:
                # 未知token
                for b in token:
                    result.append(b)

        return result

    def encode(self, text: str) -> List[int]:
        """
        将文本编码为 token id 列表

        Args:
            text: 要编码的文本

        Returns:
            token ID 列表
        """
        if not text:
            return []

        # 按特殊token分割文本
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped = '|'.join(re.escape(token) for token in sorted_special_tokens)
            split_re = re.compile(f"({escaped})")
            segments = split_re.split(text)
        else:
            segments = [text]

        result: List[int] = []

        for segment in segments:
            if not segment:
                continue

            # 如果片段是特殊token，直接添加其ID
            if segment in self.special_token_ids:
                result.append(self.special_token_ids[segment])
                
            # 预分词
            for match in PAT.finditer(segment):
                word = match.group()
                word_bytes = word.encode("utf-8")

                if word_bytes in self.byte_to_rank:
                    result.append(self.byte_to_rank[word_bytes])
                else:
                    result.extend(self._encode_bytees(word_bytes))

        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        将可迭代对象中的文本编码为 token id 列表

        Args:
            iterable: 字符串的可迭代对象

        Yields:
            逐个产生 token id
        """
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, token_ids: List[int]) -> str:
        """
        将 token id 列表解码为文本

        Args:
            ids: token id 列表

        Returns:
            解码后的文本
        """
        bytes_parts: List[bytes] = []
        for token_id in token_ids:
            if token_id in self.vocab:
                bytes_parts.append(self.vocab[token_id])
            else:
                bytes_parts.append(b'')

        full_bytes = b''.join(bytes_parts)
        return full_bytes.decode("utf-8", errors="replace")
