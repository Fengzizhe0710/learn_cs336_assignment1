import os
import regex as re
from collections import defaultdict
from typing import Dict, List, Tuple, Union, BinaryIO, Set
import multiprocessing as mp


PAT = re.compile(r"""'(?:[smdt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def find_chunk_boundaried(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes
) -> List[int]:
    """
    将文档切分成若干块，每块以 split_special_token 作为边界。
    """
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_sizes = file_size // desired_num_chunks
    boundaries = [i * chunk_sizes for i in range(desired_num_chunks + 1)]
    boundaries[-1] = file_size

    mini_chunk_size = 4096
    for i in range(1, len(boundaries) - 1):
        pos = boundaries[i]
        file.seek(pos)
        while True:
            chunk = file.read(mini_chunk_size)
            if not chunk:
                boundaries[i] = file_size
                break
            found = chunk.find(split_special_token)
            if found != -1:
                boundaries[i] = pos + found
                break
            pos += mini_chunk_size
    return sorted(set(boundaries))


def process_chunk_for_word_counts(
    input_path: str,
    start: int,
    end: int,
    special_tokens: List[str]
) -> Dict[Tuple[bytes, ...], int]:
    """
    处理文件块，统计词频
    """
    with open(input_path, 'rb') as f:
        f.seek(start)
        data = f.read(end - start)
        text = data.decode('utf-8', errors='ignore')
        # 将 CRLF 统一为 LF
        text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 按特殊token切分，要保证特殊token完整性
    if special_tokens:
        escaped = '|'.join(re.escape(token) for token in special_tokens)
        split_re = re.compile(f'({escaped})')
        segments = split_re.split(text)
    else:
        segments = [text]

    word_counts = defaultdict(int)

    for seg in segments:
        # 跳过空串和特殊token
        if not seg or seg in special_tokens:
            continue
        
        # 预分词并进行统计
        for m in PAT.finditer(seg):
            word_bytes = m.group().encode('utf-8')
            # 存为 tuple 后面可以作为 dict key
            word_tuple = tuple(bytes([b]) for b in word_bytes)
            if len(word_tuple) >= 2:  # 只有长度大于等于2才有pair
                word_counts[word_tuple] += 1
                
    return word_counts


def merge_word(
    word: Tuple[bytes, ...],
    pair: Tuple[bytes, bytes]
) -> Tuple[bytes, ...]:
    """
    在单个词中合并指定的pair
    """
    merged = pair[0] + pair[1]
    i = 0 
    res = []
    L = len(word)
    while i < L:
        if i + 1 < L and word[i] == pair[0] and word[i+1] == pair[1]:
            res.append(merged)
            i += 2
        else:
            res.append(word[i])
            i += 1
    return tuple(res)
    

def train_bytes_bpe(
    input_path: Union[str, os.PathLike],
    vocab_size: int,
    special_tokens: List[str],
    num_workers: int = 4
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    训练 byte level BPE分词器
    返回：
    vocab: 分词器词汇表 (int -> bytes)
    merges: 合并列表 [(bytes, bytes)]
    """
    input_path = os.fspath(input_path)

    # 初始化词汇表 大小为256
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    special_ids = {}

    # 添加特殊标记到词汇表 如：<|endoftext|>
    for token in special_tokens:
        token_bytes = token.encode('utf-8')
        vocab[next_id] = token_bytes
        special_ids[token] = next_id
        next_id += 1

    # 计算需要合并的次数
    base_vocab_size = len(vocab)
    n_merges = max(0, vocab_size - base_vocab_size)

    # 将文档进行分块，多线程统计词频
    split_token = special_tokens[0].encode('utf-8') if special_tokens else None
    with open(input_path, 'rb') as f:
        if split_token:
            boundaries = find_chunk_boundaried(f, num_workers, split_token)
        else:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            f.seek(0)
            chunk_size = file_size // num_workers
            boundaries = [i * chunk_size for i in range(num_workers + 1)]
            boundaries[-1] = file_size

    # 并行处理
    with mp.Pool(processes=num_workers) as pool:
        args = [
            (input_path, start, end, special_tokens)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        results = pool.starmap(process_chunk_for_word_counts, args)

    # 合并词频
    word_counts = defaultdict(int)
    for res in results:
        for word, count in res.items():
            word_counts[word] += count

    words = list(word_counts.keys())
    wcnt_list = [word_counts[w] for w in words]

    # pair频率和出现位置
    pair_freq = defaultdict(int)
    pair_occ = defaultdict(set)

    for wid, (word, count) in enumerate(zip(words, wcnt_list)):
        if len(word) < 2:
            continue
        word_pairs = [(word[i], word[i+1]) for i in range(len(word) - 1)]
        for p in word_pairs:
            pair_freq[p] += count
        for p in set(word_pairs):
            pair_occ[p].add(wid)

    # 开始合并
    merges = []

    for step in range(n_merges):
        if not pair_freq:
            break

        # 找到频率最高的pair
        best_pair = None
        best_count = 0

        for pair, count in pair_freq.items():
            if count < best_count:
                continue
            if count > best_count:
                best_pair = pair
                best_count = count
            elif count == best_count:
                # 平局处理：按字典序排序
                if pair[0] > best_pair[0] or (pair[0] == best_pair[0] and pair[1] > best_pair[1]):
                    best_pair = pair
                    best_count = count

        if best_pair is None or best_count <= 0:
            break

        # 记录合并
        merges.append(best_pair)

        # 注册新token
        new_tid = base_vocab_size + step
        new_bytes = best_pair[0] + best_pair[1]
        vocab[new_tid] = new_bytes

        # 更新词频统计
        affected_wids = list(pair_occ.get(best_pair, []))
        if not affected_wids:
            continue

        for wid in affected_wids:
            old_word = words[wid]
            if len(old_word) < 2:
                continue
            count = wcnt_list[wid]

            # 移除旧pair
            for i in range(len(old_word) - 1):
                p = (old_word[i], old_word[i+1])
                pair_freq[p] -= count
                if pair_freq[p] <= 0:
                    pair_freq.pop(p, None)
                occ = pair_occ.get(p)
                if occ is not None:
                    occ.discard(wid)
                    if not occ:
                        pair_occ.pop(p, None)

            # 应用合并
            new_word = merge_word(old_word, best_pair)
            words[wid] = new_word

            # 添加新pair
            for i in range(len(new_word) - 1):
                p = (new_word[i], new_word[i+1])
                pair_freq[p] = pair_freq.get(p, 0) + count
                pair_occ.setdefault(p, set()).add(wid)

    return vocab, merges
