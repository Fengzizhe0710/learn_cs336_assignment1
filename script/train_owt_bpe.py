import os
import time
import pickle
from cs336_basics import train_bpe


if __name__ == "__main__":
    data_path = "data/owt_train.txt"
    out = "model/tokenizer"

    # 确保输出目录存在
    os.makedirs(out, exist_ok=True)

    print(f"开始训练 BPE...")
    print(f"  语料: {data_path}")
    print(f"  词汇表大小: 32000")
    print(f"  特殊标记: ['<|endoftext|>']")
    print(f"  并行进程: 4")

    start_time = time.time()

    vocab, merges = train_bpe.train_bytes_bpe(
        input_path=data_path,
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
        num_workers=4,
    )

    elapsed = time.time() - start_time

    # 保存结果
    with open(os.path.join(out, "vocab_owt.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    with open(os.path.join(out, "merges_owt.pkl"), "wb") as f:
        pickle.dump(merges, f)

    print(f"\n训练完成！")
    print(f"  词汇表大小: {len(vocab)}")
    print(f"  合并规则数: {len(merges)}")
    print(f"  总耗时: {elapsed:.2f}s ({elapsed/60:.2f}min)")
    print(f"  保存路径: {out}/")
