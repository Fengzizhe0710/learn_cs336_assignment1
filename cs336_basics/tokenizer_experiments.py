import time
from bpe_tokenizer import Tokenizer


def tiny_compression_ratio(tiny_tokenizer, text):
    """计算 tiny tokenizer 在 tiny 数据上的压缩比"""
    byte_count = len(text.encode("utf-8"))
    token_ids = tiny_tokenizer.encode(text)
    token_count = len(token_ids)

    if token_count > 0:
        compression_ratio = byte_count / token_count
        print("-----使用tiny tokenizer解码tiny_valid数据-----")
        print(f"文本字节数: {byte_count}")
        print(f"token数: {token_count}")
        print(f"压缩比 (字节/token): {compression_ratio:.2f}")
    else:
        print("编码后标记数为0，无法计算压缩比。")


def owt_compression_ratio(owt_tokenizer, text):
    """计算 owt tokenizer 在 owt 数据上的压缩比"""
    byte_count = len(text.encode("utf-8"))
    token_ids = owt_tokenizer.encode(text)
    token_count = len(token_ids)

    if token_count > 0:
        compression_ratio = byte_count / token_count
        print("-----使用owt tokenizer解码owt_valid数据-----")
        print(f"文本字节数: {byte_count}")
        print(f"token数: {token_count}")
        print(f"压缩比 (字节/token): {compression_ratio:.2f}")
    else:
        print("编码后标记数为0，无法计算压缩比。")


def tiny_tokenizer_on_owt(tiny_tokenizer, text):
    """使用 tiny tokenizer 对 owt 数据进行编码，评估跨域压缩效果"""
    byte_count = len(text.encode("utf-8"))
    # 修复：此处应使用传入的 tiny_tokenizer，而非未定义的 owt_tokenizer
    token_ids = tiny_tokenizer.encode(text)
    token_count = len(token_ids)

    if token_count > 0:
        compression_ratio = byte_count / token_count
        print("-----使用tiny tokenizer解码owt_valid数据-----")
        print(f"文本字节数: {byte_count}")
        print(f"token数: {token_count}")
        print(f"压缩比 (字节/token): {compression_ratio:.2f}")
    else:
        print("编码后标记数为0，无法计算压缩比。")


def estimate_owt_throughput(owt_tokenizer, text, num_runs=5):
    """估算 owt tokenizer 的吞吐量（tokens/秒）"""
    # 预热：先执行一次编码，避免首次加载/编译开销影响测量
    owt_tokenizer.encode(text)

    total_tokens = 0
    total_time = 0.0

    for i in range(num_runs):
        start = time.perf_counter()
        token_ids = owt_tokenizer.encode(text)
        end = time.perf_counter()

        total_tokens += len(token_ids)
        total_time += end - start

    if total_time > 0:
        avg_tokens_per_run = total_tokens / num_runs
        avg_time_per_run = total_time / num_runs
        throughput = total_tokens / total_time

        print("-----估算owt tokenizer吞吐量-----")
        print(f"运行次数: {num_runs}")
        print(f"平均每次编码token数: {avg_tokens_per_run:.0f}")
        print(f"平均每次编码耗时: {avg_time_per_run:.4f} 秒")
        print(f"吞吐量: {throughput:.2f} tokens/秒")
    else:
        print("编码耗时为0，无法计算吞吐量。")


if __name__ == "__main__":
    # ========== 数据与tokenizer只初始化一次 ==========
    print("正在加载tokenizer和文本数据...")

    # 加载两个 tokenizer
    tiny_tokenizer = Tokenizer.from_files(
        vocab_path="model/tokenizer/vocab_tiny.pkl",
        merges_path="model/tokenizer/merges_tiny.pkl",
        special_tokens=["<|endoftext|>"],
    )
    owt_tokenizer = Tokenizer.from_files(
        vocab_path="model/tokenizer/vocab_owt.pkl",
        merges_path="model/tokenizer/merges_owt.pkl",
        special_tokens=["<|endoftext|>"],
    )

    # 加载两份验证文本
    with open("data/TinyStoriesV2-GPT4-valid.txt", "r", encoding="utf-8") as f:
        tiny_text = f.read()
    with open("data/owt_valid.txt", "r", encoding="utf-8") as f:
        owt_text = f.read()

    print("加载完成，开始评估...\n")

    # 两个 tokenizer 在各自验证集上的压缩比
    tiny_compression_ratio(tiny_tokenizer, tiny_text)
    print()

    owt_compression_ratio(owt_tokenizer, owt_text)
    print()

    # 使用 tiny tokenizer 对 owt 数据进行编码（跨域评估）
    tiny_tokenizer_on_owt(tiny_tokenizer, owt_text)
    print()

    # 估算 owt tokenizer 的吞吐量
    estimate_owt_throughput(owt_tokenizer, owt_text)
    