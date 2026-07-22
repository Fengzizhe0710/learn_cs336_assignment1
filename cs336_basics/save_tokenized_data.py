import os
import pickle
import numpy as np
from bpe_tokenizer import Tokenizer 

# ==========================================
# 1. 路径配置 (请根据你的实际文件夹结构修改)
# ==========================================
CONFIG = {
    "tiny": {
        "vocab": "model/tokenizer/vocab_tiny.pkl",
        "merges": "model/tokenizer/merges_tiny.pkl",
        # 输入：原始文本训练集
        "input_txt": "data/TinyStoriesV2-GPT4-train.txt", 
        # 输出：保存为 .npy 格式
        "output_npy": "data/tiny_stories_train.npy" 
    },
    "owt": {
        "vocab": "model/tokenizer/vocab_owt.pkl",
        "merges": "model/tokenizer/merges_owt.pkl",
        # 输入：原始文本训练集
        "input_txt": "data/owt_train.txt", 
        # 输出：保存为 .npy 格式
        "output_npy": "data/owt_train.npy" 
    }
}

# 每次读取的文本块大小 (例如 50MB)，防止内存溢出
CHUNK_SIZE_BYTES = 50 * 1024 * 1024 

def load_tokenizer(vocab_path, merges_path):
    """
    正确加载 Tokenizer 的辅助函数。
    从 pkl 文件中读取 vocab 和 merges 数据，然后传入构造函数。
    """
    print(f"  -> 正在从磁盘加载词表: {vocab_path}")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        
    print(f"  -> 正在从磁盘加载合并规则: {merges_path}")
    with open(merges_path, 'rb') as f:
        merges = pickle.load(f)
    
    # 使用正确的参数初始化 Tokenizer
    # 注意：这里假设你的 Tokenizer 类支持 special_tokens 参数
    # 如果不需要特殊 token，可以传空列表或 None，视具体实现而定
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=["<|endoftext|>"])
    return tokenizer

def encode_and_save_dataset(tokenizer, input_path, output_path):
    """
    流式读取文本文件 -> 编码 -> 追加保存到 npy 文件
    """
    print(f"\n开始处理: {os.path.basename(input_path)}")
    print(f"目标文件: {output_path}")
    
    all_ids = []
    total_tokens = 0
    
    # 使用 'r' 模式读取文本，encoding='utf-8' 处理特殊字符
    with open(input_path, 'r', encoding='utf-8') as f:
        while True:
            # 1. 分块读取文本 (避免 MemoryError)
            chunk_text = f.read(CHUNK_SIZE_BYTES)
            
            if not chunk_text:
                break
            
            # 2. 对当前块进行编码
            ids = tokenizer.encode(chunk_text)
            
            # 3. 收集 ID
            all_ids.extend(ids)
            total_tokens += len(ids)
            
            print(f"  ...已处理 {total_tokens:,} tokens", end='\r')

    print(f"\n  处理完成! 总 Token 数: {total_tokens:,}")

    # 4. 转换为 uint16 并保存
    data_array = np.array(all_ids, dtype=np.uint16)
    
    np.save(output_path, data_array)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  文件已保存至: {output_path} (大小: {file_size_mb:.2f} MB)")
    
    # 释放内存
    del all_ids, data_array


def main():
    print("="*60)
    print("任务 (d): 将训练数据编码并保存为 uint16 NumPy 数组")
    print("="*60)

    for dataset_name, paths in CONFIG.items():
        print(f"\n>>> 正在初始化 {dataset_name.upper()} 分词器...")
        
        # 检查文件是否存在
        if not os.path.exists(paths['vocab']) or not os.path.exists(paths['merges']):
            print(f"错误: 找不到 {dataset_name} 的分词器文件，跳过。")
            continue
            
        if not os.path.exists(paths['input_txt']):
            print(f"错误: 找不到输入文件 {paths['input_txt']}，跳过。")
            continue

        # 【修复点】使用辅助函数正确加载分词器
        try:
            tokenizer = load_tokenizer(paths['vocab'], paths['merges'])
        except Exception as e:
            print(f"加载分词器失败: {e}")
            continue
        
        # 执行编码和保存
        encode_and_save_dataset(
            tokenizer=tokenizer,
            input_path=paths['input_txt'],
            output_path=paths['output_npy']
        )

    print("\n" + "="*60)
    print("所有任务完成！")
    print("="*60)
    
    # --- 回答题目中的思考题 ---
    print("\n[思考题回答] 为什么 uint16 是一个合适的选择？")
    print("答：因为常用的 BPE 分词器词汇表大小通常在 50,000 左右，")
    print("远小于 uint16 的最大值 65,535。使用 uint16 可以将每个 token 的")
    print("存储空间从默认的 int64 (8字节) 减少到 2字节，节省 75% 的内存/显存。")

if __name__ == "__main__":
    main()
    