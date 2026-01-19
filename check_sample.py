from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from collections import defaultdict
import argparse
import glob
import os

# ===== 配置 =====
MODEL_NAME = "Qwen/Qwen3-8B"
BATCH_SIZE = 1024
THRESHOLD = 512

# ===== 解析参数 =====
parser = argparse.ArgumentParser(description="统计 FinBench 数据集的样本分布")
parser.add_argument(
    "path_prefix",
    type=str,
    help="路径前缀，例如 /home/dan/data/finbench_verl/train_v6",
)
args = parser.parse_args()

# ===== 找到所有匹配的 parquet 文件 =====
path_prefix = args.path_prefix
# 如果路径前缀不包含 .parquet，则添加通配符
if not path_prefix.endswith(".parquet"):
    # 尝试精确匹配
    exact_path = path_prefix + ".parquet"
    if os.path.exists(exact_path):
        parquet_files = [exact_path]
    else:
        # 使用通配符匹配所有相关文件
        pattern = path_prefix + "*.parquet"
        parquet_files = sorted(glob.glob(pattern))
else:
    parquet_files = [path_prefix] if os.path.exists(path_prefix) else []

if not parquet_files:
    print(f"错误: 未找到匹配的文件: {path_prefix}*.parquet")
    exit(1)

print(f"找到 {len(parquet_files)} 个文件:")
for f in parquet_files:
    print(f"  - {f}")

# ===== 加载 parquet =====
# 为每个文件创建一个 split 名称
data_files = {}
for i, file_path in enumerate(parquet_files):
    # 使用文件名（不含路径和扩展名）作为 split 名称
    split_name = os.path.splitext(os.path.basename(file_path))[0]
    data_files[split_name] = file_path

ds = load_dataset("parquet", data_files=data_files)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# -------------------------------------------------------------------
# map 1: 计算 prompt_len（batched，快）
# -------------------------------------------------------------------
def add_prompt_len(batch):
    texts = [p[0]["content"] for p in batch["prompt"]]
    enc = tokenizer(
        texts,
        add_special_tokens=False,
        return_length=True,
        padding=False,
        truncation=False,
    )
    return {"prompt_len": enc["length"]}

ds = ds.map(
    add_prompt_len,
    batched=True,
    batch_size=BATCH_SIZE,
)

# -------------------------------------------------------------------
# map 2: 拉平 task / y（避免后面频繁 dict 访问）
# -------------------------------------------------------------------
def flatten_meta(batch):
    return {
        "task": [ei["task"] for ei in batch["extra_info"]],
        "y": [ei["y"] for ei in batch["extra_info"]],
    }

ds = ds.map(
    flatten_meta,
    batched=True,
    batch_size=BATCH_SIZE,
)

# -------------------------------------------------------------------
# 统计函数（map 之后，纯 numpy）
# -------------------------------------------------------------------
def summarize_split(split):
    tasks = np.array(ds[split]["task"])
    ys = np.array(ds[split]["y"])
    lens = np.array(ds[split]["prompt_len"])

    # 总体统计
    total_count = len(ys)
    positive_count = int((ys == 1).sum())
    negative_count = int((ys == 0).sum())
    
    # 按 task 统计
    task_stats = {}
    for task in np.unique(tasks):
        task_mask = (tasks == task)
        task_positive = int((task_mask & (ys == 1)).sum())
        task_negative = int((task_mask & (ys == 0)).sum())
        task_stats[task] = {
            "total": int(task_mask.sum()),
            "positive": task_positive,
            "negative": task_negative,
        }

    # 详细统计（按 task × label）
    results = []
    for task in np.unique(tasks):
        for y in (0, 1):
            mask = (tasks == task) & (ys == y)
            if not mask.any():
                continue

            arr = lens[mask]
            results.append({
                "task": task,
                "y": y,
                "count": int(arr.size),
                "max": int(arr.max()),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                ">512": int((arr > THRESHOLD).sum()),
            })

    return {
        "total": total_count,
        "positive": positive_count,
        "negative": negative_count,
        "task_stats": task_stats,
        "details": results,
    }

# -------------------------------------------------------------------
# 打印结果
# -------------------------------------------------------------------
def print_summary(split, stats):
    print("\n" + "=" * 110)
    print(f"[{split.upper()}] 样本分布统计")
    print("=" * 110)
    
    # 总体统计
    print(f"\n总体统计:")
    print(f"  总样本数: {stats['total']:,}")
    if stats['total'] > 0:
        print(f"  正样本 (y=1): {stats['positive']:,} ({stats['positive']/stats['total']*100:.2f}%)")
        print(f"  负样本 (y=0): {stats['negative']:,} ({stats['negative']/stats['total']*100:.2f}%)")
    else:
        print(f"  正样本 (y=1): {stats['positive']:,}")
        print(f"  负样本 (y=0): {stats['negative']:,}")
    
    # 按 task 统计
    print(f"\n按任务统计:")
    print(f"{'Task':>6} | {'Total':>8} | {'Positive':>10} | {'Negative':>10} | {'Pos%':>6}")
    print("-" * 60)
    for task in sorted(stats['task_stats'].keys()):
        ts = stats['task_stats'][task]
        pos_pct = ts['positive'] / ts['total'] * 100 if ts['total'] > 0 else 0
        print(
            f"{task:>6} | {ts['total']:8d} | {ts['positive']:10d} | "
            f"{ts['negative']:10d} | {pos_pct:5.2f}%"
        )

def print_table(split, stats):
    rows = stats['details']
    print(f"\n[{split.upper()}] Prompt Length 分布 (按 task × label)")
    print("-" * 110)
    print(f"{'Task':>4} | {'y':>1} | {'N':>7} | {'max':>4} | {'p95':>6} | {'p99':>6} | {'>512':>5}")
    print("-" * 110)

    for r in sorted(rows, key=lambda x: (x["task"], x["y"])):
        print(
            f"{r['task']:>4} | {r['y']:>1} | {r['count']:7d} | "
            f"{r['max']:4d} | {r['p95']:6.1f} | {r['p99']:6.1f} | {r['>512']:5d}"
        )

# ===== 运行 =====
print("\n开始处理数据...")
all_stats = {}
for split_name in ds.keys():
    print(f"\n处理 split: {split_name}")
    all_stats[split_name] = summarize_split(split_name)
    print_summary(split_name, all_stats[split_name])
    print_table(split_name, all_stats[split_name])

# 汇总所有 splits
if len(all_stats) > 0:
    print("\n" + "=" * 110)
    if len(all_stats) > 1:
        print("所有文件汇总")
    else:
        print("文件汇总")
    print("=" * 110)
    total_all = sum(s['total'] for s in all_stats.values())
    positive_all = sum(s['positive'] for s in all_stats.values())
    negative_all = sum(s['negative'] for s in all_stats.values())
    print(f"\n总样本数: {total_all:,}")
    if total_all > 0:
        print(f"正样本 (y=1): {positive_all:,} ({positive_all/total_all*100:.2f}%)")
        print(f"负样本 (y=0): {negative_all:,} ({negative_all/total_all*100:.2f}%)")
    else:
        print("正样本 (y=1): 0")
        print("负样本 (y=0): 0")
