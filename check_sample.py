from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from collections import defaultdict

# ===== 配置 =====
DATA_DIR = "/home/dan/data/finbench_verl"
MODEL_NAME = "Qwen/Qwen3-8B"
BATCH_SIZE = 1024
THRESHOLD = 512

# ===== 加载 parquet =====
ds = load_dataset(
    "parquet",
    data_files={
        "train": f"{DATA_DIR}/train_v2.parquet",
        "test": f"{DATA_DIR}/test_v2.parquet",
    },
)

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

    return results

# -------------------------------------------------------------------
# 打印结果
# -------------------------------------------------------------------
def print_table(split, rows):
    print("=" * 110)
    print(f"[{split.upper()}] prompt length distribution by task × label")
    print("-" * 110)
    print(f"{'Task':>4} | {'y':>1} | {'N':>7} | {'max':>4} | {'p95':>6} | {'p99':>6} | {'>512':>5}")
    print("-" * 110)

    for r in sorted(rows, key=lambda x: (x["task"], x["y"])):
        print(
            f"{r['task']:>4} | {r['y']:>1} | {r['count']:7d} | "
            f"{r['max']:4d} | {r['p95']:6.1f} | {r['p99']:6.1f} | {r['>512']:5d}"
        )

# ===== 运行 =====
train_rows = summarize_split("train")
test_rows = summarize_split("test")

print_table("train", train_rows)
print_table("test", test_rows)
