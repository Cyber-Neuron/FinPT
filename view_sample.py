from datasets import load_dataset
import textwrap
import random
ALL_TASKS = [
    "cd1", "cd2",
    "ld1", "ld2", "ld3",
    "cf1", "cf2",
    "cc1", "cc2", "cc3",
]

# 改成你的实际路径
DATA_DIR = "/home/dan/data/finbench_verl"

ds = load_dataset("parquet", data_files={
    "train": f"{DATA_DIR}/train.parquet",
    "test": f"{DATA_DIR}/test.parquet",
})

def show_sample(split="train", idx=0, width=100):
    row = ds[split][idx]

    print("=" * 80)
    print(f"[{split}] sample #{idx}")
    print("- data_source:", row["data_source"])
    print("- ability:", row["ability"])
    print("- reward_model:", row["reward_model"])
    print("- extra_info:", row["extra_info"])
    print("\n--- PROMPT ---")
    print(textwrap.fill(row["prompt"][0]["content"], width=width))

def list_available_data_sources(split="train"):
    """List all available data_source values in the dataset"""
    data_sources = set()
    for idx in range(len(ds[split])):
        data_sources.add(ds[split][idx]["data_source"])
    return sorted(list(data_sources))

def show_sample_by_data_source_and_label(split="train", data_source=None, label=1, width=100):
    """Show a sample with specific data_source and label"""
    if data_source is None:
        print("Error: data_source must be specified")
        print(f"Available data_sources in {split}: {list_available_data_sources(split)}")
        return
    
    # Filter matching samples efficiently
    filtered_ds = ds[split].filter(
        lambda x: x["data_source"] == data_source and x["extra_info"]["y"] == label
    )
    
    if len(filtered_ds) == 0:
        print(f"No samples found with data_source='{data_source}' and label={label} in {split} split")
        print(f"Available data_sources in {split}: {list_available_data_sources(split)}")
        return
    
    # Show a random matching sample
    random_idx = random.randint(0, len(filtered_ds) - 1)
    row = filtered_ds[random_idx]
    
    print(f"Found {len(filtered_ds)} matching sample(s), showing a random one:")
    print("=" * 80)
    print(f"[{split}] sample (filtered)")
    print("- data_source:", row["data_source"])
    print("- ability:", row["ability"])
    print("- reward_model:", row["reward_model"])
    print("- extra_info:", row["extra_info"])
    print("\n--- PROMPT ---")
    print(textwrap.fill(row["prompt"][0]["content"], width=width))

# 看几条
# show_sample("train", random.randint(0, len(ds["train"])))
# show_sample("test", random.randint(0, len(ds["test"])))

# Show a specific data_source sample with label=1
# Replace "your_data_source" with the actual data_source value you want to view
# You can first check available data_sources: print(list_available_data_sources("train"))
# print(list_available_data_sources("train"))
show_sample_by_data_source_and_label("train", data_source="dhugs/FinBench/cc1", label=1)
