# finbench_process.py
# Preprocess FinBench for verl GRPO / DAPO
# - snapshot_download + load_from_disk
# - merge train + validation -> train
# - keep only train / test
# - remove all original tabular columns
# - support CoT with ####
# - store imbalance weights in extra_info

import argparse
import os
import re
import random
from typing import Optional, List

from huggingface_hub import snapshot_download
from datasets import load_from_disk, concatenate_datasets

from verl.utils.hdfs_io import copy, makedirs


# =========================
# FinBench task list
# =========================
ALL_TASKS = [
    "cd1", "cd2",
    "ld1", "ld2", "ld3",
    "cf1", "cf2",
    "cc1", "cc2", "cc3",
]

# =========================
# Positive-class weights (task-level)
# Tune later if needed
# =========================
# POS_WEIGHT = {
#     "cf1": 8.0,   # 0.67%
#     "cd1": 4.0,   # ~7%
#     "ld1": 3.0,   # ~9%
#     "cf2": 2.0,
#     "cd2": 1.5,
#     "ld2": 1.5,
#     "ld3": 1.0,
#     "cc1": 1.0,
#     "cc2": 1.0,
#     "cc3": 1.0,
# }
POS_WEIGHT = {
    "cf1": 12.2,  # 0.67% -> sqrt(148.3)
    "cd1": 3.6,   # 7.0%  -> sqrt(13.3)
    "ld1": 3.2,   # 8.9%  -> sqrt(10.2)
    "cf2": 4.0,   # 6.0%  -> sqrt(15.7)
    "cd2": 1.9,   # 22.3% -> sqrt(3.48)
    "ld2": 1.9,   # 21.7% -> sqrt(3.61)
    "ld3": 1.9,   # 21.6% -> sqrt(3.63)
    "cc1": 1.8,   # 23.5% -> sqrt(3.26)
    "cc2": 2.0,   # 20.8% -> sqrt(3.81)
    "cc3": 1.7,   # 26.1% -> sqrt(2.83)
}


# =========================
# Utility: extract final answer
# (used by rule reward later)
# =========================
def extract_final_after_hashes(text: str) -> Optional[str]:
    m = re.search(r"####\s*([01])\b", text)
    return m.group(1) if m else None
TASK_CONTEXT = {
    "cd1": "Predict credit-card default risk. Label=1 means default risk; Label=0 means no default. Positives are rare (~6%). Assign 1 only with clear and strong default evidence; do not label 1 on weak or ambiguous signals.",
    "cd2": "Predict credit-card default risk. Label=1 means default risk; Label=0 means no default. Positives are moderately common (~22%). Assign 1 when repayment behavior or credit trends reasonably indicate elevated risk.",
    "ld1": "Predict home-equity loan default risk. Label=1 means default risk; Label=0 means low risk. Positives are uncommon (~9%). Assign 1 only with concrete repayment stress or credit weakness signals.",
    "ld2": "Predict loan default risk. Label=1 means default risk; Label=0 means low risk. Positives are moderately common (~22%). Assign 1 when multiple indicators suggest elevated default risk.",
    "ld3": "Predict vehicle-loan default risk. Label=1 means default risk; Label=0 means low risk. Positives are moderately common (~22%). Assign 1 for early payment stress or affordability issues.",
    "cf1": "Detect credit-card fraud. Label=1 means fraud; Label=0 means legitimate. Fraud is extremely rare (~1%). Assign 1 only for strong, clearly fraudulent patterns; do not flag explainable or mild anomalies.",
    "cf2": "Detect credit-card fraud. Label=1 means fraud; Label=0 means legitimate. Fraud is uncommon (~6%). Assign 1 only when multiple strong anomaly or inconsistency signals are present.",
    "cc1": "Predict customer churn. Label=1 means churn risk; Label=0 means low risk. Churn is common (~23%). Assign 1 when engagement or activity shows meaningful decline.",
    "cc2": "Predict customer churn. Label=1 means churn risk; Label=0 means low risk. Churn is moderately common (~20%). Assign 1 for clear disengagement or instability patterns.",
    "cc3": "Predict customer churn. Label=1 means churn risk; Label=0 means low risk. Churn is common (~25%). Use balanced judgment based on contract, billing, and usage patterns."
}

TASK_CONTEXT_ORIGINAL = {
    "cd1": "Predict credit-card default risk for a bank customer using demographic and credit-related information.",
    "cd2": "Predict credit-card default using demographics, credit data, repayment history, and bill statements (Taiwan, 2005).",
    "ld1": "Evaluate home-equity loan default / serious delinquency risk from applicant and loan information (HMEQ).",
    "ld2": "Evaluate loan default risk using bureau-style attributes (income, employment, loan details, credit history).",
    "ld3": "Predict vehicle-loan default risk using borrower, loan, and loan-history information (early repayment/first EMI).",
    "cf1": "Detect credit-card fraud from multi-month usage patterns; fraud is rare.",
    "cf2": "Detect credit-card fraud using high-dimensional cardholder usage and account/profile attributes (no raw transaction sequence; focus on aggregate anomalies and consistency",
    "cc1": "Predict customer churn/attrition from customer demographics and past activity.",
    "cc2": "Predict bank customer churn from account and customer attributes (e.g., tenure, balance, demographics).",
    "cc3": "Predict telco customer churn using demographics, subscribed services, and billing/account info (IBM sample).",
}
TASK_CONTEXT_LONGER = {
    "cd1": "Predict credit-card default from bank customer demographics and credit attributes (incl. credit score/limit-style fields).",
    "cd2": "Predict next-month credit-card default (Taiwan, 2005) from demographics + 6-month repayment status + bill/payment amounts.",
    "ld1": "Predict home-equity loan serious delinquency (HMEQ) from loan size, collateral/value, and delinquency/credit history fields.",
    "ld2": "Predict personal-loan default from bureau-style attributes (age/income/employment, loan amount/rate/intent, credit history/default flag).",
    "ld3": "Predict vehicle-loan default on the first EMI using borrower profile, loan terms, and application/credit-history attributes.",
    "cf1": "Detect transaction-level card fraud from anonymized features (Time/Amount + PCA-like V-features); fraud is extremely rare.",
    "cf2": "Detect profile/account-level card fraud from high-dimensional application + bureau-style attributes (no raw transaction sequence; focus on aggregate anomalies/consistency).",
    "cc1": "Predict bank customer churn (next 6 months) from demographics, tenure/vintage, balance, product holdings, and activity/transaction-status signals.",
    "cc2": "Predict bank churn from account/customer attributes (credit score, geography, age, tenure, balance, products, activity).",
    "cc3": "Predict telco churn from demographics, subscribed services, contract type, billing/charges, and payment method.",
}

TASK_LABEL_SEMANTICS = {
    # Default / Fraud / Churn semantics
    "cd1": "Label 1 = will default. Label 0 = will not default.",
    "cd2": "Label 1 = will default. Label 0 = will not default.",
    "ld1": "Label 1 = will default / be seriously delinquent. Label 0 = will not.",
    "ld2": "Label 1 = will default. Label 0 = will not default.",
    "ld3": "Label 1 = will default. Label 0 = will not default.",
    "cf1": "Label 1 = fraud. Label 0 = legitimate.",
    "cf2": "Label 1 = fraud. Label 0 = legitimate.",
    "cc1": "Label 1 = churn. Label 0 = stay.",
    "cc2": "Label 1 = churn. Label 0 = stay.",
    "cc3": "Label 1 = churn. Label 0 = stay.",
}

def build_prompt(task_name: str, profile: str) -> str:
    context = TASK_CONTEXT.get(task_name, "Perform a binary risk assessment task.")
    # label_sem = TASK_LABEL_SEMANTICS.get(
    #     task_name, "Label 1 = event occurs. Label 0 = event does not occur."
    # )
    return (
        f"Task: {task_name}.\n"
        f"{context}\n"
        # f"{label_sem}\n\n"
        "Reason step by step, then output the final label after '####' as either #### 0 or #### 1.\n\n"
        f"Customer profile:\n{profile}"
    )



# =========================
# Map function
# =========================
def make_map_fn(task_name: str, split: str):
    def process_fn(example, idx):
        profile = example["X_profile"]
        y = int(example["y"])

        return {
            "data_source": "dhugs/FinBench/" + task_name,
            "prompt": [
                {
                    "role": "user",
                    "content": build_prompt(task_name, profile),
                }
            ],
            "ability": "finance_risk",
            "reward_model": {
                "style": "rule",
                "ground_truth": str(y),
            },
            "extra_info": {
                "task": task_name,
                "split": split,
                "index": idx,
                "y": y,
                "pos_weight": POS_WEIGHT.get(task_name, 1.0),
            },
        }

    return process_fn


# =========================
# Main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Tasks to include (default: all FinBench tasks)",
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/finbench_verl",
        help="Directory to save parquet files",
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="Optional HDFS directory to copy output",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v2",
        choices=["v2", "v3", "v4", "v5", "v6"],
        help="Version to generate: v2 (merge all train+validation), v3 (keep train/val/test separate, full data), v4 (half train + half validation), v5 (all positive + 2N negative), or v6 (same as v5 but merge all into train)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for V4/V5/V6 random sampling (default: 42)",
    )
    parser.add_argument(
        "--separate_datasets",
        action="store_true",
        help="Save each dataset separately to data/{task_name}/ folder",
    )

    args = parser.parse_args()
    
    # Set random seed for reproducibility (V3, V4, V5, and V6 use random sampling)
    if args.version in ["v3", "v4", "v5", "v6"]:
        random.seed(args.seed)

    tasks: List[str] = args.tasks if args.tasks is not None else ALL_TASKS
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    # =========================
    # Download FinBench once
    # =========================
    snapshot_path = snapshot_download(
        repo_id="dhugs/FinBench",
        repo_type="dataset",
    )

    train_sets = []
    test_sets = []
    validation_sets = []
    
    # Store per-task datasets for separate output
    per_task_data = {}  # {task_name: {"train": dataset, "val": dataset, "test": dataset}}

    # =========================
    # Process each task
    # =========================
    for task_idx, task in enumerate(tasks):
        task_path = os.path.join(snapshot_path, task)
        ds = load_from_disk(task_path)  # DatasetDict
        
        # Per-task datasets (initialize to None for separate_datasets mode)
        task_train = None
        task_val = None
        task_test = None

        if args.version == "v2":
            # ---- V2: merge train + validation -> train ----
            train_parts = []

            if "train" in ds:
                train_parts.append(
                    ds["train"].map(
                        function=make_map_fn(task, "train"),
                        with_indices=True,
                        remove_columns=ds["train"].column_names,
                    )
                )

            if "validation" in ds:
                mapped_val = ds["validation"].map(
                    function=make_map_fn(task, "validation"),
                    with_indices=True,
                    remove_columns=ds["validation"].column_names,
                )
                validation_sets.append(mapped_val)
                task_val = mapped_val

            if train_parts:
                task_train = concatenate_datasets(train_parts)
                train_sets.append(task_train)
        
        elif args.version == "v3":
            # ---- V3: keep train, validation, test separate (full data) ----
            if "train" in ds:
                task_train = ds["train"].map(
                    function=make_map_fn(task, "train"),
                    with_indices=True,
                    remove_columns=ds["train"].column_names,
                )
                train_sets.append(task_train)
            
            if "validation" in ds:
                task_val = ds["validation"].map(
                    function=make_map_fn(task, "validation"),
                    with_indices=True,
                    remove_columns=ds["validation"].column_names,
                )
                validation_sets.append(task_val)
        
        elif args.version == "v4":
            # ---- V4: randomly use half of train + half of validation for training ----
            train_parts = []
            validation_parts = []

            if "train" in ds:
                mapped_train = ds["train"].map(
                    function=make_map_fn(task, "train"),
                    with_indices=True,
                    remove_columns=ds["train"].column_names,
                )
                # Randomly sample half of train
                half_train_size = len(mapped_train) // 2
                train_indices = random.sample(range(len(mapped_train)), half_train_size)
                train_parts.append(mapped_train.select(train_indices))

            if "validation" in ds:
                mapped_validation = ds["validation"].map(
                    function=make_map_fn(task, "validation"),
                    with_indices=True,
                    remove_columns=ds["validation"].column_names,
                )
                # Randomly sample half of validation
                half_val_size = len(mapped_validation) // 2
                all_val_indices = list(range(len(mapped_validation)))
                random.shuffle(all_val_indices)
                val_train_indices = all_val_indices[:half_val_size]
                val_val_indices = all_val_indices[half_val_size:]
                validation_parts.append(mapped_validation.select(val_train_indices))
                # Keep the other half as validation set
                task_val = mapped_validation.select(val_val_indices)
                validation_sets.append(task_val)

            # Combine half train + half validation for training
            if train_parts and validation_parts:
                task_train = concatenate_datasets(train_parts + validation_parts)
                train_sets.append(task_train)
            elif train_parts:
                task_train = concatenate_datasets(train_parts)
                train_sets.append(task_train)
            elif validation_parts:
                task_train = concatenate_datasets(validation_parts)
                train_sets.append(task_train)
        
        elif args.version == "v5":
            # ---- V5: all positive samples + 2N negative samples (N = num positive) ----
            train_parts = []
            validation_parts = []
            
            # Collect all train and validation data first
            all_train_data = []
            all_val_data = []
            
            if "train" in ds:
                all_train_data.append(
                    ds["train"].map(
                        function=make_map_fn(task, "train"),
                        with_indices=True,
                        remove_columns=ds["train"].column_names,
                    )
                )
            
            if "validation" in ds:
                all_val_data.append(
                    ds["validation"].map(
                        function=make_map_fn(task, "validation"),
                        with_indices=True,
                        remove_columns=ds["validation"].column_names,
                    )
                )
            
            if all_train_data or all_val_data:
                # Combine all train and validation data
                combined_data = concatenate_datasets(
                    (all_train_data + all_val_data) if all_train_data and all_val_data
                    else (all_train_data if all_train_data else all_val_data)
                )
                
                # Filter positive and negative samples efficiently
                positive_samples = combined_data.filter(lambda x: x["extra_info"]["y"] == 1)
                negative_samples_all = combined_data.filter(lambda x: x["extra_info"]["y"] == 0)
                
                N = len(positive_samples)  # Number of positive samples
                
                # Randomly sample 2N negative samples (or all if we have fewer)
                num_negative_to_sample = min(2 * N, len(negative_samples_all))
                if num_negative_to_sample > 0:
                    negative_indices = list(range(len(negative_samples_all)))
                    sampled_indices = random.sample(negative_indices, num_negative_to_sample)
                    negative_samples = negative_samples_all.select(sampled_indices)
                    
                    # Combine all positive + 2N negative for training
                    train_sets.append(concatenate_datasets([positive_samples, negative_samples]))
                    
                    # Keep remaining negative samples as validation (if any)
                    remaining_indices = [idx for idx in negative_indices if idx not in sampled_indices]
                    if remaining_indices:
                        validation_sets.append(negative_samples_all.select(remaining_indices))
                else:
                    # Only positive samples (unlikely but handle edge case)
                    train_sets.append(positive_samples)
        
        elif args.version == "v6":
            # ---- V6: same as V5 but merge all train and validation into train (no separate validation) ----
            # Collect all train and validation data first
            all_train_data = []
            all_val_data = []
            
            if "train" in ds:
                all_train_data.append(
                    ds["train"].map(
                        function=make_map_fn(task, "train"),
                        with_indices=True,
                        remove_columns=ds["train"].column_names,
                    )
                )
            
            if "validation" in ds:
                all_val_data.append(
                    ds["validation"].map(
                        function=make_map_fn(task, "validation"),
                        with_indices=True,
                        remove_columns=ds["validation"].column_names,
                    )
                )
            
            if all_train_data or all_val_data:
                # Combine all train and validation data
                combined_data = concatenate_datasets(
                    (all_train_data + all_val_data) if all_train_data and all_val_data
                    else (all_train_data if all_train_data else all_val_data)
                )
                
                # Filter positive and negative samples efficiently
                positive_samples = combined_data.filter(lambda x: x["extra_info"]["y"] == 1)
                negative_samples_all = combined_data.filter(lambda x: x["extra_info"]["y"] == 0)
                
                N = len(positive_samples)  # Number of positive samples
                
                # Randomly sample 2N negative samples (or all if we have fewer)
                num_negative_to_sample = min(2 * N, len(negative_samples_all))
                if num_negative_to_sample > 0:
                    negative_indices = list(range(len(negative_samples_all)))
                    sampled_indices = random.sample(negative_indices, num_negative_to_sample)
                    negative_samples = negative_samples_all.select(sampled_indices)
                    
                    # Combine all positive + 2N negative for training (merge everything into train)
                    task_train = concatenate_datasets([positive_samples, negative_samples])
                    train_sets.append(task_train)
                else:
                    # Only positive samples (unlikely but handle edge case)
                    task_train = positive_samples
                    train_sets.append(task_train)

        # ---- test only for final evaluation ----
        if "test" in ds:
            mapped_test = ds["test"].map(
                function=make_map_fn(task, "test"),
                with_indices=True,
                remove_columns=ds["test"].column_names,
            )
            test_sets.append(mapped_test)
            task_test = mapped_test
        
        # Store per-task data for separate output
        if args.separate_datasets:
            per_task_data[task] = {
                "train": task_train,
                "val": task_val,
                "test": task_test,
            }

    # =========================
    # Save parquet
    # =========================
    version_suffix = args.version
    
    if args.separate_datasets:
        # Save each dataset separately to data/{task_name}/ folder
        for task_name, task_datasets in per_task_data.items():
            task_data_dir = os.path.join(local_save_dir, "data", task_name)
            os.makedirs(task_data_dir, exist_ok=True)
            
            # Save full datasets
            if task_datasets["train"] is not None:
                task_datasets["train"].to_parquet(
                    os.path.join(task_data_dir, f"train_{version_suffix}.parquet")
                )
                # Save small train (30% sample)
                small_train_size = max(1, int(len(task_datasets["train"]) * 0.3))
                small_train_indices = random.sample(range(len(task_datasets["train"])), small_train_size)
                task_datasets["train"].select(small_train_indices).to_parquet(
                    os.path.join(task_data_dir, f"small_train_{version_suffix}.parquet")
                )
            
            if task_datasets["val"] is not None:
                task_datasets["val"].to_parquet(
                    os.path.join(task_data_dir, f"validation_{version_suffix}.parquet")
                )
                # Save small validation (30% sample)
                small_val_size = max(1, int(len(task_datasets["val"]) * 0.3))
                small_val_indices = random.sample(range(len(task_datasets["val"])), small_val_size)
                task_datasets["val"].select(small_val_indices).to_parquet(
                    os.path.join(task_data_dir, f"small_validation_{version_suffix}.parquet")
                )
            
            if task_datasets["test"] is not None:
                task_datasets["test"].to_parquet(
                    os.path.join(task_data_dir, f"test_{version_suffix}.parquet")
                )
                # Save small test (30% sample)
                small_test_size = max(1, int(len(task_datasets["test"]) * 0.3))
                small_test_indices = random.sample(range(len(task_datasets["test"])), small_test_size)
                task_datasets["test"].select(small_test_indices).to_parquet(
                    os.path.join(task_data_dir, f"small_test_{version_suffix}.parquet")
                )
            
            print(f"Saved {task_name} to {task_data_dir}")
    else:
        # Original behavior: save all datasets merged together
        # Randomly sample 30% of each dataset for small datasets
        small_train_sets = []
        for ds in train_sets:
            sample_size = max(1, int(len(ds) * 0.3))
            indices = random.sample(range(len(ds)), sample_size)
            small_train_sets.append(ds.select(indices))
        
        small_test_sets = []
        for ds in test_sets:
            sample_size = max(1, int(len(ds) * 0.3))
            indices = random.sample(range(len(ds)), sample_size)
            small_test_sets.append(ds.select(indices))
        
        small_validation_sets = []
        for ds in validation_sets:
            sample_size = max(1, int(len(ds) * 0.3))
            indices = random.sample(range(len(ds)), sample_size)
            small_validation_sets.append(ds.select(indices))
        
        if train_sets:
            concatenate_datasets(train_sets).to_parquet(
                os.path.join(local_save_dir, f"train_{version_suffix}.parquet")
            )
            if validation_sets:
                concatenate_datasets(validation_sets).to_parquet(
                    os.path.join(local_save_dir, f"validation_{version_suffix}.parquet")
                )
            concatenate_datasets(small_train_sets).to_parquet(
                os.path.join(local_save_dir, f"small_train_{version_suffix}.parquet")
            )
            if small_validation_sets:
                concatenate_datasets(small_validation_sets).to_parquet(
                    os.path.join(local_save_dir, f"small_validation_{version_suffix}.parquet")
                )

        if test_sets:
            concatenate_datasets(test_sets).to_parquet(
                os.path.join(local_save_dir, f"test_{version_suffix}.parquet")
            )
            concatenate_datasets(small_test_sets).to_parquet(
                os.path.join(local_save_dir, f"small_test_{version_suffix}.parquet")
            )

    # =========================
    # Optional HDFS copy
    # =========================
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)

    print("Preprocessing finished.")
    print(f"Saved to: {local_save_dir}")
