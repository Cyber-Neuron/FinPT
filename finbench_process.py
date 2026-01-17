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
TASK_CONTEXT_V1 = {
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
TASK_CONTEXT = {
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
    label_sem = TASK_LABEL_SEMANTICS.get(
        task_name, "Label 1 = event occurs. Label 0 = event does not occur."
    )
    return (
        f"Task: {task_name}.\n"
        f"{context}\n"
        f"{label_sem}\n\n"
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

    args = parser.parse_args()

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

    # =========================
    # Process each task
    # =========================
    for task in tasks:
        task_path = os.path.join(snapshot_path, task)
        ds = load_from_disk(task_path)  # DatasetDict

        # ---- merge train + validation -> train ----
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
            validation_sets.append(
                ds["validation"].map(
                    function=make_map_fn(task, "validation"),
                    with_indices=True,
                    remove_columns=ds["validation"].column_names,
                )
            )

        if train_parts:
            train_sets.append(concatenate_datasets(train_parts))

        # ---- test only for final evaluation ----
        if "test" in ds:
            test_sets.append(
                ds["test"].map(
                    function=make_map_fn(task, "test"),
                    with_indices=True,
                    remove_columns=ds["test"].column_names,
                )
            )

    # =========================
    # Save parquet
    # =========================
    small_train_sets = [ds.select(range(min(5000, len(ds)))) for ds in train_sets]
    small_test_sets = [ds.select(range(min(500, len(ds)))) for ds in test_sets]
    small_validation_sets = [ds.select(range(min(500, len(ds)))) for ds in validation_sets]
    if train_sets:
        concatenate_datasets(train_sets).to_parquet(
            os.path.join(local_save_dir, "train_v2.parquet")
        )
        concatenate_datasets(validation_sets).to_parquet(
            os.path.join(local_save_dir, "validation_v2.parquet")
        )
        concatenate_datasets(small_train_sets).to_parquet(
            os.path.join(local_save_dir, "small_train_v2.parquet")
        )
        concatenate_datasets(small_validation_sets).to_parquet(
            os.path.join(local_save_dir, "small_validation_v2.parquet")
        )

    if test_sets:
        concatenate_datasets(test_sets).to_parquet(
            os.path.join(local_save_dir, "test_v2.parquet")
        )
        concatenate_datasets(small_test_sets).to_parquet(
            os.path.join(local_save_dir, "small_test_v2.parquet")
        )

    # =========================
    # Optional HDFS copy
    # =========================
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)

    print("Preprocessing finished.")
    print(f"Saved to: {local_save_dir}")
