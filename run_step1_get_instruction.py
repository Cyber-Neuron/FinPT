#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import os
import sys
import json
import logging
import argparse
import numpy as np

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from huggingface_hub import snapshot_download
sub_instruct_cd1="""
write the narrative in the context of credit-card delinquency/default risk. If the input includes any payment-history, utilization, statement/billing, or credit-limit related fields, explicitly describe repayment behavior consistency, revolving balance pressure, and any signs of recent stress using the exact feature_name=value pairs. Avoid inventing field semantics that are not clearly implied by the feature names; when fields look like identifiers (e.g., id-like), mention them as identifiers only.
"""
sub_instruct_cd2="""
 treat the narrative as a monthly credit-card repayment risk profile. If present, explicitly contextualize LIMIT_BAL, PAY_0/PAY_2… style repayment-status fields, BILL_AMT* statement balances, and PAY_AMT* repayment amounts by describing whether repayment appears timely, whether balances are building, and whether payment amounts appear sufficient relative to billed amounts, always citing exact feature_name=value. If demographic fields like SEX/EDUCATION/MARRIAGE/AGE appear, mention them neutrally as background context without stereotypes. Do not output any decision or probability.
"""
sub_instruct_ld1="""
frame the narrative as a home-equity underwriting risk summary. If present, explicitly describe collateral/coverage signals (e.g., LOAN, MORTDUE, VALUE), capacity signals (e.g., DEBTINC), stability signals (e.g., YOJ, JOB), and credit-adversity signals (e.g., DEROG, DELINQ, NINQ, CLAGE, CLNO) using exact feature_name=value. When you see REASON, interpret it only as the stated loan purpose category without adding extra claims. Keep a neutral, compliance-friendly tone and avoid making an approval/decline decision.
"""
sub_instruct_ld2="""
write the narrative as a consumer-loan risk profile emphasizing affordability and credit quality. If present, explicitly connect person_income with loan_amnt, loan_int_rate, and loan_percent_income (or similarly named affordability ratios) by describing payment burden pressure, always using exact feature_name=value. If loan_grade or loan_intent appears, mention them as underwriting segmentation/purpose signals only. If cb_person_default_on_file and cb_person_cred_hist_length appear, describe them as credit-history depth/adverse-record indicators without over-interpreting.
"""
sub_instruct_ld3="""
describe risk in a “first-instalment performance” context. If the input includes loan structure and collateral coverage fields (e.g., ltv-like, asset_cost-like, disbursed_amount-like names), explicitly describe first-payment stress and leverage using exact feature_name=value. If identity/document flags or verification indicators appear (flag-like names), describe documentation completeness/verification strength cautiously and neutrally. If bureau score / account count / delinquency-history style fields appear, summarize credit depth and recent stress without assuming meanings beyond what feature names suggest.
"""
sub_instruct_cf1="""
write the narrative at the transaction level. If present, explicitly mention Time and Amount as transaction-timing and transaction-size context using exact feature_name=value. For V1–V28 (or similarly anonymized components), do not invent semantic meanings; instead describe them as anonymized signal patterns and note whether the combination appears unusual or extreme, while still listing exact feature_name=value in the text. Emphasize that this is fraud-risk screening, not creditworthiness.
"""
sub_instruct_cf2="""
adapt to whatever schema appears in the input: identify the transaction identifiers, timing/amount-like fields, merchant/channel/location-like fields, and any fraud/label-like indicator purely by their feature names. Do not assume the label name (it might be Class, fraud, is_fraud, etc.); if it is provided among features, mention it neutrally as the observed outcome field. Keep the narrative transaction-centric and avoid credit-default language unless the schema clearly indicates default rather than fraud.
"""
sub_instruct_cc1="""
frame the narrative as a retention-risk profile over a six-month horizon. If the input includes engagement/usage intensity, balance/relationship depth, tenure/vintage, product holding, or service interaction fields, describe signs of weakening engagement or dissatisfaction using exact feature_name=value. Avoid claiming “will churn” or “will not churn”; describe only risk signals and current state reflected by the features.
"""
sub_instruct_cc2="""
write the narrative as a retail-banking relationship summary. If present, explicitly describe relationship depth and stickiness using Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, and EstimatedSalary (or similar) with exact feature_name=value. If RowNumber/CustomerId/Surname appear, mention them as identifiers with no behavioral interpretation. If Exited appears, treat it as an observed churn outcome field and mention it neutrally without turning the output into a label prediction.
"""
sub_instruct_cc3="""
write the narrative as a subscription/service retention profile. If present, explicitly describe tenure and billing pressure using tenure-related, MonthlyCharges, and TotalCharges fields with exact feature_name=value. If service bundle fields appear (phone/internet/add-ons) or contract/payment method fields appear, describe potential switching risk and price sensitivity cautiously, grounded only in feature names. If Churn appears, mention it neutrally as the recorded churn status, not as a model decision.
"""
system_instruct="""
You are a senior credit risk analyst. Convert the applicant’s structured risk-control features into ONE coherent English narrative paragraph for model training.
"""
base_instruct= system_instruct+"""


Hard requirements:

Output plain text only (no JSON, no markdown, no bullet points).

The narrative must be natural, professional, and realistic in a financial risk-control context.

You must explicitly include the exact feature names and their corresponding feature values exactly as they appear in the input.

Do not rename, translate, or paraphrase feature keys. Keep them verbatim (for example, days_of_employment, contract_type).

Do not invent or infer any information that is not supported by the provided features.

If a feature is missing, null, or not provided, do NOT mention it at all.

Every feature that appears in the input must be mentioned at least once in the narrative.

You may group features logically (demographics, income and repayment pressure, employment stability, residence and assets, credit bureau behavior, social and communication signals, regional consistency, application behavior, document completeness), but all content must be written as a single continuous paragraph.


"""
instruct_dic={"cd1":sub_instruct_cd1, "cd2":sub_instruct_cd2, "ld1":sub_instruct_ld1, "ld2":sub_instruct_ld2, "ld3":sub_instruct_ld3, "cf1":sub_instruct_cf1, "cf2":sub_instruct_cf2, "cc1":sub_instruct_cc1, "cc2":sub_instruct_cc2, "cc3":sub_instruct_cc3}

def convert_finbench_to_local_dataset(snapshot_path, ds_name, local_cache_dir, logger):
    """
    Convert yuweiyin/FinBench raw files to local HuggingFace dataset format.
    Based on the FinBenchDataset._generate_examples logic.
    """
    task_path = os.path.join(snapshot_path, "data", ds_name)
    local_dataset_path = os.path.join(local_cache_dir, "datasets", "yuweiyin--FinBench", ds_name)
    
    # Check if already converted
    if os.path.exists(local_dataset_path):
        logger.info(f">>> Dataset already converted at {local_dataset_path}, skipping conversion")
        return local_dataset_path
    
    logger.info(f">>> Converting {ds_name} from {task_path} to {local_dataset_path}")
    os.makedirs(local_dataset_path, exist_ok=True)
    
    # File paths
    file_paths = {
        "train": {
            "X_ml": os.path.join(task_path, "X_train.npy"),
            "X_ml_unscale": os.path.join(task_path, "X_train_unscale.npy"),
            "y": os.path.join(task_path, "y_train.npy"),
            "instruction": os.path.join(task_path, "instruction_for_profile_X_train.jsonl"),
            "profile": os.path.join(task_path, "profile_X_train.jsonl"),
        },
        "validation": {
            "X_ml": os.path.join(task_path, "X_val.npy"),
            "X_ml_unscale": os.path.join(task_path, "X_val_unscale.npy"),
            "y": os.path.join(task_path, "y_val.npy"),
            "instruction": os.path.join(task_path, "instruction_for_profile_X_validation.jsonl"),
            "profile": os.path.join(task_path, "profile_X_validation.jsonl"),
        },
        "test": {
            "X_ml": os.path.join(task_path, "X_test.npy"),
            "X_ml_unscale": os.path.join(task_path, "X_test_unscale.npy"),
            "y": os.path.join(task_path, "y_test.npy"),
            "instruction": os.path.join(task_path, "instruction_for_profile_X_test.jsonl"),
            "profile": os.path.join(task_path, "profile_X_test.jsonl"),
        },
    }
    
    stat_ml_path = os.path.join(task_path, "stat_dict.json")
    
    # Load metadata
    if not os.path.exists(stat_ml_path):
        raise FileNotFoundError(f"stat_dict.json not found at {stat_ml_path}")
    stat_ml_dict = json.load(open(stat_ml_path, "r"))
    
    datasets_dict = {}
    
    # Process each split
    for split_name, paths in file_paths.items():
        # Check if files exist
        if not os.path.exists(paths["X_ml"]):
            logger.warning(f">>> {split_name} split files not found, skipping")
            continue
        
        # Load numpy arrays
        X_ml_np = np.load(paths["X_ml"], allow_pickle=True)
        X_ml_unscale_np = np.load(paths["X_ml_unscale"], allow_pickle=True)
        y_np = np.load(paths["y"], allow_pickle=True)
        
        assert len(X_ml_np) == len(y_np), f"len(X_ml_np) = {len(X_ml_np)}; len(y_np) = {len(y_np)}"
        
        # Load JSONL files (if they exist)
        X_instruction_for_profile_jsonl = []
        if os.path.exists(paths["instruction"]):
            with open(paths["instruction"], mode="r", encoding="utf-8") as f_in:
                for line in f_in:
                    cur_jsonl = json.loads(line.strip())
                    X_instruction_for_profile_jsonl.append(str(cur_jsonl).strip())
        else:
            # Create empty strings if file doesn't exist
            X_instruction_for_profile_jsonl = [""] * len(X_ml_np)
        
        X_profile_jsonl = []
        if os.path.exists(paths["profile"]):
            with open(paths["profile"], mode="r", encoding="utf-8") as f_in:
                for line in f_in:
                    cur_jsonl = json.loads(line.strip())
                    X_profile_jsonl.append(str(cur_jsonl).strip())
        else:
            # Create empty strings if file doesn't exist
            X_profile_jsonl = [""] * len(X_ml_np)
        
        total = len(X_ml_np)
        if len(X_instruction_for_profile_jsonl) != total:
            logger.warning(f">>> instruction JSONL length mismatch, padding with empty strings")
            X_instruction_for_profile_jsonl.extend([""] * (total - len(X_instruction_for_profile_jsonl)))
        if len(X_profile_jsonl) != total:
            logger.warning(f">>> profile JSONL length mismatch, padding with empty strings")
            X_profile_jsonl.extend([""] * (total - len(X_profile_jsonl)))
        
        # Create dataset
        data_items = []
        for idx in range(total):
            data_item = {
                "X_ml": X_ml_np[idx].tolist() if isinstance(X_ml_np[idx], np.ndarray) else list(X_ml_np[idx]),
                "X_ml_unscale": X_ml_unscale_np[idx].tolist() if isinstance(X_ml_unscale_np[idx], np.ndarray) else list(X_ml_unscale_np[idx]),
                "y": int(y_np[idx]),
                "num_classes": stat_ml_dict["num_classes"],
                "num_features": stat_ml_dict["num_features"],
                "num_idx": stat_ml_dict["num_idx"],
                "cat_idx": stat_ml_dict["cat_idx"],
                "cat_dim": stat_ml_dict["cat_dim"],
                "cat_str": stat_ml_dict["cat_str"],
                "col_name": stat_ml_dict["col_name"],
                "X_instruction_for_profile": X_instruction_for_profile_jsonl[idx] if idx < len(X_instruction_for_profile_jsonl) else "",
                "X_profile": X_profile_jsonl[idx] if idx < len(X_profile_jsonl) else "",
            }
            data_items.append(data_item)
        
        datasets_dict[split_name] = Dataset.from_list(data_items)
        logger.info(f">>> Converted {split_name} split: {len(data_items)} samples")
    
    # Create DatasetDict and save
    dataset_dict = DatasetDict(datasets_dict)
    dataset_dict.save_to_disk(local_dataset_path)
    logger.info(f">>> Saved converted dataset to {local_dataset_path}")
    
    return local_dataset_path


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s - %(name)s] -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Step1 Get_Instruction Args")
    args = parser.parse_args()

    logger.info(args)

    cache_dir = os.path.expanduser("~/.cache/huggingface/")
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    local_cache_dir = cache_dir

    profile_root_dir = os.path.join("./data/profile")
    ds_name_list =["cd1", "cd2", "ld1", "ld2", "ld3", "cf1", "cf2", "cc1", "cc2", "cc3"]
    
    # Download yuweiyin/FinBench dataset
    logger.info(">>> Downloading yuweiyin/FinBench dataset...")
    snapshot_path = snapshot_download(
        repo_id="yuweiyin/FinBench",
        repo_type="dataset",
        local_dir_use_symlinks=False
    )
    logger.info(f">>> Downloaded to: {snapshot_path}")
    
    for ds_name in ds_name_list:
        logger.info(f"\n\n>>> ds_name: {ds_name}")
        profile_dir = os.path.join(profile_root_dir, ds_name)
        os.makedirs(profile_dir, exist_ok=True)
        
        # Convert to local dataset format if needed
        local_dataset_path = convert_finbench_to_local_dataset(snapshot_path, ds_name, local_cache_dir, logger)
        
        # Load the converted dataset
        logger.info(f">>> Loading dataset from {local_dataset_path}")
        data = load_from_disk(local_dataset_path)  # DatasetDict

        if "train" in data:
            logger.info(f">>> len(data['train']) = {len(data['train'])}")
        if "validation" in data:
            logger.info(f">>> len(data['validation']) = {len(data['validation'])}")
        if "test" in data:
            logger.info(f">>> len(data['test']) = {len(data['test'])}")

        for data_split in ["train", "validation", "test"]:
            if data_split in data:
                cur_data = data[data_split]
            else:
                logger.info(f">>> >>> {data_split} NOT in data")
                continue  # should NOT enter here

            instruction_path = os.path.join(profile_dir, f"instruction_for_profile_X_{data_split}.jsonl")
            
            # Analyze features from first instance
            first_instance = cur_data[0]
            num_idx = first_instance["num_idx"]
            cat_idx = first_instance["cat_idx"]
            col_name = first_instance["col_name"]
            num_idx_set = set(num_idx)
            cat_idx_set = set(cat_idx)
            
            logger.info(f">>> >>> Feature Analysis for {ds_name}/{data_split}:")
            logger.info(f">>> >>> Total features: {len(col_name)}")
            logger.info(f">>> >>> Numerical features ({len(num_idx)}): {[col_name[i] for i in num_idx]}")
            logger.info(f">>> >>> Categorical features ({len(cat_idx)}): {[col_name[i] for i in cat_idx]}")
            
            # Check for skipped features
            skipped_features = []
            for col_idx, cur_col_name in enumerate(col_name):
                if ds_name == "cf3" and cur_col_name[:2] == "x_":
                    skipped_features.append(cur_col_name)
                elif col_idx not in num_idx_set and col_idx not in cat_idx_set:
                    skipped_features.append(cur_col_name)
            
            if skipped_features:
                logger.info(f">>> >>> Skipped features ({len(skipped_features)}): {skipped_features}")
            else:
                logger.info(f">>> >>> All features are used (none skipped)")
            
            # Get metadata (same for all instances)
            cat_str = first_instance["cat_str"]
            col_idx_2_cat_idx = dict({
                col_idx: cat_idx for cat_idx, col_idx in enumerate(cat_idx)
            })
            
            with open(instruction_path, mode="w", encoding="utf-8") as fp_out:
                for instance_idx, instance in enumerate(cur_data):
                    # X_ml = instance["X_ml"]  # List[float] (The tabular data array of the current instance)
                    X_ml_unscale = instance["X_ml_unscale"]  # List[float] (Scaled tabular data array)
                    # y = instance["y"]  # int (The label / ground-truth)
                    # num_classes = instance["num_classes"]  # int (The total number of classes)
                    # num_features = instance["num_features"]  # int (The total number of features)
                    assert len(X_ml_unscale) == len(num_idx) + len(cat_idx) == len(col_name)
                    # Construct the customer profiles
                    instruction = base_instruct + instruct_dic[ds_name]+"""
                    Do not output any approval, rejection, score, probability, or risk label.

                    Do not explain your reasoning process or steps.

                    Input:
                    You will receive numerical_features and categorical_features as input.

                    Now generate one narrative paragraph using the following input.
                    """
                    for col_idx, x in enumerate(X_ml_unscale):
                        cur_col_name = col_name[col_idx]
                        if ds_name == "cf3" and cur_col_name[:2] == "x_":
                            continue  # skip features without certain meaning
                        if col_idx in num_idx_set:
                            instruction += f"{cur_col_name}: {x};\n"
                        elif col_idx in cat_idx_set:
                            x = int(x)
                            cat_str_idx = col_idx_2_cat_idx[col_idx]
                            cat_string = cat_str[cat_str_idx][x]
                            instruction += f"{cur_col_name}: {cat_string};\n"
                        else:
                            continue  # should NOT enter here
                    instruction = instruction.replace("_", " ")
                    ins_json = json.dumps(instruction.strip())
                    fp_out.write(ins_json + "\n")
                    
                    # Show first instruction as example
                    if instance_idx == 0:
                        logger.info(f">>> >>> Sample instruction (first instance):")
                        logger.info(f">>> >>> {instruction[:500]}..." if len(instruction) > 500 else f">>> >>> {instruction}")

        # except Exception as e:
        #     logger.info(f"Exception: {e}")
        #     continue  # should NOT enter here

    sys.exit(0)
