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

sub_instruct_cd1 = """
Write one continuous English paragraph as a credit-card delinquency and default risk narrative. The paragraph must be analysis-dominant: interpret what the provided values may suggest about repayment pressure, delinquency likelihood, and recent stress using cautious analyst language (may indicate, can signal, is often associated with). Include both potential strengths and potential concerns, and include explicit contrasts (e.g., while ..., however ...) without giving a final verdict.

If payment-history, utilization, statement or billing, or credit-limit related fields are present, explicitly discuss repayment consistency and timeliness, revolving balance pressure relative to stated limit capacity, and any signs of recent stress, grounding each interpretation in the specific numbers and categories shown in the input.

Avoid inventing meanings that are not reasonably implied by the field names. If a field appears to be an identifier, you may mention it briefly as a reference only and do not treat it as a risk driver.
"""

sub_instruct_cd2 = """
Write one continuous English paragraph as a monthly credit-card repayment risk profile. The paragraph must be analysis-dominant and value-grounded: interpret the provided monthly values to describe possible repayment behavior and pressure, including both supportive signals and risk signals, with at least two explicit contrasts, and no final verdict.

If present, contextualize LIMIT_BAL as stated limit capacity, PAY_0, PAY_2 and similar repayment-status fields as monthly repayment status indicators, BILL_AMT fields as statement balances, and PAY_AMT fields as repayment amounts. Discuss whether repayment appears timely, whether balances appear to be building across months, and whether payment amounts appear sufficient relative to billed amounts, explicitly referencing the values you are interpreting.

If demographic fields such as SEX, EDUCATION, MARRIAGE, or AGE appear, include them neutrally as background context only, without stereotypes or causal claims. Do not output any decision, probability, or overall risk label.
"""

sub_instruct_ld1 = """
Write one continuous English paragraph framed as a home-equity underwriting risk summary. The paragraph must be analysis-dominant and grounded in the provided values: discuss potential collateral coverage, affordability, stability, and credit adversity signals using cautious language, include both strengths and concerns, and use explicit contrasts, without a final verdict.

If present, interpret collateral and coverage fields such as LOAN, MORTDUE, and VALUE by discussing relative coverage and potential equity buffer using the stated amounts. Interpret capacity fields such as DEBTINC as affordability pressure, stability fields such as YOJ and JOB as employment and tenure context, and credit-adversity fields such as DEROG, DELINQ, NINQ, CLAGE, and CLNO as adverse history, delinquency, inquiry intensity, credit age, and account depth context consistent with their names, without adding extra semantics beyond what the field names reasonably imply.

If REASON appears, treat it only as the stated purpose category and interpret it only at that level, without inventing applicant-specific stories. Keep a neutral, compliance-friendly tone and avoid any approval or decline decision.
"""

sub_instruct_ld2 = """
Write one continuous English paragraph as a consumer-loan risk profile emphasizing affordability and credit quality. The paragraph must be analysis-dominant and grounded in the provided values: connect the stated income, loan terms, and ratios to payment burden pressure and risk sensitivity using cautious language, include both mitigating and risk signals, and include explicit contrasts, without a final verdict.

If present, explicitly connect person_income with loan_amnt, loan_int_rate, and loan_percent_income or similarly named affordability ratios by describing potential payment burden and rate-driven cost sensitivity, referencing the specific values you are interpreting.

If loan_grade or loan_intent appears, mention it as underwriting segmentation or purpose context only, not as a deterministic label. If cb_person_default_on_file and cb_person_cred_hist_length appear, describe them as adverse record and credit history depth context consistent with their names, without over-interpreting beyond what the fields suggest.
"""

sub_instruct_ld3 = """
Write one continuous English paragraph describing risk in a first-instalment performance context. The paragraph must be analysis-dominant and grounded in the provided values: interpret what the loan structure, leverage, coverage, and verification indicators may suggest about early payment stress using cautious language, include both strengths and concerns, and use explicit contrasts, without a final verdict.

If loan structure and collateral coverage fields appear (for example, ltv-like, asset_cost-like, disbursed_amount-like names), discuss leverage and coverage and how the stated ratios and amounts could relate to first-payment stress, explicitly referencing the values shown.

If identity or document flags, verification indicators, or similar fields appear, interpret them as documentation and verification strength or weakness cautiously, based only on what the field names and values suggest.

If bureau score, account count, or delinquency-history style fields appear, summarize credit depth and recent stress as suggested by the field names, grounded in the stated values.
"""

sub_instruct_cf1 = """
Write one continuous English paragraph at the transaction level for fraud-risk screening, not creditworthiness. The paragraph must be analysis-dominant and grounded in the provided values: interpret what the timing, amount, and anonymized signals may suggest about normal versus unusual activity using cautious language, include both benign context and risk signals, and use explicit contrasts, without a final verdict.

If present, mention Time and Amount as transaction timing and transaction size context, and interpret their magnitude only relative to the other provided fields, without introducing external baselines.

For V1 to V28 or similarly anonymized components, do not invent real-world meanings. Treat them as anonymized signal components and describe whether the observed combination appears extreme or unusual versus internally consistent, explicitly referring to the provided values when making that judgment.
"""

sub_instruct_cf2 = """
Write one continuous English paragraph that adapts to whatever schema appears in the input, staying transaction-centric. The paragraph must be analysis-dominant and grounded in the provided values: identify likely identifiers, timing and amount fields, merchant, channel, or location fields, and any outcome-like field purely from field names, then interpret only what the names and values support. Include both potential benign context and risk signals, with explicit contrasts, and no final verdict.

Do not assume the outcome field name. If an outcome-like field is present, mention it neutrally as a recorded field, and keep the paragraph focused on the transaction context and signals implied by the other fields.

Avoid credit-default language unless the schema clearly indicates default rather than fraud.
"""

sub_instruct_cc1 = """
Write one continuous English paragraph framed as a retention-risk profile over a six-month horizon. The paragraph must be analysis-dominant and grounded in the provided values: interpret what engagement, tenure, balance, product holding, and service interaction values may suggest about retention strength versus churn pressure using cautious language, include both strengths and concerns, and use explicit contrasts, without a final verdict.

If present, discuss engagement and usage intensity, relationship depth, tenure and vintage, product holding, and service interactions by describing signs of strengthening versus weakening engagement, price sensitivity, or dissatisfaction, grounded in the stated values and field names.

Avoid deterministic claims such as will churn or will not churn.
"""

sub_instruct_cc2 = """
Write one continuous English paragraph as a retail-banking relationship summary. The paragraph must be analysis-dominant and grounded in the provided values: interpret what relationship depth, activity, tenure, and salary context may suggest about stickiness versus attrition pressure using cautious language, include both strengths and concerns, and use explicit contrasts, without a final verdict.

If present, discuss Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, and EstimatedSalary or similarly named fields as relationship depth, activity, and affordability context, referencing the specific values shown.

If RowNumber, CustomerId, or Surname appear, you may mention them briefly as record references and do not treat them as behavioral drivers.

If Exited appears, mention it neutrally as a recorded field and keep the paragraph focused on describing the relationship signals in the other fields.
"""

sub_instruct_cc3 = """
Write one continuous English paragraph as a subscription or service retention profile. The paragraph must be analysis-dominant and grounded in the provided values: interpret what tenure and billing, service-bundle, and contract signals may suggest about retention versus switching pressure using cautious language, include both strengths and concerns, and use explicit contrasts, without a final verdict.

If present, discuss tenure-related fields, MonthlyCharges, and TotalCharges as tenure stability and billing pressure context, grounded in the stated values.

If service bundle fields such as phone, internet, and add-ons or contract and payment method fields appear, describe potential switching risk and price sensitivity cautiously, based only on what the field names and values suggest.

If Churn appears, mention it neutrally as a recorded field, not as a model decision.
"""

system_instruct="""
You are a senior financial risk analyst. Convert the applicant's structured risk-control features into ONE coherent English narrative paragraph for model training. 
"""
base_instruct= """


Hard requirements:

Output plain text only (no JSON, no markdown, no bullet points).

Write exactly ONE continuous English paragraph (no headings, no line breaks, no list formatting).

The paragraph must be natural, professional, and realistic in a financial risk-control context, and it must read like a credit analyst's narrative.

Coverage requirements:

You may rewrite feature names into natural language (you do NOT need to keep feature keys verbatim), but you must incorporate EVERY provided feature value and ensure every provided feature is referenced at least once.

Do not invent any applicant facts that are not supported by the provided features (no new fields, no fabricated history, no unstated behaviors).

If a feature is missing, null, or not provided, do NOT mention it at all.

Core requirement (analysis-focused; this is the priority):

This is NOT a pure feature restatement. The paragraph must be dominated by analysis and interpretation grounded in the provided values.

For each feature (or coherent feature group), explicitly explain what the specific value(s) may imply about delinquency/default risk using cautious analyst language (e.g., "may indicate," "is often associated with," "can signal," "could be consistent with," "may reduce/increase uncertainty," "may be a mitigating factor," "may introduce risk").

Balanced view is required:

Include both potential strengths/mitigating signals and potential risks/uncertainties suggested by the values, and explicitly connect them to the same applicant (e.g., "this may be supportive…, however it may also…").

Mandatory contrasts and interactions:

Include at least THREE explicit contrast/interactions in-sentence using connectors such as "while…," "however…," "whereas…," "on the other hand…," linking:
- a supportive value vs a concerning/uncertain value, and/or
- one feature's implication vs another feature's implication,
and make these contrasts directly grounded in the provided values (not generic).

Value-anchoring rule (prevents empty commentary):

Every analytical clause must cite at least one concrete provided value (a number or category) in the same sentence, so the analysis is visibly "based on data" rather than generic.

Context handling:

If the features describe a product that differs from the training target (e.g., car loan vs credit-card delinquency), explicitly acknowledge the mismatch in one short clause and interpret only what the provided values can support (do not introduce revolving-credit behaviors unless such fields exist).

Prohibited outputs:

Do not output any approval/decline decision, overall risk label, probability, or score, and do not conclude with a final verdict (no "overall risk is high/low," no "therefore approve/decline").

Output constraint (to avoid useless meta text, but keep analysis):

Do not include task restatements or template boilerplate (e.g., "We are given…", "The input features are…", "Constraints: …"). Use the entire paragraph for the applicant narrative and the background analysis itself.


"""
instruct_dic={"cd1":sub_instruct_cd1, "cd2":sub_instruct_cd2, "ld1":sub_instruct_ld1, "ld2":sub_instruct_ld2, "ld3":sub_instruct_ld3, "cf1":sub_instruct_cf1, "cf2":sub_instruct_cf2, "cc1":sub_instruct_cc1, "cc2":sub_instruct_cc2, "cc3":sub_instruct_cc3}
end_instruct="""
                    Do not output any approval, rejection, score, probability, or risk label.

                    Input:
                    You will receive financial features as input.

                    Now generate one narrative paragraph using the following input.
                    """
def get_instruction(ds_name):
    return system_instruct+ instruct_dic[ds_name]+ base_instruct + end_instruct 
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
                    instruction = get_instruction(ds_name)
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
