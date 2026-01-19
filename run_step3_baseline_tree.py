#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import os
import sys
import logging
import argparse
import gc
import json
from collections import defaultdict

import numpy as np

# Optional pandas import for CSV output
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score
)

from huggingface_hub import snapshot_download
from datasets import load_from_disk

from model import *
from utils.seed import set_seed

# All available datasets and models
ALL_DATASETS = ["cd1", "cd2", "ld1", "ld2", "ld3", "cf1", "cf2", "cc1", "cc2", "cc3"]
ALL_MODELS = ["RandomForestClassifier", "XGBClassifier", "CatBoostClassifier", "LGBMClassifier"]


def calc_classification_metrics(y_true, y_pred, pos_label=1):
    """
    Calculate comprehensive classification metrics from confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        pos_label: Positive label value. Defaults to 1.
    
    Returns:
        Dictionary containing:
        - Basic counts: TP, TN, FP, FN
        - Accuracy metrics: acc, balanced_acc
        - Precision/Recall: precision, recall, f1
        - Specificity: specificity
        - Error rates: FPR, FNR
        - Predictive values: NPV
        - MCC: Matthews Correlation Coefficient
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            "acc": 0.0,
            "balanced_acc": 0.0,
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "specificity": 0.0,
            "FPR": 0.0,
            "FNR": 0.0,
            "NPV": 0.0,
            "MCC": 0.0,
            "TP": 0,
            "TN": 0,
            "FP": 0,
            "FN": 0,
        }
    
    tp = int(np.sum((y_pred == pos_label) & (y_true == pos_label)))
    fp = int(np.sum((y_pred == pos_label) & (y_true != pos_label)))
    fn = int(np.sum((y_pred != pos_label) & (y_true == pos_label)))
    tn = int(np.sum((y_pred != pos_label) & (y_true != pos_label)))
    
    total = tp + fp + fn + tn
    acc = float(tp + tn) / total if total > 0 else 0.0
    
    # Precision (PPV - Positive Predictive Value)
    denom_p = tp + fp
    precision = float(tp / denom_p) if denom_p > 0 else 0.0
    
    # Recall (Sensitivity, TPR - True Positive Rate)
    denom_r = tp + fn
    recall = float(tp / denom_r) if denom_r > 0 else 0.0
    
    # Specificity (TNR - True Negative Rate)
    denom_spec = tn + fp
    specificity = float(tn / denom_spec) if denom_spec > 0 else 0.0
    
    # F1 score
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    # Balanced Accuracy
    balanced_acc = (recall + specificity) / 2.0
    
    # False Positive Rate (FPR)
    FPR = float(fp / denom_spec) if denom_spec > 0 else 0.0
    
    # False Negative Rate (FNR)
    FNR = float(fn / denom_r) if denom_r > 0 else 0.0
    
    # Negative Predictive Value (NPV)
    denom_npv = tn + fn
    NPV = float(tn / denom_npv) if denom_npv > 0 else 0.0
    
    # Matthews Correlation Coefficient (MCC)
    # MCC = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc_denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if mcc_denom > 0:
        mcc = float((tp * tn - fp * fn) / np.sqrt(mcc_denom))
    else:
        mcc = 0.0
    
    return {
        "acc": acc,
        "balanced_acc": balanced_acc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "specificity": specificity,
        "FPR": FPR,
        "FNR": FNR,
        "NPV": NPV,
        "MCC": mcc,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def run_baselines(cur_ds_name, cur_model_name):
    logger.info(f"\n\n\n *** *** cur_ds_name: {cur_ds_name}; cur_model_name: {cur_model_name}")
    
    # Use snapshot_download + load_from_disk (consistent with finbench_process.py)
    snapshot_path = snapshot_download(
        repo_id="dhugs/FinBench",
        repo_type="dataset",
    )
    task_path = os.path.join(snapshot_path, cur_ds_name)
    data = load_from_disk(task_path)  # DatasetDict
    
    train_set = data["train"] if "train" in data else []
    val_set = data["validation"] if "validation" in data else []
    test_set = data["test"] if "test" in data else []

    x_key = "X_ml"
    # x_key = "X_ml_unscale"
    train_X_ml, train_y = np.asarray(train_set[x_key], dtype=np.float32), np.asarray(train_set["y"], dtype=np.int64)
    val_X_ml, val_y = np.asarray(val_set[x_key], dtype=np.float32), np.asarray(val_set["y"], dtype=np.int64)
    test_X_ml, test_y = np.asarray(test_set[x_key], dtype=np.float32), np.asarray(test_set["y"], dtype=np.int64)

    # get pos ratio of the training set for loss computing
    total_y = float(len(train_y))
    pos_y = float(sum(train_y))
    assert total_y >= pos_y > 0.0
    neg_to_pos = float((total_y - pos_y) / pos_y)
    pos_ratio = float(pos_y / total_y)
    logger.info(f">>> pos_ratio = {pos_ratio}; neg_to_pos = {neg_to_pos}")
    class_weight = {0: 1.0, 1: neg_to_pos}
    args.class_weight = class_weight

    model = MODEL_DICT[cur_model_name](args)
    if grid_search and hasattr(model, "param_grid") and isinstance(model.param_grid, dict):
        clf = GridSearchCV(model, model.param_grid, cv=5, scoring="f1")
        clf.fit(train_X_ml, train_y)
        logger.info(f"clf.best_params_: {clf.best_params_}")
        best_model = clf.best_estimator_
    else:
        model.fit(train_X_ml, train_y)
        best_model = model

    y_pred_train = best_model.predict(train_X_ml)
    # Calculate comprehensive confusion matrix metrics
    metrics_train = calc_classification_metrics(train_y, y_pred_train)
    # Also calculate AUC and Average Precision (requires probability predictions)
    try:
        y_pred_proba_train = best_model.predict_proba(train_X_ml)[:, 1] if hasattr(best_model, "predict_proba") else None
        auc_train = roc_auc_score(train_y, y_pred_proba_train) if y_pred_proba_train is not None else 0.0
        avg_p_train = average_precision_score(train_y, y_pred_proba_train) if y_pred_proba_train is not None else 0.0
    except:
        auc_train = 0.0
        avg_p_train = 0.0

    y_pred_val = best_model.predict(val_X_ml)
    metrics_val = calc_classification_metrics(val_y, y_pred_val)
    try:
        y_pred_proba_val = best_model.predict_proba(val_X_ml)[:, 1] if hasattr(best_model, "predict_proba") else None
        auc_val = roc_auc_score(val_y, y_pred_proba_val) if y_pred_proba_val is not None else 0.0
        avg_p_val = average_precision_score(val_y, y_pred_proba_val) if y_pred_proba_val is not None else 0.0
    except:
        auc_val = 0.0
        avg_p_val = 0.0

    y_pred_test = best_model.predict(test_X_ml)
    metrics_test = calc_classification_metrics(test_y, y_pred_test)
    try:
        y_pred_proba_test = best_model.predict_proba(test_X_ml)[:, 1] if hasattr(best_model, "predict_proba") else None
        auc_test = roc_auc_score(test_y, y_pred_proba_test) if y_pred_proba_test is not None else 0.0
        avg_p_test = average_precision_score(test_y, y_pred_proba_test) if y_pred_proba_test is not None else 0.0
    except:
        auc_test = 0.0
        avg_p_test = 0.0

    logger.info(f">>> Dataset = {cur_ds_name}; Model = {cur_model_name}")
    logger.info(f">>> [{cur_ds_name}---{cur_model_name}] Evaluation (Training set):")
    logger.info(f"    Accuracy = %.4f; Balanced Acc = %.4f; F1 = %.4f; Precision = %.4f; Recall = %.4f" % (
        metrics_train["acc"], metrics_train["balanced_acc"], metrics_train["f1"],
        metrics_train["precision"], metrics_train["recall"]))
    logger.info(f"    Specificity = %.4f; FPR = %.4f; FNR = %.4f; NPV = %.4f; MCC = %.4f" % (
        metrics_train["specificity"], metrics_train["FPR"], metrics_train["FNR"],
        metrics_train["NPV"], metrics_train["MCC"]))
    logger.info(f"    TP = %d; TN = %d; FP = %d; FN = %d; AUC = %.4f; Avg Precision = %.4f" % (
        metrics_train["TP"], metrics_train["TN"], metrics_train["FP"], metrics_train["FN"],
        auc_train, avg_p_train))
    
    logger.info(f">>> [{cur_ds_name}---{cur_model_name}] Evaluation (Validation set):")
    logger.info(f"    Accuracy = %.4f; Balanced Acc = %.4f; F1 = %.4f; Precision = %.4f; Recall = %.4f" % (
        metrics_val["acc"], metrics_val["balanced_acc"], metrics_val["f1"],
        metrics_val["precision"], metrics_val["recall"]))
    logger.info(f"    Specificity = %.4f; FPR = %.4f; FNR = %.4f; NPV = %.4f; MCC = %.4f" % (
        metrics_val["specificity"], metrics_val["FPR"], metrics_val["FNR"],
        metrics_val["NPV"], metrics_val["MCC"]))
    logger.info(f"    TP = %d; TN = %d; FP = %d; FN = %d; AUC = %.4f; Avg Precision = %.4f" % (
        metrics_val["TP"], metrics_val["TN"], metrics_val["FP"], metrics_val["FN"],
        auc_val, avg_p_val))
    
    logger.info(f">>> [{cur_ds_name}---{cur_model_name}] Evaluation (Test set):")
    logger.info(f"    Accuracy = %.4f; Balanced Acc = %.4f; F1 = %.4f; Precision = %.4f; Recall = %.4f" % (
        metrics_test["acc"], metrics_test["balanced_acc"], metrics_test["f1"],
        metrics_test["precision"], metrics_test["recall"]))
    logger.info(f"    Specificity = %.4f; FPR = %.4f; FNR = %.4f; NPV = %.4f; MCC = %.4f" % (
        metrics_test["specificity"], metrics_test["FPR"], metrics_test["FNR"],
        metrics_test["NPV"], metrics_test["MCC"]))
    logger.info(f"    TP = %d; TN = %d; FP = %d; FN = %d; AUC = %.4f; Avg Precision = %.4f" % (
        metrics_test["TP"], metrics_test["TN"], metrics_test["FP"], metrics_test["FN"],
        auc_test, avg_p_test))

    # Store comprehensive metrics
    eval_results_train[f"{cur_ds_name}---{cur_model_name}---train"] = {
        "acc": metrics_train["acc"],
        "balanced_acc": metrics_train["balanced_acc"],
        "f1": metrics_train["f1"],
        "precision": metrics_train["precision"],
        "recall": metrics_train["recall"],
        "specificity": metrics_train["specificity"],
        "FPR": metrics_train["FPR"],
        "FNR": metrics_train["FNR"],
        "NPV": metrics_train["NPV"],
        "MCC": metrics_train["MCC"],
        "TP": metrics_train["TP"],
        "TN": metrics_train["TN"],
        "FP": metrics_train["FP"],
        "FN": metrics_train["FN"],
        "auc": auc_train,
        "avg_precision": avg_p_train,
    }
    eval_results_val[f"{cur_ds_name}---{cur_model_name}---val"] = {
        "acc": metrics_val["acc"],
        "balanced_acc": metrics_val["balanced_acc"],
        "f1": metrics_val["f1"],
        "precision": metrics_val["precision"],
        "recall": metrics_val["recall"],
        "specificity": metrics_val["specificity"],
        "FPR": metrics_val["FPR"],
        "FNR": metrics_val["FNR"],
        "NPV": metrics_val["NPV"],
        "MCC": metrics_val["MCC"],
        "TP": metrics_val["TP"],
        "TN": metrics_val["TN"],
        "FP": metrics_val["FP"],
        "FN": metrics_val["FN"],
        "auc": auc_val,
        "avg_precision": avg_p_val,
    }
    eval_results_test[f"{cur_ds_name}---{cur_model_name}---test"] = {
        "acc": metrics_test["acc"],
        "balanced_acc": metrics_test["balanced_acc"],
        "f1": metrics_test["f1"],
        "precision": metrics_test["precision"],
        "recall": metrics_test["recall"],
        "specificity": metrics_test["specificity"],
        "FPR": metrics_test["FPR"],
        "FNR": metrics_test["FNR"],
        "NPV": metrics_test["NPV"],
        "MCC": metrics_test["MCC"],
        "TP": metrics_test["TP"],
        "TN": metrics_test["TN"],
        "FP": metrics_test["FP"],
        "FN": metrics_test["FN"],
        "auc": auc_test,
        "avg_precision": avg_p_test,
    }

    del data
    del model
    gc.collect()


def save_results_to_csv(eval_results_train, eval_results_val, eval_results_test, output_file="results.csv"):
    """
    Save all evaluation results to CSV file.
    
    Args:
        eval_results_train: Dictionary of training set results
        eval_results_val: Dictionary of validation set results
        eval_results_test: Dictionary of test set results
        output_file: Output CSV file path
    """
    if not HAS_PANDAS:
        logger.warning(">>> pandas not available, cannot save CSV file")
        return
    
    all_rows = []
    
    # Process each split
    for split_name, results_dict in [("train", eval_results_train), 
                                     ("val", eval_results_val), 
                                     ("test", eval_results_test)]:
        for key, metrics in results_dict.items():
            parts = key.split("---")
            if len(parts) >= 3:
                ds_name, model_name = parts[0], parts[1]
                row = {
                    "model_name": model_name,
                    "ds_name": ds_name,
                    "split": split_name,
                }
                row.update(metrics)
                all_rows.append(row)
    
    if all_rows:
        df = pd.DataFrame(all_rows)
        # Reorder columns: model_name, ds_name, split, then all metrics
        metric_cols = [col for col in df.columns if col not in ["model_name", "ds_name", "split"]]
        df = df[["model_name", "ds_name", "split"] + metric_cols]
        df.to_csv(output_file, index=False)
        logger.info(f">>> Results saved to CSV file: {output_file}")
        logger.info(f">>> Total rows: {len(df)}")
    else:
        logger.warning(">>> No results to save")


def print_all_results_summary(eval_results_train, eval_results_val, eval_results_test, output_format="table"):
    """
    Print a comprehensive summary of all evaluation results.
    
    Args:
        eval_results_train: Dictionary of training set results
        eval_results_val: Dictionary of validation set results
        eval_results_test: Dictionary of test set results
        output_format: "table" for formatted table, "json" for JSON, "csv" for CSV
    """
    logger.info("\n" + "="*100)
    logger.info("COMPREHENSIVE EVALUATION RESULTS SUMMARY")
    logger.info("="*100)
    
    # Collect all results by dataset and model
    results_by_ds_model = defaultdict(lambda: {
        "train": None,
        "val": None,
        "test": None
    })
    
    for key, metrics in eval_results_train.items():
        parts = key.split("---")
        if len(parts) >= 3:
            ds_name, model_name = parts[0], parts[1]
            results_by_ds_model[f"{ds_name}---{model_name}"]["train"] = metrics
    
    for key, metrics in eval_results_val.items():
        parts = key.split("---")
        if len(parts) >= 3:
            ds_name, model_name = parts[0], parts[1]
            results_by_ds_model[f"{ds_name}---{model_name}"]["val"] = metrics
    
    for key, metrics in eval_results_test.items():
        parts = key.split("---")
        if len(parts) >= 3:
            ds_name, model_name = parts[0], parts[1]
            results_by_ds_model[f"{ds_name}---{model_name}"]["test"] = metrics
    
    if output_format == "json":
        # Output as JSON
        logger.info("\n>>> JSON Format Results:")
        all_results = {
            "train": eval_results_train,
            "val": eval_results_val,
            "test": eval_results_test
        }
        print(json.dumps(all_results, indent=2))
    
    elif output_format == "csv":
        # Output as CSV tables
        logger.info("\n>>> CSV Format Results:")
        
        if not HAS_PANDAS:
            logger.warning(">>> pandas not available, falling back to table format")
            output_format = "table"
        else:
            # Create DataFrames for each split
            for split_name, results_dict in [("train", eval_results_train), 
                                             ("val", eval_results_val), 
                                             ("test", eval_results_test)]:
                rows = []
                for key, metrics in results_dict.items():
                    parts = key.split("---")
                    if len(parts) >= 3:
                        ds_name, model_name = parts[0], parts[1]
                        row = {"dataset": ds_name, "model": model_name}
                        row.update(metrics)
                        rows.append(row)
                
                if rows:
                    df = pd.DataFrame(rows)
                    logger.info(f"\n{split_name.upper()} Set Results:")
                    print(df.to_csv(index=False))
    
    if output_format == "table":
        # Output as formatted tables
        logger.info("\n>>> Formatted Table Results:\n")
        
        # Group by dataset
        datasets = sorted(set([key.split("---")[0] for key in results_by_ds_model.keys()]))
        models = sorted(set([key.split("---")[1] for key in results_by_ds_model.keys()]))
        
        for split_name in ["train", "val", "test"]:
            logger.info(f"\n{'='*100}")
            logger.info(f"{split_name.upper()} SET RESULTS")
            logger.info(f"{'='*100}\n")
            
            # Create table header
            header = f"{'Dataset':<10} {'Model':<25} {'Acc':<8} {'Bal_Acc':<8} {'F1':<8} {'Precision':<10} {'Recall':<8} {'MCC':<8} {'AUC':<8}"
            logger.info(header)
            logger.info("-" * len(header))
            
            for ds_name in datasets:
                for model_name in models:
                    key = f"{ds_name}---{model_name}"
                    if key in results_by_ds_model:
                        metrics = results_by_ds_model[key].get(split_name)
                        if metrics:
                            row = (f"{ds_name:<10} {model_name:<25} "
                                   f"{metrics['acc']:<8.4f} {metrics['balanced_acc']:<8.4f} "
                                   f"{metrics['f1']:<8.4f} {metrics['precision']:<10.4f} "
                                   f"{metrics['recall']:<8.4f} {metrics['MCC']:<8.4f} "
                                   f"{metrics.get('auc', 0.0):<8.4f}")
                            logger.info(row)
        
        # Detailed metrics for test set
        logger.info(f"\n{'='*100}")
        logger.info("DETAILED TEST SET METRICS")
        logger.info(f"{'='*100}\n")
        
        for ds_name in datasets:
            logger.info(f"\nDataset: {ds_name}")
            logger.info("-" * 80)
            for model_name in models:
                key = f"{ds_name}---{model_name}"
                if key in results_by_ds_model:
                    metrics = results_by_ds_model[key].get("test")
                    if metrics:
                        logger.info(f"\n  Model: {model_name}")
                        logger.info(f"    Accuracy: {metrics['acc']:.4f}, Balanced Acc: {metrics['balanced_acc']:.4f}")
                        logger.info(f"    F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
                        logger.info(f"    Specificity: {metrics['specificity']:.4f}, MCC: {metrics['MCC']:.4f}")
                        logger.info(f"    FPR: {metrics['FPR']:.4f}, FNR: {metrics['FNR']:.4f}, NPV: {metrics['NPV']:.4f}")
                        logger.info(f"    TP: {metrics['TP']}, TN: {metrics['TN']}, FP: {metrics['FP']}, FN: {metrics['FN']}")
                        logger.info(f"    AUC: {metrics.get('auc', 0.0):.4f}, Avg Precision: {metrics.get('avg_precision', 0.0):.4f}")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s - %(name)s] -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Baseline args")

    parser.add_argument("--seed", type=int, default=0, help="Seed of random modules")
    parser.add_argument("--ds_name", type=str, default="cd1", help="Specify which dataset to use.",
                        choices=["cd1", "cd2", "ld1", "ld2", "ld3", "cf1", "cf2", "cc1", "cc2", "cc3", "all"])
    parser.add_argument("--model_name", type=str, default="LogisticRegression", help="Specify which model to use.",
                        choices=["RandomForestClassifier", "XGBClassifier", "CatBoostClassifier", "LGBMClassifier", "all"])
    parser.add_argument("--bsz", type=int, default=128, help="TrainingArguments: per_device_train/eval_batch_size")
    parser.add_argument("--epoch", type=int, default=100, help="TrainingArguments: num_train_epochs")
    parser.add_argument("--objective", type=str, default="classification",
                        choices=["classification", "binary", "regression"],
                        help="The type of the current task")
    parser.add_argument("--grid_search", action="store_true", help="GridSearch")
    parser.add_argument("--output_format", type=str, default="table", 
                        choices=["table", "json", "csv"],
                        help="Output format for results summary: table, json, or csv")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Path to save CSV file with all results (model_name, ds_name, metrics)")

    args = parser.parse_args()
    logger.info(args)

    seed = int(args.seed)
    ds_name = str(args.ds_name)
    model_name = str(args.model_name)
    bsz = int(args.bsz)
    epoch = int(args.epoch)
    grid_search = bool(args.grid_search)
    output_format = str(args.output_format)
    output_csv = args.output_csv

    set_seed(seed)

    cache_dir = "~/.cache/huggingface/"
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    cache_model = os.path.join(cache_dir, "models")
    cache_ds = os.path.join(cache_dir, "datasets")

    eval_results_train = dict({})  # dict{ds_name: Tuple(acc, f1, auc, p, r, avg_p)}
    eval_results_val = dict({})
    eval_results_test = dict({})

    # Determine which datasets and models to run
    if ds_name == "all":
        datasets_to_run = ALL_DATASETS
    else:
        datasets_to_run = [ds_name]
    
    if model_name == "all":
        models_to_run = ALL_MODELS
    else:
        models_to_run = [model_name]
    
    # Run baselines for all combinations
    total_combinations = len(datasets_to_run) * len(models_to_run)
    current = 0
    
    logger.info(f"\n>>> Running {total_combinations} combinations: {len(datasets_to_run)} datasets × {len(models_to_run)} models")
    logger.info(f">>> Datasets: {datasets_to_run}")
    logger.info(f">>> Models: {models_to_run}\n")
    
    for cur_ds_name in datasets_to_run:
        for cur_model_name in models_to_run:
            current += 1
            logger.info(f"\n>>> Progress: {current}/{total_combinations}")
            try:
                run_baselines(cur_ds_name=cur_ds_name, cur_model_name=cur_model_name)
            except Exception as e:
                logger.error(f">>> ERROR running {cur_ds_name}---{cur_model_name}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                continue

    # Print comprehensive summary
    print_all_results_summary(eval_results_train, eval_results_val, eval_results_test, output_format=output_format)
    
    # Save to CSV if requested
    if output_csv:
        save_results_to_csv(eval_results_train, eval_results_val, eval_results_test, output_file=output_csv)
    elif output_format == "csv":
        # If output_format is csv but no file specified, save to default file
        save_results_to_csv(eval_results_train, eval_results_val, eval_results_test, output_file="results.csv")

    logger.info(f"\n\n>>> END: Total combinations completed: {len(eval_results_test)}")
    logger.info(f">>> END: eval_results_train keys: {list(eval_results_train.keys())}")
    logger.info(f">>> END: eval_results_val keys: {list(eval_results_val.keys())}")
    logger.info(f">>> END: eval_results_test keys: {list(eval_results_test.keys())}")

    sys.exit(0)
