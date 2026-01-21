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
import asyncio
import time
from openai import AsyncOpenAI
from run_step1_get_instruction import system_instruct

# All available datasets and splits
ALL_DATASETS = ["cd1", "cd2", "ld1", "ld2", "ld3", "cf1", "cf2", "cc1", "cc2", "cc3"]
ALL_SPLITS = ["train", "validation", "test"]

async def process_single_request(client, instruction: str, line_idx: int):
    """Process a single OpenAI API request asynchronously"""
    start_time = time.time()
    try:
        response = await client.chat.completions.create(
            model="Qwen/Qwen3-4B-Thinking-2507",
            messages=[
                {"role": "system", "content": system_instruct},
                {"role": "user", "content": f"{instruction}"},
            ],
        )
        elapsed_time = time.time() - start_time
        res_content = response.choices[0].message.content
        return line_idx, json.dumps(res_content.strip()), elapsed_time
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f">>> Error processing line {line_idx}: {e} (took {elapsed_time:.2f}s)")
        return line_idx, None, elapsed_time

async def run_openai(client, ds_name: str, ds_split: str, start_idx: int = 0, end_idx: int = -1, batch_size: int = 10) -> int:
    # print_cnt = int(1e3)
    print_cnt = 100
    # for ds_name in ds_name_list:
    profile_dir = os.path.join(profile_root_dir, ds_name)
    os.makedirs(profile_dir, exist_ok=True)

    logger.info(f"\n\n>>> ds_name: {ds_name}; ds_split: {ds_split}")
    instruction_path = os.path.join(profile_dir, f"instruction_for_profile_X_{ds_split}.jsonl")
    if end_idx > 0:
        profile_path = os.path.join(profile_dir, f"profile_X_{ds_split}_{end_idx}.jsonl")
    else:
        profile_path = os.path.join(profile_dir, f"profile_X_{ds_split}_all.jsonl")
    logger.info(f">>> profile_path: {profile_path}")

    logger.info(f"\n\n>>> >>> start_idx: {start_idx}; batch_size: {batch_size}")
    
    # First, collect all instructions that need to be processed
    instructions_to_process = []
    read_cnt = 0
    
    with open(instruction_path, mode="r", encoding="utf-8") as fp_in:
        for line_idx, line in enumerate(fp_in):
            read_cnt += 1
            if line_idx < start_idx:
                continue
            if line_idx >= end_idx > 0:
                logger.info(f">>> >>> [{ds_name} - {ds_split}] line_idx >= end_idx > 0; "
                            f"read_cnt = {read_cnt}")
                break
            
            instruction = str(json.loads(line.strip()))
            instructions_to_process.append((line_idx, instruction))
    
    logger.info(f">>> >>> Total instructions to process: {len(instructions_to_process)}")
    
    # Process in batches and write incrementally to avoid data loss
    write_cnt = 0
    all_latencies = []  # Track all request latencies for statistics
    total_start_time = time.time()
    
    # Open file in append mode to write results incrementally
    with open(profile_path, mode="a+", encoding="utf-8") as fp_out:
        for batch_start in range(0, len(instructions_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(instructions_to_process))
            batch = instructions_to_process[batch_start:batch_end]
            
            batch_start_time = time.time()
            logger.info(f">>> >>> Processing batch {batch_start // batch_size + 1} "
                       f"(lines {batch[0][0]}-{batch[-1][0]})")
            
            # Create tasks for this batch
            tasks = [
                process_single_request(client, instruction, line_idx)
                for line_idx, instruction in batch
            ]
            
            # Execute batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            batch_elapsed_time = time.time() - batch_start_time
            
            # Collect successful results from this batch and track latencies
            batch_results_dict = {}
            batch_latencies = []
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f">>> Exception in batch: {result}")
                    continue
                line_idx, res_json, latency = result
                batch_latencies.append(latency)
                all_latencies.append(latency)
                if res_json is not None:
                    batch_results_dict[line_idx] = res_json
            
            # Calculate batch statistics
            if batch_latencies:
                avg_latency = sum(batch_latencies) / len(batch_latencies)
                min_latency = min(batch_latencies)
                max_latency = max(batch_latencies)
                successful_count = len(batch_results_dict)
                
                logger.info(f">>> >>> Batch {batch_start // batch_size + 1} completed in {batch_elapsed_time:.2f}s | "
                           f"Requests: {successful_count}/{len(batch)} | "
                           f"Avg latency: {avg_latency:.2f}s | "
                           f"Min: {min_latency:.2f}s | Max: {max_latency:.2f}s | "
                           f"Throughput: {successful_count/batch_elapsed_time:.2f} req/s")
            
            # Write batch results immediately in order (sorted by line_idx)
            batch_results_sorted = sorted(batch_results_dict.items())
            for line_idx, res_json in batch_results_sorted:
                fp_out.write(res_json + "\n")
                write_cnt += 1
            
            # Flush to ensure data is written to disk
            fp_out.flush()
            
            if (batch_start + batch_size) % print_cnt == 0 or batch_end == len(instructions_to_process):
                # Calculate overall statistics
                total_elapsed = time.time() - total_start_time
                if all_latencies:
                    overall_avg = sum(all_latencies) / len(all_latencies)
                    overall_min = min(all_latencies)
                    overall_max = max(all_latencies)
                    overall_throughput = write_cnt / total_elapsed if total_elapsed > 0 else 0
                    
                    logger.info(f">>> >>> [{ds_name} - {ds_split}] Progress Report:")
                    logger.info(f">>> >>>   Processed: {batch_end}/{len(instructions_to_process)}; "
                               f"Written: {write_cnt}")
                    logger.info(f">>> >>>   Overall Stats - Avg latency: {overall_avg:.2f}s | "
                               f"Min: {overall_min:.2f}s | Max: {overall_max:.2f}s | "
                               f"Throughput: {overall_throughput:.2f} req/s | "
                               f"Total time: {total_elapsed:.2f}s")
    
    total_elapsed = time.time() - total_start_time

    # Calculate final statistics
    if all_latencies:
        overall_avg = sum(all_latencies) / len(all_latencies)
        overall_min = min(all_latencies)
        overall_max = max(all_latencies)
        overall_throughput = write_cnt / total_elapsed if total_elapsed > 0 else 0
        
        logger.info(f"\n>>> DONE: [{ds_name} - {ds_split}]")
        logger.info(f">>> Final Statistics:")
        logger.info(f">>>   Total processed: {read_cnt}; Successfully written: {write_cnt}")
        logger.info(f">>>   Total time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} minutes)")
        logger.info(f">>>   Average latency per request: {overall_avg:.2f}s")
        logger.info(f">>>   Min latency: {overall_min:.2f}s | Max latency: {overall_max:.2f}s")
        logger.info(f">>>   Overall throughput: {overall_throughput:.2f} requests/second")
        logger.info(f">>>   Total requests tracked: {len(all_latencies)}\n\n")
    else:
        logger.info(f"\n>>> DONE: [{ds_name} - {ds_split}] read_cnt = {read_cnt}; write_cnt = {write_cnt}\n\n")
    
    return 0


if __name__ == "__main__":
    """
    python3 run_step2_gpt_profile.py --ds_name cd1 --ds_split train --start_idx 0 --end_idx -1 --batch_size 10
    python3 run_step2_gpt_profile.py --ds_name cd1 --ds_split validation --start_idx 0 --end_idx -1 --batch_size 10
    python3 run_step2_gpt_profile.py --ds_name cd1 --ds_split test --start_idx 0 --end_idx -1 --batch_size 10
    python3 run_step2_gpt_profile.py --ds_name all --ds_split train --start_idx 0 --end_idx -1 --batch_size 10
    python3 run_step2_gpt_profile.py --ds_name cd1 --ds_split all --start_idx 0 --end_idx -1 --batch_size 10
    python3 run_step2_gpt_profile.py --ds_name all --ds_split all --start_idx 0 --end_idx -1 --batch_size 10
    """

    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s - %(name)s] -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Step2 Get_Profile Args")
    parser.add_argument("--ds_name", type=str, default="cf1", help="Specify which dataset to use (or 'all' for all datasets)")
    parser.add_argument("--ds_split", type=str, default="train", help="train OR validation OR test (or 'all' for all splits)")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for continue generating")
    parser.add_argument("--end_idx", type=int, default=-1, help="Ending index for continue generating")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for concurrent requests")
    args = parser.parse_args()

    logger.info(args)

    ds_name = str(args.ds_name)
    ds_split = str(args.ds_split)
    start_idx = int(args.start_idx)
    end_idx = int(args.end_idx)
    batch_size = int(args.batch_size)
    
    # Determine which datasets and splits to run
    if ds_name == "all":
        datasets_to_run = ALL_DATASETS
    else:
        datasets_to_run = [ds_name]
    
    if ds_split == "all":
        splits_to_run = ALL_SPLITS
    else:
        splits_to_run = [ds_split]

    cache_dir = "~/.cache/huggingface/"
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    cache_model = os.path.join(cache_dir, "models")
    cache_ds = os.path.join(cache_dir, "datasets")

    # OpenAI settings - use AsyncOpenAI for batch processing
    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    # Use 127.0.0.1 instead of localhost to avoid IPv6 resolution issues
    openai_api_base = "http://127.0.0.1:8000/v1"

    client = AsyncOpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    profile_root_dir = os.path.join("./data/profile")
    os.makedirs(profile_root_dir, exist_ok=True)

    # Run for all combinations
    total_combinations = len(datasets_to_run) * len(splits_to_run)
    current = 0
    
    logger.info(f"\n>>> Running {total_combinations} combinations: {len(datasets_to_run)} datasets × {len(splits_to_run)} splits")
    logger.info(f">>> Datasets: {datasets_to_run}")
    logger.info(f">>> Splits: {splits_to_run}\n")
    
    for cur_ds_name in datasets_to_run:
        for cur_ds_split in splits_to_run:
            current += 1
            logger.info(f"\n>>> Progress: {current}/{total_combinations}")
            try:
                # Run async function
                asyncio.run(run_openai(client, ds_name=cur_ds_name, ds_split=cur_ds_split, 
                                       start_idx=start_idx, end_idx=end_idx, batch_size=batch_size))
            except Exception as e:
                logger.error(f">>> ERROR running {cur_ds_name}---{cur_ds_split}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                continue

    sys.exit(0)
