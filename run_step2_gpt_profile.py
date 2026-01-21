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
from openai import AsyncOpenAI
from run_step1_get_instruction import system_instruct

async def process_single_request(client, instruction: str, line_idx: int):
    """Process a single OpenAI API request asynchronously"""
    try:
        response = await client.chat.completions.create(
            model="Qwen/Qwen3-4B-Thinking-2507",
            messages=[
                {"role": "system", "content": system_instruct},
                {"role": "user", "content": f"{instruction}"},
            ],
        )
        res_content = response.choices[0].message.content
        return line_idx, json.dumps(res_content.strip())
    except Exception as e:
        logger.error(f">>> Error processing line {line_idx}: {e}")
        return line_idx, None

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
    
    # Open file in append mode to write results incrementally
    with open(profile_path, mode="a+", encoding="utf-8") as fp_out:
        for batch_start in range(0, len(instructions_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(instructions_to_process))
            batch = instructions_to_process[batch_start:batch_end]
            
            logger.info(f">>> >>> Processing batch {batch_start // batch_size + 1} "
                       f"(lines {batch[0][0]}-{batch[-1][0]})")
            
            # Create tasks for this batch
            tasks = [
                process_single_request(client, instruction, line_idx)
                for line_idx, instruction in batch
            ]
            
            # Execute batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect successful results from this batch
            batch_results_dict = {}
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f">>> Exception in batch: {result}")
                    continue
                line_idx, res_json = result
                if res_json is not None:
                    batch_results_dict[line_idx] = res_json
            
            # Write batch results immediately in order (sorted by line_idx)
            batch_results_sorted = sorted(batch_results_dict.items())
            for line_idx, res_json in batch_results_sorted:
                fp_out.write(res_json + "\n")
                write_cnt += 1
            
            # Flush to ensure data is written to disk
            fp_out.flush()
            
            if (batch_start + batch_size) % print_cnt == 0 or batch_end == len(instructions_to_process):
                logger.info(f">>> >>> [{ds_name} - {ds_split}] "
                           f"processed: {batch_end}/{len(instructions_to_process)}; "
                           f"written: {write_cnt}")

    logger.info(f"\n>>> DONE: [{ds_name} - {ds_split}] read_cnt = {read_cnt}; write_cnt = {write_cnt}\n\n")
    return 0


if __name__ == "__main__":
    """
    python3 run_step2_gpt_profile.py --ds_name cd1 --ds_split train --start_idx 0 --end_idx -1 --batch_size 10
    python3 run_step2_gpt_profile.py --ds_name cd1 --ds_split validation --start_idx 0 --end_idx -1 --batch_size 10
    python3 run_step2_gpt_profile.py --ds_name cd1 --ds_split test --start_idx 0 --end_idx -1 --batch_size 10
    """

    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s - %(name)s] -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Step2 Get_Profile Args")
    parser.add_argument("--ds_name", type=str, default="cf1", help="Specify which dataset to use")
    parser.add_argument("--ds_split", type=str, default="train", help="train OR validation OR test")
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

    # Run async function
    asyncio.run(run_openai(client, ds_name=ds_name, ds_split=ds_split, 
                           start_idx=start_idx, end_idx=end_idx, batch_size=batch_size))

    sys.exit(0)
