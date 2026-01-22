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
import random
from openai import AsyncOpenAI
from run_step1_get_instruction import system_instruct
import httpx


# All available datasets and splits
ALL_DATASETS = ["cd1", "cd2", "ld1", "ld2", "ld3", "cf1", "cf2", "cc1", "cc2", "cc3"]
ALL_SPLITS = ["train", "validation", "test"]


class ClientLoadBalancer:
    """
    Client-side load balancer for distributing requests across multiple vLLM backends.
    Uses round-robin algorithm to ensure even distribution.
    """
    def __init__(self, backend_ports: list[int], openai_api_key: str = "EMPTY"):
        """
        Initialize load balancer with multiple backend clients.
        
        Args:
            backend_ports: List of backend port numbers (e.g., [8000, 8001, ..., 8007])
            openai_api_key: API key for OpenAI clients
        """
        self.backend_ports = backend_ports
        self.current_index = 0
        self.request_counts = {port: 0 for port in backend_ports}
        self.lock = asyncio.Lock()
        
        # Create an AsyncOpenAI client for each backend
        self.clients = {}
        timeout = httpx.Timeout(
            connect=10.0,
            read=600.0,
            write=30.0,
            pool=5.0,
        )
        
        limits = httpx.Limits(
            max_connections=16,  # Per backend
            max_keepalive_connections=16,
        )
        
        for port in backend_ports:
            base_url = f"http://127.0.0.1:{port}/v1"
            self.clients[port] = AsyncOpenAI(
                api_key=openai_api_key,
                base_url=base_url,
                http_client=httpx.AsyncClient(
                    limits=limits,
                    timeout=timeout,
                    http2=False,
                ),
            )
        
        # Log initialization (logger may not be available yet, use print as fallback)
        try:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f">>> Initialized client load balancer with {len(backend_ports)} backends")
            logger.info(f">>>   Backend ports: {backend_ports}")
        except:
            print(f">>> Initialized client load balancer with {len(backend_ports)} backends")
            print(f">>>   Backend ports: {backend_ports}")
    
    async def get_client(self, strategy: str = "round_robin") -> tuple[AsyncOpenAI, int]:
        """
        Get a client based on load balancing strategy.
        
        Args:
            strategy: "round_robin" or "least_requests"
        
        Returns:
            Tuple of (client, port)
        """
        async with self.lock:
            if strategy == "round_robin":
                # Round-robin: cycle through backends
                port = self.backend_ports[self.current_index]
                self.current_index = (self.current_index + 1) % len(self.backend_ports)
                self.request_counts[port] += 1
                return self.clients[port], port
            elif strategy == "least_requests":
                # Least requests: choose backend with fewest requests
                port = min(self.request_counts, key=self.request_counts.get)
                self.request_counts[port] += 1
                return self.clients[port], port
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
    
    async def release_client(self, port: int):
        """Release a client (decrement request count)"""
        async with self.lock:
            self.request_counts[port] = max(0, self.request_counts[port] - 1)
    
    def get_stats(self) -> dict:
        """Get load balancing statistics"""
        return {
            "request_counts": self.request_counts.copy(),
            "total_requests": sum(self.request_counts.values()),
        }

async def process_single_request(load_balancer: ClientLoadBalancer, instruction: str, line_idx: int, 
                                  max_retries: int = 3, retry_delay: float = 1.0, 
                                  strategy: str = "round_robin"):
    """
    Process a single OpenAI API request asynchronously with retry mechanism.
    Uses client-side load balancing to distribute requests across backends.
    
    Args:
        load_balancer: ClientLoadBalancer instance
        instruction: Instruction text to process
        line_idx: Line index for tracking
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (exponential backoff)
        strategy: Load balancing strategy ("round_robin" or "least_requests")
    """
    start_time = time.time()
    last_exception = None
    client = None
    port = None
    
    for attempt in range(max_retries):
        try:
            # Get a client from load balancer
            client, port = await load_balancer.get_client(strategy=strategy)
            
            response = await client.chat.completions.create(
                model="Qwen/Qwen3-4B-Thinking-2507",
                messages=[
                    {"role": "system", "content": system_instruct},
                    {"role": "user", "content": f"{instruction}"},
                ],
            )
            elapsed_time = time.time() - start_time
            res_content = response.choices[0].message.content
            
            # Release client after successful request
            if port is not None:
                await load_balancer.release_client(port)
            
            return line_idx, json.dumps(res_content.strip()), elapsed_time
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
            # Network-related errors: retry with exponential backoff
            last_exception = e
            if port is not None:
                await load_balancer.release_client(port)
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f">>> Line {line_idx} attempt {attempt + 1}/{max_retries} failed on port {port}: {e}. Retrying in {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)
            else:
                elapsed_time = time.time() - start_time
                logger.error(f">>> Error processing line {line_idx} after {max_retries} attempts: {e} (took {elapsed_time:.2f}s)")
                return line_idx, None, elapsed_time
        except httpx.HTTPStatusError as e:
            # HTTP status errors (5xx from backend): retry
            last_exception = e
            if port is not None:
                await load_balancer.release_client(port)
            if e.response.status_code >= 500 and attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(f">>> Line {line_idx} HTTP {e.response.status_code} on port {port}, attempt {attempt + 1}/{max_retries}. Retrying in {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)
            else:
                elapsed_time = time.time() - start_time
                logger.error(f">>> HTTP error processing line {line_idx} on port {port}: {e} (status: {e.response.status_code}, took {elapsed_time:.2f}s)")
                return line_idx, None, elapsed_time
        except Exception as e:
            # Other errors: log and return failure
            if port is not None:
                await load_balancer.release_client(port)
            elapsed_time = time.time() - start_time
            logger.error(f">>> Error processing line {line_idx} on port {port}: {e} (took {elapsed_time:.2f}s)")
            return line_idx, None, elapsed_time
    
    # Should not reach here, but handle it anyway
    elapsed_time = time.time() - start_time
    logger.error(f">>> Failed to process line {line_idx} after {max_retries} attempts: {last_exception} (took {elapsed_time:.2f}s)")
    return line_idx, None, elapsed_time

async def run_openai(load_balancer: ClientLoadBalancer, ds_name: str, ds_split: str, start_idx: int = 0, end_idx: int = -1, 
                     batch_size: int = 10, strategy: str = "round_robin") -> int:
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
            # Client-side load balancer will distribute requests across all backends
            # Using round-robin or least-requests strategy ensures even distribution
            tasks = [
                process_single_request(load_balancer, instruction, line_idx, strategy=strategy)
                for line_idx, instruction in batch
            ]
            
            # Execute batch concurrently
            # Load balancer ensures requests are distributed across all 8 vLLM backends
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
                    
                    # Get load balancer statistics
                    lb_stats = load_balancer.get_stats()
                    
                    logger.info(f">>> >>> [{ds_name} - {ds_split}] Progress Report:")
                    logger.info(f">>> >>>   Processed: {batch_end}/{len(instructions_to_process)}; "
                               f"Written: {write_cnt}")
                    logger.info(f">>> >>>   Overall Stats - Avg latency: {overall_avg:.2f}s | "
                               f"Min: {overall_min:.2f}s | Max: {overall_max:.2f}s | "
                               f"Throughput: {overall_throughput:.2f} req/s | "
                               f"Total time: {total_elapsed:.2f}s")
                    logger.info(f">>> >>>   Load Balancer Stats: {lb_stats['request_counts']}")
    
    total_elapsed = time.time() - total_start_time

    # Calculate final statistics
    if all_latencies:
        overall_avg = sum(all_latencies) / len(all_latencies)
        overall_min = min(all_latencies)
        overall_max = max(all_latencies)
        overall_throughput = write_cnt / total_elapsed if total_elapsed > 0 else 0
        
        # Get final load balancer statistics
        lb_stats = load_balancer.get_stats()
        
        logger.info(f"\n>>> DONE: [{ds_name} - {ds_split}]")
        logger.info(f">>> Final Statistics:")
        logger.info(f">>>   Total processed: {read_cnt}; Successfully written: {write_cnt}")
        logger.info(f">>>   Total time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} minutes)")
        logger.info(f">>>   Average latency per request: {overall_avg:.2f}s")
        logger.info(f">>>   Min latency: {overall_min:.2f}s | Max latency: {overall_max:.2f}s")
        logger.info(f">>>   Overall throughput: {overall_throughput:.2f} requests/second")
        logger.info(f">>>   Total requests tracked: {len(all_latencies)}")
        logger.info(f">>>   Load Balancer Distribution:")
        for port, count in sorted(lb_stats['request_counts'].items()):
            percentage = (count / lb_stats['total_requests'] * 100) if lb_stats['total_requests'] > 0 else 0
            logger.info(f">>>     Port {port}: {count} requests ({percentage:.1f}%)")
        logger.info(f"\n")
    else:
        logger.info(f"\n>>> DONE: [{ds_name} - {ds_split}] read_cnt = {read_cnt}; write_cnt = {write_cnt}\n\n")
    
    return 0


if __name__ == "__main__":
    """
    Run profile generation with client-side load balancing across multiple vLLM backends.
    
    The script uses client-side load balancing to directly connect to each vLLM backend,
    ensuring even distribution of requests across all backends. This avoids the issue
    where nginx load balancer may route all requests to the same backend.
    
    Features:
    - Direct connection to each backend (bypasses nginx)
    - Round-robin or least-requests load balancing strategy
    - Automatic retry with exponential backoff
    - Per-backend connection pooling
    - Load balancing statistics tracking
    
    Recommended batch_size:
    - With 8 vLLM backends (each MAX_SEQS=8), theoretical max is 64 concurrent requests
    - Conservative: 32-40 (50-62% of capacity, safer, more stable)
    - Balanced: 48-56 (75-87% of capacity, good balance, recommended)
    - Aggressive: 60-64 (94-100% of capacity, maximum throughput but risky)
    - Note: 64 is the theoretical limit with no headroom. If requests have varying processing times,
      some backends may be overloaded. Monitor backend status if using >56.
    
    Examples:
    python3 run_step2_gpt_profile.py --ds_name cd1 --ds_split train --start_idx 0 --end_idx -1 --batch_size 32
    python3 run_step2_gpt_profile.py --ds_name cd1 --ds_split validation --start_idx 0 --end_idx -1 --batch_size 32 --lb_strategy round_robin
    python3 run_step2_gpt_profile.py --ds_name cd1 --ds_split test --start_idx 0 --end_idx -1 --batch_size 32 --lb_strategy least_requests
    python3 run_step2_gpt_profile.py --ds_name cd1,cd2,ld1 --ds_split train --start_idx 0 --end_idx -1 --batch_size 32
    python3 run_step2_gpt_profile.py --ds_name all --ds_split train --start_idx 0 --end_idx -1 --batch_size 32
    python3 run_step2_gpt_profile.py --ds_name cd1 --ds_split all --start_idx 0 --end_idx -1 --batch_size 32
    python3 run_step2_gpt_profile.py --ds_name all --ds_split all --start_idx 0 --end_idx -1 --batch_size 32
    """

    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s - %(name)s] -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Step2 Get_Profile Args")
    parser.add_argument("--ds_name", type=str, default="cf1", 
                       help="Specify which dataset(s) to use. Can be: a single dataset name, comma-separated list (e.g., 'cd1,cd2,ld1'), or 'all' for all datasets")
    parser.add_argument("--ds_split", type=str, default="train", help="train OR validation OR test (or 'all' for all splits)")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for continue generating")
    parser.add_argument("--end_idx", type=int, default=-1, help="Ending index for continue generating")
    parser.add_argument("--batch_size", type=int, default=48, 
                       help="Batch size for concurrent requests. Recommended: 32-56 for 8 backends (MAX_SEQS=8 each). "
                            "64 is theoretical max but risky. Default: 48 (75% of capacity)")
    parser.add_argument("--lb_strategy", type=str, default="round_robin",
                       choices=["round_robin", "least_requests"],
                       help="Load balancing strategy: 'round_robin' (default) or 'least_requests'")
    parser.add_argument("--backend_ports", type=str, default="8000,8001,8002,8003,8004,8005,8006,8007",
                       help="Comma-separated list of backend ports. Default: 8000,8001,8002,8003,8004,8005,8006,8007")
    args = parser.parse_args()

    logger.info(args)

    ds_name_input = str(args.ds_name)
    ds_split = str(args.ds_split)
    start_idx = int(args.start_idx)
    end_idx = int(args.end_idx)
    batch_size = int(args.batch_size)
    lb_strategy = str(args.lb_strategy)
    backend_ports = [int(p.strip()) for p in args.backend_ports.split(",")]
    
    # Warn if batch_size is too high
    num_backends = len(backend_ports)
    theoretical_max = num_backends * 8  # Assuming MAX_SEQS=8 per backend
    if batch_size > theoretical_max:
        logger.warning(f">>> WARNING: batch_size ({batch_size}) exceeds theoretical max ({theoretical_max})!")
        logger.warning(f">>>   This may cause backend overload. Consider reducing batch_size.")
    elif batch_size > theoretical_max * 0.9:
        logger.warning(f">>> WARNING: batch_size ({batch_size}) is very close to theoretical max ({theoretical_max})")
        logger.warning(f">>>   This leaves little headroom. Monitor backend status carefully.")
    elif batch_size > theoretical_max * 0.75:
        logger.info(f">>> INFO: batch_size ({batch_size}) is {batch_size/theoretical_max*100:.1f}% of theoretical max ({theoretical_max})")
        logger.info(f">>>   This is aggressive but should work. Monitor for backend overload.")
    else:
        logger.info(f">>> batch_size ({batch_size}) is {batch_size/theoretical_max*100:.1f}% of theoretical max ({theoretical_max}) - safe")
    
    # Determine which datasets and splits to run
    # Support: "all", single dataset name, or comma-separated list
    if ds_name_input.lower() == "all":
        datasets_to_run = ALL_DATASETS
    else:
        # Parse comma-separated list and strip whitespace
        datasets_to_run = [name.strip() for name in ds_name_input.split(",") if name.strip()]
        # Validate that all specified datasets are in ALL_DATASETS
        invalid_datasets = [ds for ds in datasets_to_run if ds not in ALL_DATASETS]
        if invalid_datasets:
            logger.error(f">>> ERROR: Invalid dataset name(s): {invalid_datasets}")
            logger.error(f">>> Valid datasets are: {ALL_DATASETS}")
            sys.exit(1)
    
    if ds_split == "all":
        splits_to_run = ALL_SPLITS
    else:
        splits_to_run = [ds_split]

    cache_dir = "~/.cache/huggingface/"
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    cache_model = os.path.join(cache_dir, "models")
    cache_ds = os.path.join(cache_dir, "datasets")

    # Initialize client-side load balancer
    # Directly connect to each vLLM backend port for better control
    openai_api_key = "EMPTY"
    load_balancer = ClientLoadBalancer(backend_ports=backend_ports, openai_api_key=openai_api_key)
    
    logger.info(f">>> Load balancing strategy: {lb_strategy}")

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
                asyncio.run(run_openai(load_balancer, ds_name=cur_ds_name, ds_split=cur_ds_split, 
                                       start_idx=start_idx, end_idx=end_idx, 
                                       batch_size=batch_size, strategy=lb_strategy))
            except Exception as e:
                logger.error(f">>> ERROR running {cur_ds_name}---{cur_ds_split}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                continue

    sys.exit(0)
