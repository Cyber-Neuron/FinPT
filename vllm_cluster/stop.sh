#!/usr/bin/env bash

echo "🛑 Stopping vLLM instances..."
pkill -f "vllm serve Qwen/Qwen3-4B-Thinking-2507" || true

echo "🛑 Stopping nginx..."
nginx -s stop || true

echo "✅ All services stopped."

