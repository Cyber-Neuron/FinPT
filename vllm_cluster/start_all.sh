#!/usr/bin/env bash
set -e

MODEL="Qwen/Qwen3-4B-Thinking-2507"
NUM_GPUS=8
BASE_PORT=8000
NGINX_PORT=9000

MAX_SEQS=8
MAX_BATCH_TOKENS=2048
MAX_MODEL_LEN=8192
GPU_MEM_UTIL=0.9

LOG_DIR="$(pwd)/logs"
NGINX_CONF="$(pwd)/nginx_vllm.conf"

echo "🚀 Starting vLLM instances..."

for ((i=0; i<NUM_GPUS; i++)); do
  PORT=$((BASE_PORT + i))
  LOG_FILE="${LOG_DIR}/vllm_${i}.log"

  echo "  - GPU $i → port $PORT"

  CUDA_VISIBLE_DEVICES=$i \
  nohup /home/dan/.local/bin/vllm serve "$MODEL" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --max-num-seqs $MAX_SEQS \
    --max-num-batched-tokens $MAX_BATCH_TOKENS \
    --max-model-len $MAX_MODEL_LEN \
    --port $PORT \
    > "$LOG_FILE" 2>&1 &
done

sleep 5

echo "🧩 Generating nginx config..."

cat > "$NGINX_CONF" <<EOF
worker_processes auto;

events {
    worker_connections 4096;
}

http {
    upstream vllm_backend {
        least_conn;
EOF

for ((i=0; i<NUM_GPUS; i++)); do
  PORT=$((BASE_PORT + i))
  echo "        server 127.0.0.1:${PORT};" >> "$NGINX_CONF"
done

cat >> "$NGINX_CONF" <<EOF
    }

    server {
        listen ${NGINX_PORT};

        location /v1/ {
            proxy_http_version 1.1;
            proxy_set_header Connection "close";
            proxy_set_header Host \$host;
            proxy_set_header X-Forwarded-For \$remote_addr;
            proxy_read_timeout 600s;
            proxy_send_timeout 600s;
            proxy_pass http://vllm_backend;
        }
    }
}
EOF

echo "🌐 Starting nginx..."
nginx -c "$NGINX_CONF"

echo "✅ All services started!"
echo "👉 API endpoint: http://127.0.0.1:${NGINX_PORT}/v1/chat/completions"

