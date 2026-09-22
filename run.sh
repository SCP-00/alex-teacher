#!/usr/bin/env bash
# Alex Teacher — Ejecutar (asume ./setup.sh ya ejecutado; si no, lo ejecuta por ti)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
MODELS_DIR="$HOME/.local/share/alex/models"
GGUF="$MODELS_DIR/llm/Qwen3.5-2B-Q4_K_M.gguf"
LLAMA_PORT="${LLAMA_PORT:-8080}"

if [ ! -d ".venv" ]; then
    echo "⚙️  Primera ejecución — preparando el entorno (setup.sh)..."
    bash "$ROOT/setup.sh"
fi

# ── llama-server (el chat lo necesita) ────────────────────────
if ! curl -s -m 2 "http://127.0.0.1:$LLAMA_PORT/health" >/dev/null 2>&1; then
    if command -v llama-server >/dev/null 2>&1 && [ -f "$GGUF" ]; then
        echo "⏳ Iniciando llama-server en :$LLAMA_PORT (carga en ~15s)..."
        nohup llama-server -m "$GGUF" --port "$LLAMA_PORT" \
            --ctx-size 8192 --n-gpu-layers 99 --reasoning off \
            > /tmp/alex-llama.log 2>&1 &
        for i in $(seq 1 60); do
            curl -s -m 2 "http://127.0.0.1:$LLAMA_PORT/health" >/dev/null 2>&1 && break
            sleep 1
        done
        curl -s -m 2 "http://127.0.0.1:$LLAMA_PORT/health" >/dev/null 2>&1 \
            && echo "✅ LLM listo" \
            || { echo "❌ llama-server no respondió — revisa /tmp/alex-llama.log"; exit 1; }
    else
        echo "⚠️  Sin llama-server/modelo: la web funciona pero el chat no. Ejecuta ./setup.sh"
    fi
fi

echo "🚀 Alex Teacher → http://localhost:3000"
exec "$ROOT/.venv/bin/python" $ENTRY "$@"
