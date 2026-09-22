#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  Alex Teacher — Setup autónomo (100% independiente, sin monorepo)
#
#  1. Crea .venv e instala dependencias
#  2. Descarga modelos a ~/.local/share/alex/models/ (reanudable;
#     compartidos entre apps ALEX: si ya existen no se re-descargan)
#  3. Verifica llama.cpp (motor LLM local)
#
#  Uso:  ./setup.sh [--check]
# ═══════════════════════════════════════════════════════════════
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$HOME/.local/share/alex/models"
GGUF="$MODELS_DIR/llm/Qwen3.5-2B-Q4_K_M.gguf"
GGUF_URL="https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF/resolve/main/Qwen_Qwen3.5-2B-Q4_K_M.gguf"
KOKORO_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

green() { echo "  ✅ $1"; }; info() { echo "  ℹ️  $1"; }; warn() { echo "  ⚠️  $1"; }; fail() { echo "  ❌ $1"; }

# ── --check: diagnóstico sin instalar nada ────────────────────
if [ "${1:-}" = "--check" ]; then
    echo "═══ Diagnóstico Alex Teacher ═══"
    [ -d "$ROOT/.venv" ] && green "venv presente" || warn "venv ausente → ./setup.sh"
    "$ROOT/.venv/bin/python" -c "import kokoro_onnx, faster_whisper" >/dev/null 2>&1 && green "deps Python OK" || warn "deps incompletas → ./setup.sh"
    [ -f "$GGUF" ] && green "modelo LLM presente" || warn "falta Qwen3.5-2B GGUF (~1.4GB) → ./setup.sh"
    [ -f "$MODELS_DIR/onnx/kokoro-v1.0.onnx" ] && green "Kokoro TTS presente" || warn "falta Kokoro TTS → ./setup.sh"
    if command -v llama-server >/dev/null 2>&1; then green "llama.cpp OK ($(command -v llama-server))"
    else warn "llama-server no está en PATH — https://github.com/ggml-org/llama.cpp/releases"; fi
    curl -s -m 2 http://127.0.0.1:8080/health >/dev/null 2>&1 && green "llama-server activo :8080" || info "llama-server no activo (run.sh lo arranca)"
    exit 0
fi

echo "╔══════════════════════════════════════════════╗"
echo "║  Alex Teacher — Setup                                ║"
echo "╚══════════════════════════════════════════════╝"

# ── 1. Python venv + dependencias ─────────────────────────────
if [ ! -d "$ROOT/.venv" ]; then
    info "Creando .venv..."
    python3 -m venv "$ROOT/.venv"
fi
PY="$ROOT/.venv/bin/python"
"$PY" -m pip install --quiet --upgrade pip
info "Instalando dependencias (esto puede tardar varios minutos)..."
"$PY" -m pip install --quiet -r "$ROOT/requirements.txt"
green "Dependencias instaladas"

# ── 2. Modelos (reanudable) ───────────────────────────────────
mkdir -p "$MODELS_DIR/llm" "$MODELS_DIR/onnx"
dl() { # dl <url> <dest> <desc>
    if [ -f "$2" ]; then green "$3 presente"; return 0; fi
    info "Descargando $3 (reanudable)..."
    curl -L -C - --fail --retry 3 -o "$2.tmp" "$1" && mv "$2.tmp" "$2" && green "$3 descargado"         || { fail "Fallo descargando $3 — descárgalo manualmente en: $2"; return 1; }
}
dl "$GGUF_URL"   "$GGUF"                          "Qwen3.5-2B LLM (~1.4GB)" || true
dl "$KOKORO_URL" "$MODELS_DIR/onnx/kokoro-v1.0.onnx"  "Kokoro TTS modelo (~310MB)" || true
dl "$VOICES_URL" "$MODELS_DIR/onnx/voices-v1.0.bin"   "Kokoro TTS voces (~27MB)" || true

# ── 3. llama.cpp ──────────────────────────────────────────────
if command -v llama-server >/dev/null 2>&1; then
    green "llama.cpp: $(command -v llama-server)"
else
    warn "llama-server no está instalado (el chat lo necesita)."
    info "  Linux:   sudo apt install llama.cpp  — o binario oficial:"
    info "           https://github.com/ggml-org/llama.cpp/releases"
    info "  macOS:   brew install llama.cpp"
fi

echo ""
green "Setup completo. Lanza la app con:  ./run.sh"
echo "    Web UI:  http://localhost:3000"
