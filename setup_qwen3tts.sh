#!/usr/bin/env bash
# ============================================================
# setup_qwen3tts.sh — descarga Qwen3-TTS-12Hz (Base, clonación
# de voz) desde ModelScope sin gating.
#
# Uso:
#   bash setup_qwen3tts.sh                     # por defecto 0.6B-Base
#   MODEL=1.7b bash setup_qwen3tts.sh          # 1.7B-Base (más calidad)
#   QWEN3_MODEL_DIR=/ruta bash setup_qwen3tts.sh  # destino custom
#
# Resultado:
#   0.6b → ~/.local/share/alex/models/qwen3tts/       (usa teacher por defecto)
#   1.7b → ~/.local/share/alex/models/qwen3tts-1.7b/  (mejor calidad de clonación)
#
# Los pesos: model.safetensors (el LM de TTS) + speech_tokenizer/
# (codec 12Hz) + config/tokenizer/vocab.
#
# Nota: usa el CDN directo de ModelScope (varias veces más rápido
# que snapshot_download). Reanudable con curl -C -. Si el repo 1.7B
# no está disponible en ModelScope, usa setup_qwen3tts_hf.sh (HF).
# ============================================================
set -euo pipefail

MODEL="${MODEL:-0.6b}"
BASE_DIR="${QWEN3_MODEL_DIR:-$HOME/.local/share/alex/models}"

if [ "$MODEL" = "1.7b" ]; then
    REPO="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    MODEL_DIR="$BASE_DIR/qwen3tts-1.7b"
else
    REPO="Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    MODEL_DIR="$BASE_DIR/qwen3tts"
fi
BASE="https://www.modelscope.cn/api/v1/models/$REPO/repo?Revision=master&FilePath="

dl() {  # dl <ruta-destino> <path-en-repo>
  local dest="$1" rel="$2"
  if [[ -f "$dest" && -s "$dest" ]]; then
    echo "✓ existe: $dest"
    return 0
  fi
  echo "⬇ $rel ..."
  curl -L --retry 5 -C - -o "$dest" "${BASE}${rel}"
  echo "✓ $dest ($(du -h "$dest" | cut -f1))"
}

mkdir -p "$MODEL_DIR/speech_tokenizer"

# Archivos del snapshot (pequeños)
for f in config.json configuration.json generation_config.json merges.txt \
         preprocessor_config.json tokenizer_config.json vocab.json; do
  dl "$MODEL_DIR/$f" "$f" || true
done
dl "$MODEL_DIR/speech_tokenizer/config.json" "speech_tokenizer/config.json" || true
dl "$MODEL_DIR/speech_tokenizer/configuration.json" "speech_tokenizer/configuration.json" || true
dl "$MODEL_DIR/speech_tokenizer/preprocessor_config.json" "speech_tokenizer/preprocessor_config.json" || true

# Pesos grandes
dl "$MODEL_DIR/model.safetensors" "model.safetensors"
dl "$MODEL_DIR/speech_tokenizer/model.safetensors" "speech_tokenizer/model.safetensors"

echo
echo "=== Listo ($REPO). Resumen: ==="
du -sh "$MODEL_DIR"
