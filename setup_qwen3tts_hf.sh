#!/usr/bin/env bash
# ============================================================
# setup_qwen3tts_hf.sh — descarga modelos Qwen3-TTS desde
# HuggingFace (sin gating, más rápido que ModelScope) con
# reanudación (curl -C -).
#
# Uso:
#   bash setup_qwen3tts_hf.sh                     # por defecto 0.6B-Base
#   MODEL=1.7b bash setup_qwen3tts_hf.sh          # 1.7B-Base (calidad)
#   WEBUI=1 bash setup_qwen3tts_hf.sh             # 1.7B + layout del qwen3-tts-webui
#   FULL=1  bash setup_qwen3tts_hf.sh             # + CustomVoice y VoiceDesign (~5GB más,
#                                                   pestañas del webui)
#   DEST=/ruta bash setup_qwen3tts_hf.sh          # destino custom
#
# Resultado (0.6B): ~/.local/share/alex/models/qwen3tts/       (teacher por defecto)
# Resultado (1.7B): ~/.local/share/alex/models/qwen3tts-1.7b/  (teacher con QWEN3_TTS_MODEL=1.7b)
# WEBUI=1 (con MODEL=1.7b): además enlaza el modelo a
#   ~/.local/share/alex/models/qwen3tts/Qwen3-TTS-12Hz-1.7B-Base/
#   (layout que espera `./alex start qwen3-tts-webui`)
# ============================================================
set -euo pipefail

MODEL="${MODEL:-0.6b}"
BASE_DIR="${DEST:-$HOME/.local/share/alex/models}"
if [ "$MODEL" = "1.7b" ]; then
    REPO="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    MODEL_DIR="$BASE_DIR/qwen3tts-1.7b"
else
    REPO="Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    MODEL_DIR="$BASE_DIR/qwen3tts"
fi
HF="https://huggingface.co/$REPO/resolve/main"

dl() {  # dl <ruta-destino> <path-en-repo>  — verifica contra Content-Length
    local dest="$1" rel="$2" expected="" got=0
    expected=$(curl -sIL --max-time 30 "$HF/$rel" | awk 'tolower($1)=="content-length:"{v=$2} END{print v}' | tr -d '\r')
    if [[ -n "$expected" ]] && [[ -f "$dest" ]] && [[ $(stat -c%s "$dest" 2>/dev/null || echo 0) -eq "$expected" ]]; then
        echo "✓ completo: $dest ($(du -h "$dest" | cut -f1))"
        return 0
    fi
    echo "⬇ $rel ..."
    curl -L --retry 5 -C - -o "$dest" "$HF/$rel"
    got=$(stat -c%s "$dest" 2>/dev/null || echo 0)
    if [[ -n "$expected" ]] && [[ "$got" -ne "$expected" ]]; then
        echo "⚠ incompleto: $got / $expected bytes — vuelve a ejecutar este script para reanudar"
    else
        echo "✓ $dest ($(du -h "$dest" | cut -f1))"
    fi
}

mkdir -p "$MODEL_DIR/speech_tokenizer"

for f in config.json configuration.json generation_config.json merges.txt \
         preprocessor_config.json tokenizer_config.json vocab.json; do
    dl "$MODEL_DIR/$f" "$f" || true
done
dl "$MODEL_DIR/speech_tokenizer/config.json" "speech_tokenizer/config.json" || true
dl "$MODEL_DIR/speech_tokenizer/configuration.json" "speech_tokenizer/configuration.json" || true
dl "$MODEL_DIR/speech_tokenizer/preprocessor_config.json" "speech_tokenizer/preprocessor_config.json" || true

dl "$MODEL_DIR/model.safetensors" "model.safetensors"
dl "$MODEL_DIR/speech_tokenizer/model.safetensors" "speech_tokenizer/model.safetensors"

# ── (Opcional) Modelos de las pestañas Custom Voice y Voice Design ──
# El webui tiene 3 pestañas; cada una usa un modelo distinto:
#   Voice Clone   → Base (arriba)
#   Custom Voice  → Qwen3-TTS-12Hz-1.7B-CustomVoice   (~2.5GB)
#   Voice Design  → Qwen3-TTS-12Hz-1.7B-VoiceDesign   (~2.5GB)
# FULL=1 los descarga también (layout que espera `./alex start qwen3-tts-webui`).
if [ "${FULL:-0}" = "1" ]; then
    for extra in "Qwen3-TTS-12Hz-1.7B-CustomVoice" "Qwen3-TTS-12Hz-1.7B-VoiceDesign"; do
        EXTRA_DIR="$BASE_DIR/qwen3tts/$extra"
        mkdir -p "$EXTRA_DIR/speech_tokenizer"
        HF="https://huggingface.co/Qwen/$extra/resolve/main"   # dl() usa $HF
        echo "⬇ $extra ..."
        for f in config.json configuration.json generation_config.json merges.txt \
                 preprocessor_config.json tokenizer_config.json vocab.json; do
            dl "$EXTRA_DIR/$f" "$f" || true
        done
        dl "$EXTRA_DIR/speech_tokenizer/config.json" "speech_tokenizer/config.json" || true
        dl "$EXTRA_DIR/speech_tokenizer/configuration.json" "speech_tokenizer/configuration.json" || true
        dl "$EXTRA_DIR/speech_tokenizer/preprocessor_config.json" "speech_tokenizer/preprocessor_config.json" || true
        dl "$EXTRA_DIR/model.safetensors" "model.safetensors"
        dl "$EXTRA_DIR/speech_tokenizer/model.safetensors" "speech_tokenizer/model.safetensors"
        echo "✓ $extra listo"
    done
fi

# ── (Opcional) Layout para qwen3-tts-webui ──
# El webui (./alex start qwen3-tts-webui) espera el modelo en
# <model-dir>/Qwen3-TTS-12Hz-1.7B-Base/. WEBUI=1 crea ese layout
# con symlinks (sin duplicar ~2.5GB).
if [ "${WEBUI:-0}" = "1" ] && [ "$MODEL" = "1.7b" ]; then
    TARGET="$BASE_DIR/qwen3tts/Qwen3-TTS-12Hz-1.7B-Base"
    mkdir -p "$TARGET/speech_tokenizer"
    for f in "$MODEL_DIR"/*; do
        [ -d "$f" ] && continue
        ln -sfn "$f" "$TARGET/$(basename "$f")"
    done
    for f in "$MODEL_DIR"/speech_tokenizer/*; do
        ln -sfn "$f" "$TARGET/speech_tokenizer/$(basename "$f")"
    done
    echo "✓ WebUI layout listo: $TARGET (symlinks)"
fi

echo
echo "=== Listo ($REPO). Resumen: ==="
du -sh "$MODEL_DIR"
