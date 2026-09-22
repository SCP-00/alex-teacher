"""alex-teacher — Configuración central.

Todo lo que puede cambiar entre máquinas/entornos vive aquí:
variables de entorno, puertos, rutas de modelos y constantes globales.
Cada módulo del proyecto importa desde este archivo; nada importa aquí
desde los módulos de dominio (evita ciclos).
"""

import os
import sys
from pathlib import Path

# ── Modelo LLM ───────────────────────────────────────────────
# Motor: llama.cpp (llama-server) — inferencia local con GGUF, API OpenAI-compatible
# Modelo: Qwen3.5-2B-Q4_K_M.gguf (el campo "model" es informativo; llama-server lo ignora)
# Contexto: 32K optimizado para velocidad + VRAM mínima
LLAMA_MODEL = os.environ.get("LLAMA_MODEL", "Qwen3.5-2B-Q4_K_M")
NUM_CTX = int(os.environ.get("NUM_CTX", "8192"))  # 8K por defecto: más rápido, menos VRAM (configurable)
LLAMA_HOST = os.environ.get("LLAMA_HOST", "http://localhost:8080/v1")

# Soporte para --mode y --port desde línea de comandos (conv_server.py wrapper)
SERVER_MODE = os.environ.get("CONV_MODE", "")
PORT = int(os.environ.get("PLAN_B_PORT", "3000"))
for i, arg in enumerate(sys.argv):
    if arg == "--port" and i + 1 < len(sys.argv):
        PORT = int(sys.argv[i + 1])
    elif arg == "--mode" and i + 1 < len(sys.argv):
        SERVER_MODE = sys.argv[i + 1]

# ── Directorios del proyecto ─────────────────────────────────
FRONTEND_DIR = Path(__file__).parent / "frontend"
PROJECT_ROOT = Path(__file__).parent


# ── Kokoro-82M ONNX TTS (CPU, 0 VRAM) ────────────────────────
# Map: B language (es/en) -> Kokoro ONNX lang_code + voice + speed
# Velocidades ajustadas para naturalidad:
#   ES: 0.9x (más pausado, mejor pronunciación)
#   EN: 1.0x (natural)
#   JA: 0.9x (más claro para kanji)
KOKORO_CONFIG = {
    'es': {'lang': 'es', 'voice': 'em_alex', 'speed': 0.9},
    'en': {'lang': 'en-us', 'voice': 'af_heart', 'speed': 1.0},
    'ja': {'lang': 'ja', 'voice': 'jf_alpha', 'speed': 0.9},
    'fr': {'lang': 'fr-fr', 'voice': 'ff_siwis', 'speed': 1.0},
    'de': {'lang': 'de', 'voice': 'bf_emma', 'speed': 1.0},
}

# Paths para modelos ONNX (descargados por setup.sh)
KOKORO_MODEL_PATH = Path.home() / ".local/share/alex/models/onnx/kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = Path.home() / ".local/share/alex/models/onnx/voices-v1.0.bin"


# ── Cuentos (Stories) ────────────────────────────────────────
# El LLM escribe la historia; el TTS le pone voz a cada personaje;
# el audio + embedding se cachean para reproducir al instante sin regenerar.
STORY_DIR = Path.home() / ".local/share/alex/data/stories"
STORY_INDEX = STORY_DIR / "index.json"

# ── Qwen3-TTS (clonación de voz) ────────────────────────────────
# Cada mascota tiene una voz seed de Kokoro; Qwen3 clona ese timbre y
# lo habla en cualquier idioma. El modelo (Base = voice clone) se descarga
# a ~/.local/share/alex/models/. Carga en CPU por defecto (0 VRAM); con
# QWEN3_TTS_DEVICE=cuda usa la GPU (fallback automático a CPU si no cabe).
# Variante: QWEN3_TTS_MODEL=0.6b (rápida) | 1.7b (más calidad, requiere más RAM/VRAM)
QWEN3_MODEL_VERSION = os.environ.get("QWEN3_TTS_MODEL", "0.6b")
QWEN3_MODEL_PATH = Path.home() / ".local/share/alex/models" / ("qwen3tts-1.7b" if QWEN3_MODEL_VERSION == "1.7b" else "qwen3tts")
QWEN3_REFS_DIR = QWEN3_MODEL_PATH / "refs"

# Cast de personajes estilo Duolingo: cada mascota tiene una voz seed única
# (Kokoro, timbres distintos en es/en/ja). Qwen3-TTS clona la seed y habla
# cualquier idioma con el mismo timbre → voces 100% distintas por personaje.
# `seed_voice` indica qué voz Kokoro genera el clip de referencia; `voices`
# sigue siendo el fallback directo si Qwen3 no está disponible.
STORY_CAST = [
    {"id": "narrator", "name": "Narrator", "emoji": "🎙️",
     "seed_voice": {"es": "em_alex"},
     "voices": {"es": "em_alex", "ja": "jm_kumo", "en": "am_adam"}},
    {"id": "zorro", "name": "Zorro", "emoji": "🦊",
     "seed_voice": {"es": "em_santa"},
     "voices": {"es": "em_santa", "ja": "jf_tebukuro", "en": "am_michael"}},
    {"id": "panda", "name": "Panda", "emoji": "🐼",
     "seed_voice": {"en": "af_bella"},
     "voices": {"es": "ef_dora", "ja": "jf_alpha", "en": "af_bella"}},
    {"id": "dragon", "name": "Dragón", "emoji": "🐉",
     "seed_voice": {"en": "am_eric"},
     "voices": {"es": "em_alex", "ja": "jm_kumo", "en": "am_eric"}},
    {"id": "conejo", "name": "Conejo", "emoji": "🐰",
     "seed_voice": {"es": "ef_dora"},
     "voices": {"es": "ef_dora", "ja": "jf_gongitsune", "en": "af_nicole"}},
    {"id": "buho", "name": "Búho", "emoji": "🦉",
     "seed_voice": {"en": "bf_emma"},
     "voices": {"es": "em_santa", "ja": "jf_nezumi", "en": "bf_emma"}},
]
STORY_CAST_IDS = [c["id"] for c in STORY_CAST]
STORY_CAST_HINT = ", ".join(STORY_CAST_IDS)
