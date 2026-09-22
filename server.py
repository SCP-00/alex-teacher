#!/usr/bin/env python3
"""Alex Voice — Teacher + Conversation.

Servidor modular. Este archivo es solo el entrypoint: conecta los módulos
del proyecto (config, tts, asr, stats, llm, stories, handler) y arranca el
servidor HTTP. LLM en GPU, TTS híbrido Kokoro/Piper en CPU.
Corre en puerto 3000 por defecto.

Módulos:
    config    — configuración central (env, paths, constantes)
    util      — helpers compartidos (_safe_print)
    sanitize  — limpieza de texto para TTS y detección de idioma
    llm       — caché LRU + completado contra llama-server
    tts       — Kokoro-82M ONNX (primario) + Piper (fallback)
    asr       — faster-whisper con Silero VAD
    stats     — recolector de estadísticas del sistema
    stories   — feature Cuentos (LLM + TTS por personaje + embeddings)
    handler   — Handler HTTP (rutas)
    state     — singletons compartidos (cache, stats, piper)
"""

import sys
from http.server import ThreadingHTTPServer
from pathlib import Path

import state
from config import PORT, LLAMA_HOST, SERVER_MODE
from stats import StatsCollector, HAVE_NVML
from tts import PiperTTS, HAVE_KOKORO, _get_kokoro_instance
from asr import HAVE_FASTER_WHISPER, _get_asr_model
from handler import Handler
from util import _safe_print


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    state.stats = StatsCollector()
    # Threading: un request lento (story generate, streaming) no congela el resto
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)

    mode_name = SERVER_MODE if SERVER_MODE else "teacher"
    mode_label = {'teacher': '🎓 Teacher', 'conversation': '💬 Conversación'}.get(mode_name, mode_name)
    _safe_print(f"\n{'='*50}")
    _safe_print(f"  >> Alex Voice — {mode_label} (GPU+CPU)")
    _safe_print(f"{'='*50}")
    _safe_print(f"  Mode:         {mode_label}")
    _safe_print(f"  Web UI:       http://localhost:{PORT}")
    _safe_print(f"  Stats API:    http://localhost:{PORT}/api/stats")
    _safe_print(f"  llama-server: {LLAMA_HOST}")
    _safe_print(f"  Cache:        LRU {state.cache._maxsize} entradas")
    tts_str = "Kokoro-82M (principal) + Piper (fallback)" if HAVE_KOKORO else "Piper Python API (~45ms)"
    _safe_print(f"  TTS:          {tts_str}")
    _safe_print(f"  ASR:          faster-whisper small (GPU, ~1.5 GB VRAM)")
    _safe_print(f"{'='*50}\n")

    # Cargar ASR small al inicio (~1.5 GB VRAM en GPU)
    if HAVE_FASTER_WHISPER:
        _safe_print(f"[B] [ASR] Cargando faster-whisper small...")
        _get_asr_model("small")

    # Precargar Kokoro-ONNX: sin esto HAVE_KOKORO queda False y el handler
    # salta el único motor TTS que funciona (bug C4 del análisis)
    _safe_print(f"[B] [Kokoro] Precargando modelo TTS...")
    inst = _get_kokoro_instance()
    if inst is not None:
        _safe_print(f"[B] [Kokoro] Listo — TTS primario activo")
    else:
        _safe_print(f"[B] [Kokoro] No disponible (usa Piper como fallback)")

    # Cargar Piper TTS
    es_model = Path.home() / ".local/share/alex/models/piper/es_ES-sharvard-medium.onnx"
    en_model = Path.home() / ".local/share/alex/models/piper/en_US-lessac-medium.onnx"
    if es_model.exists() or en_model.exists():
        state.piper = PiperTTS(
            es_model_path=es_model if es_model.exists() else None,
            en_model_path=en_model if en_model.exists() else None,
        )
        if state.piper and state.piper.available:
            _safe_print(f"[B] [PiperTTS] Listo! Latencia ~45ms por sintesis")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        _safe_print("\n[B] Deteniendo...")
    finally:
        state.stats.stop()
        if state.piper:
            state.piper.stop()
        httpd.server_close()
        if HAVE_NVML:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
        _safe_print("[B] Detenido.")


if __name__ == "__main__":
    main()
