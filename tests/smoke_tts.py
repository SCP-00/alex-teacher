#!/usr/bin/env python3
"""Smoke test para el stack TTS del ecosistema ALEX.

Valida Kokoro-ONNX para inglés, español y japonés, midiendo latencia de
generación. Los modelos se descargan automáticamente en el primer uso.

Uso:
    python3 tests/smoke_tts.py [--output-dir DIR]
"""
import argparse
import os
import time
from pathlib import Path

from kokoro_onnx import Kokoro

# Ruta canónica de modelos del ecosistema ALEX (~/.local/share/alex/models)
MODELS_DIR = Path(os.environ.get("ALEX_MODELS_DIR", Path.home() / ".local/share/alex/models"))
KOKORO_DIR = MODELS_DIR / "kokoro"


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test TTS (Kokoro ES/JA/EN)")
    parser.add_argument("--output-dir", default="tests/out", help="Carpeta de salida de los WAV")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_file = KOKORO_DIR / "kokoro-v1.0.onnx"
    voices_file = KOKORO_DIR / "voices-v1.0.bin"
    if not model_file.exists() or not voices_file.exists():
        raise SystemExit(
            f"Modelos no encontrados en {KOKORO_DIR}. Descárgalos con:\n"
            "  wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx\n"
            "  wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
        )

    t0 = time.perf_counter()
    print(f"Cargando modelos desde {KOKORO_DIR}...")
    kokoro = Kokoro(str(model_file), str(voices_file))
    print(f"Modelos listos en {time.perf_counter() - t0:.1f}s")

    # (texto, idioma espeak, voz)
    samples = [
        ("¡Hola! Este es Alex, tu asistente en español.", "es", "em_alex"),
        ("こんにちは、アレックスです。日本語で話します。", "ja", "jf_alpha"),
        ("Hello! This is Alex, your English assistant.", "en-us", "af_sarah"),
    ]

    for text, lang, voice in samples:
        t1 = time.perf_counter()
        samples_out, sr = kokoro.create(text, voice=voice, speed=1.0, lang=lang)
        dt = time.perf_counter() - t1
        path = out / f"kokoro_{lang}.wav"
        import soundfile as sf

        sf.write(path, samples_out, sr)
        print(f"[{lang}] {voice} · {dt*1000:.0f} ms · {len(samples_out)/sr:.1f}s audio → {path}")


if __name__ == "__main__":
    main()
