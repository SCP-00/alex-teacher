"""alex-teacher — Qwen3-TTS-12Hz-0.6B-Base con clonación de voz.

Cada mascota del cast (zorro, panda, dragón…) tiene una voz seed generada
con Kokoro (timbres masculinos/femeninos en es/en/ja). Qwen3-TTS clona esa
voz y sintetiza cualquier idioma con el mismo timbre: voces 100% distintas
por personaje, como Duolingo.

Carga en CPU por defecto (0 VRAM, no compite con llama-server en la GPU).
Con QWEN3_TTS_DEVICE=cuda usa la GPU cuando hay VRAM libre.
"""

import os
import threading
import time

from config import QWEN3_MODEL_PATH, QWEN3_REFS_DIR, STORY_CAST, KOKORO_CONFIG
from tts import _get_kokoro_instance, _kokoro_synthesize
from util import _safe_print

HAVE_QWEN3 = False
_qwen3_model = None
QWEN3_LOCK = threading.Lock()

# Texto de calibración por idioma nativo de la voz seed (~5s).
SEED_REF_TEXTS = {
    "es": "Me llamo Alex y me encanta explorar lugares nuevos cada día.",
    "en": "My name is Alex, and I love to explore new places every single day.",
    "ja": "私の名前はアレックスです。毎日新しい場所を探検するのが大好きです。",
}

# Idiomas soportados por Qwen3-TTS (nombres para el parámetro `language`).
QWEN3_LANG_NAMES = {
    "es": "Spanish",
    "en": "English",
    "ja": "Japanese",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ko": "Korean",
}


def _get_seed_voice(cast):
    """Voz seed del personaje (Kokoro) + su idioma nativo para el ref clip.

    Usa el PRIMER par del dict `seed_voice` (Python preserva el orden de
    inserción): así se controla exactamente qué timbre seed usa cada mascota.
    """
    seeds = cast.get("seed_voice") or {}
    for lang, voice in seeds.items():
        return voice, lang
    fallback = KOKORO_CONFIG["es"]["voice"]
    return fallback, "es"


def _qwen3_generate_refs(force=False):
    """Genera (si no existen) los clips de referencia de cada mascota con Kokoro.

    Cada personaje usa una voz seed distinta; el clip queda en
    <QWEN3_REFS_DIR>/<id>.wav + <id>.txt. Devuelve lista de (id, ok).
    """
    QWEN3_REFS_DIR.mkdir(parents=True, exist_ok=True)
    k = _get_kokoro_instance()
    if k is None:
        _safe_print("[B] [Qwen3-TTS] Kokoro no disponible para generar refs")
        return []
    results = []
    for cast in STORY_CAST:
        cid = cast["id"]
        wav_path = QWEN3_REFS_DIR / f"{cid}.wav"
        txt_path = QWEN3_REFS_DIR / f"{cid}.txt"
        if wav_path.exists() and txt_path.exists() and not force:
            results.append((cid, True))
            continue
        voice, lang = _get_seed_voice(cast)
        text = SEED_REF_TEXTS.get(lang, SEED_REF_TEXTS["es"])
        try:
            # Usar la voz seed directamente (NO _kokoro_synthesize que usa voz fija)
            cfg = KOKORO_CONFIG.get(lang, KOKORO_CONFIG["es"])
            audio, sr = k.create(text, voice=voice, speed=cfg.get("speed", 1.0), lang=cfg["lang"])
            if audio is None or sr is None or len(audio) == 0:
                results.append((cid, False))
                continue
            import numpy as np
            import soundfile as sf
            sf.write(str(wav_path), np.asarray(audio, dtype=np.float32), int(sr))
            txt_path.write_text(text, encoding="utf-8")
            _safe_print(f"[B] [Qwen3-TTS] Ref {cid} ({voice}/{lang}) -> {wav_path.name}")
            results.append((cid, True))
        except Exception as e:
            _safe_print(f"[B] [Qwen3-TTS] Ref {cid} error: {e}")
            results.append((cid, False))
    return results


def _get_qwen3_model():
    """Lazy singleton de Qwen3-TTS-0.6B-Base. Retorna modelo o None."""
    global _qwen3_model, HAVE_QWEN3
    if _qwen3_model is not None:
        return _qwen3_model
    with QWEN3_LOCK:
        if _qwen3_model is not None:
            return _qwen3_model
        if not QWEN3_MODEL_PATH.exists():
            _safe_print(f"[B] [Qwen3-TTS] Modelo no encontrado en {QWEN3_MODEL_PATH} (ejecuta download)")
            return None
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
            device = os.environ.get("QWEN3_TTS_DEVICE", "cpu")
            t0 = time.time()
            _safe_print(f"[B] [Qwen3-TTS] Cargando modelo ({device})… (puede tardar 30-60s)")
            used_device = device
            try:
                _qwen3_model = Qwen3TTSModel.from_pretrained(
                    str(QWEN3_MODEL_PATH),
                    device_map=device,
                    dtype=torch.bfloat16 if device != "cpu" else torch.float32,
                )
            except Exception as e:
                if device != "cpu":
                    # Fallback: si no cabe en GPU (p.ej. 1.7B en 6GB VRAM), CPU
                    _safe_print(f"[B] [Qwen3-TTS] Error en {device} ({e}) → reintento en CPU")
                    torch.cuda.empty_cache()
                    _qwen3_model = Qwen3TTSModel.from_pretrained(
                        str(QWEN3_MODEL_PATH),
                        device_map="cpu",
                        dtype=torch.float32,
                    )
                    used_device = "cpu"
                else:
                    raise
            elapsed = (time.time() - t0)
            HAVE_QWEN3 = True
            _safe_print(f"[B] [Qwen3-TTS] Modelo listo en {elapsed:.1f}s ({used_device})")
            return _qwen3_model
        except Exception as e:
            _safe_print(f"[B] [Qwen3-TTS] Error cargando: {e}")
            return None


def _qwen3_synth_line(text, cast_id, lang):
    """Clona la voz de la mascota y sintetiza `text` en `lang`.

    Retorna (sr, np_float32) o (None, None) si algo falla (caller hace fallback).
    """
    text = (text or "").strip()
    if not text:
        return None, None
    wav_path = QWEN3_REFS_DIR / f"{cast_id}.wav"
    txt_path = QWEN3_REFS_DIR / f"{cast_id}.txt"
    if not wav_path.exists() or not txt_path.exists():
        return None, None
    model = _get_qwen3_model()
    if model is None:
        return None, None
    ref_text = txt_path.read_text(encoding="utf-8").strip()
    lang_name = QWEN3_LANG_NAMES.get(lang, lang)
    try:
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=lang_name,
            ref_audio=str(wav_path),
            ref_text=ref_text,
        )
        if not wavs:
            return None, None
        import numpy as np
        return int(sr), np.asarray(wavs[0], dtype=np.float32)
    except Exception as e:
        _safe_print(f"[B] [Qwen3-TTS] Clone error ({cast_id}/{lang}): {e}")
        return None, None


def _qwen3_unload():
    """Libera el modelo (RAM/VRAM) — útil tras generaciones batch."""
    global _qwen3_model, HAVE_QWEN3
    if _qwen3_model is not None:
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    _qwen3_model = None
    HAVE_QWEN3 = False
    _safe_print("[B] [Qwen3-TTS] Modelo descargado de memoria")
