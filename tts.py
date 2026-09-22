"""alex-teacher — Motor TTS híbrido.

Kokoro-82M ONNX (CPU, 0 VRAM) como motor primario y Piper como fallback.
Incluye streaming por chunks y síntesis por archivo como último recurso.
"""

import os
import struct
import subprocess
import threading
import time

import numpy as np

from config import KOKORO_CONFIG, KOKORO_MODEL_PATH, KOKORO_VOICES_PATH
from util import _safe_print

# ── Kokoro-82M ONNX TTS (CPU, 0 VRAM) ──────────────────
# Reemplaza el Kokoro pip que depende de spacy/blis (incompatible con Python 3.13)
# Usa kokoro-onnx: misma calidad, ejecución 100% CPU, sin dependencias pesadas
HAVE_KOKORO = False
_kokoro_instance = None
KOKORO_LOCK = threading.Lock()


def _get_kokoro_instance():
    """Lazy-load singleton de Kokoro ONNX.
    Retorna instancia de Kokoro o None.
    """
    global _kokoro_instance, HAVE_KOKORO
    if _kokoro_instance is not None:
        return _kokoro_instance
    with KOKORO_LOCK:
        if _kokoro_instance is not None:
            return _kokoro_instance
        try:
            from kokoro_onnx import Kokoro
            if not KOKORO_MODEL_PATH.exists() or not KOKORO_VOICES_PATH.exists():
                _safe_print(f"[B] [Kokoro-ONNX] Modelos no encontrados en {KOKORO_MODEL_PATH.parent}")
                return None
            t0 = time.time()
            _kokoro_instance = Kokoro(
                str(KOKORO_MODEL_PATH),
                str(KOKORO_VOICES_PATH),
            )
            elapsed = (time.time() - t0) * 1000
            voices = _kokoro_instance.get_voices()
            _safe_print(f"[B] [Kokoro-ONNX] Cargado en {elapsed:.0f}ms — {len(voices)} voces")
            HAVE_KOKORO = True
            return _kokoro_instance
        except Exception as e:
            _safe_print(f"[B] [Kokoro-ONNX] Error cargando: {e}")
            return None


# ── Normalización japonés (kanji → hiragana) para el TTS ──────────────
# kokoro-onnx fonemiza 'ja' vía espeak-ng, que lee los kanji con lecturas
# CHINAS ("chinese character") y propaga esa lectura hasta el próximo kanji.
# pykakasi convierte kanji→hiragana con lecturas japonesas reales (QA 2026-09-22).
_kakasi = None
_kakasi_tried = False


def _ja_kana(text):
    """Convierte kanji a hiragana (lectura japonesa). Latin y kana quedan intactos."""
    global _kakasi, _kakasi_tried
    if not any('\u4e00' <= c <= '\u9fff' for c in text):  # sin kanji no hay nada que hacer
        return text
    if not _kakasi_tried:
        _kakasi_tried = True
        try:
            import pykakasi
            _kakasi = pykakasi.kakasi()
        except Exception as e:
            _safe_print(f"[B] [TTS] pykakasi no disponible ({e}): los kanji se leerán con fonemización china")
    if _kakasi is None:
        return text
    try:
        return ''.join(r['hira'] for r in _kakasi.convert(text))
    except Exception:
        return text


def _kokoro_synthesize(text, lang):
    """Sintetiza texto con Kokoro ONNX. Retorna (sample_rate, audio_float32) o (None, None)."""
    k = _get_kokoro_instance()
    if k is None:
        return None, None
    if lang == 'ja':
        text = _ja_kana(text)
    cfg = KOKORO_CONFIG.get(lang, KOKORO_CONFIG['es'])
    speed = cfg.get('speed', 1.0)
    try:
        audio, sr = k.create(text, voice=cfg['voice'], speed=speed, lang=cfg['lang'])
        return sr, audio
    except Exception as e:
        _safe_print(f"[B] [Kokoro-ONNX] Error sintetizando: {e}")
        return None, None


def _kokoro_synthesize_stream(text, lang):
    """Generator: sintetiza texto con Kokoro ONNX y produce tuplas (sample_rate, pcm_bytes).
    Retorna tuplas (sample_rate, pcm_int16_bytes).
    """
    k = _get_kokoro_instance()
    if k is None:
        return
    if lang == 'ja':
        text = _ja_kana(text)
    cfg = KOKORO_CONFIG.get(lang, KOKORO_CONFIG['es'])
    speed = cfg.get('speed', 1.0)
    try:
        audio, sr = k.create(text, voice=cfg['voice'], speed=speed, lang=cfg['lang'])
        if audio is None or len(audio) == 0:
            return
        audio_float32 = np.asarray(audio, dtype=np.float32)
        audio_float32 = np.clip(audio_float32, -1.0, 1.0)
        audio_int16 = (audio_float32 * 32767).astype(np.int16)
        yield (sr, audio_int16.tobytes())
    except Exception as e:
        _safe_print(f"[B] [Kokoro-ONNX] Stream error: {e}")


# ── Piper TTS Python API ───────────────────────────────────
HAVE_PIPER_PYTHON = False
try:
    from piper import PiperVoice
    HAVE_PIPER_PYTHON = True
except ImportError:
    _safe_print("[B] [!] piper-tts no instalado. pip install piper-tts")


class PiperTTS:
    """Modelos Piper cargados en memoria. Latencia ~45-65ms."""

    def __init__(self, es_model_path, en_model_path):
        self.es_voice = None
        self.en_voice = None
        self.es_model_path = str(es_model_path) if es_model_path else None
        self.en_model_path = str(en_model_path) if en_model_path else None
        self.available = False
        self._lock = threading.Lock()
        self._init_voices()

    def _init_voices(self):
        if not HAVE_PIPER_PYTHON:
            return
        if self.es_model_path:
            try:
                t0 = time.time()
                self.es_voice = PiperVoice.load(self.es_model_path, use_cuda=False)
                _safe_print(f"[B] [PiperTTS] ES cargado en {(time.time()-t0)*1000:.0f}ms")
            except Exception as e:
                _safe_print(f"[B] [PiperTTS] Error ES: {e}")
        if self.en_model_path:
            try:
                t0 = time.time()
                self.en_voice = PiperVoice.load(self.en_model_path, use_cuda=False)
                _safe_print(f"[B] [PiperTTS] EN cargado en {(time.time()-t0)*1000:.0f}ms")
            except Exception as e:
                _safe_print(f"[B] [PiperTTS] Error EN: {e}")
        self.available = (self.es_voice is not None) or (self.en_voice is not None)

    def synthesize(self, text, lang='es'):
        voice = self.es_voice if lang == 'es' else self.en_voice if lang == 'en' else (self.es_voice or self.en_voice)
        if voice is None:
            return None
        with self._lock:
            try:
                chunks = list(voice.synthesize(text))
                if not chunks:
                    return None
                sample_rate = getattr(chunks[0], 'sample_rate', 22050)
                int16_chunks = []
                for c in chunks:
                    if hasattr(c, 'audio_int16_array') and c.audio_int16_array is not None:
                        int16_chunks.append(c.audio_int16_array)
                    elif hasattr(c, 'audio_int16_bytes') and c.audio_int16_bytes is not None:
                        int16_chunks.append(np.frombuffer(c.audio_int16_bytes, dtype=np.int16))
                if not int16_chunks:
                    return None
                pcm = np.concatenate(int16_chunks)
                data_size = len(pcm) * 2
                buf = bytearray(44 + data_size)
                buf[0:4] = b'RIFF'
                struct.pack_into('<I', buf, 4, data_size + 36)
                buf[8:12] = b'WAVE'
                buf[12:16] = b'fmt '
                struct.pack_into('<I', buf, 16, 16)
                struct.pack_into('<H', buf, 20, 1)
                struct.pack_into('<H', buf, 22, 1)
                struct.pack_into('<I', buf, 24, sample_rate)
                struct.pack_into('<I', buf, 28, sample_rate * 2)
                struct.pack_into('<H', buf, 32, 2)
                struct.pack_into('<H', buf, 34, 16)
                buf[36:40] = b'data'
                struct.pack_into('<I', buf, 40, data_size)
                buf[44:44+data_size] = pcm.tobytes()
                return bytes(buf)
            except Exception as e:
                _safe_print(f"[B] [PiperTTS] Error: {e}")
                return None

    def synthesize_stream(self, text, lang='es'):
        voice = self.es_voice if lang == 'es' else self.en_voice if lang == 'en' else (self.es_voice or self.en_voice)
        if voice is None:
            return
        sr = 22050
        with self._lock:
            try:
                for c in voice.synthesize(text):
                    if sr == 22050:
                        sr = getattr(c, 'sample_rate', 22050)
                    if hasattr(c, 'audio_int16_array') and c.audio_int16_array is not None:
                        yield (sr, c.audio_int16_array.tobytes())
                    elif hasattr(c, 'audio_int16_bytes') and c.audio_int16_bytes is not None:
                        yield (sr, c.audio_int16_bytes)
            except Exception as e:
                _safe_print(f"[B] [PiperTTS] Stream error: {e}")

    def stop(self):
        self.es_voice = None
        self.en_voice = None
        self.available = False


def _run_piper_file(text, model_path, piper_exe):
    """Fallback TTS usando archivos temporales."""
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w', encoding='utf-8') as f:
            f.write(text)
            inp = f.name
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            out = f.name
        proc = subprocess.Popen(
            [str(piper_exe), "--model", str(model_path), "--output-file", out],
            stdin=open(inp, 'rb'), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        proc.wait(timeout=15)
        if proc.returncode == 0 and os.path.exists(out):
            data = open(out, 'rb').read()
            try: os.unlink(out); os.unlink(inp)
            except: pass
            return data if len(data) > 100 else None
        try: os.unlink(out); os.unlink(inp)
        except: pass
        return None
    except Exception as e:
        _safe_print(f"[B] [TTS-file] Error: {e}")
        return None
