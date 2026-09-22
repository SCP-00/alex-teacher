"""alex-teacher — Reconocimiento de voz (ASR).

faster-whisper (CTranslate2) en GPU con pre-filtrado Silero VAD.
"""

import struct
import threading
import time

import numpy as np
import torch

from util import _safe_print

HAVE_FASTER_WHISPER = False
_asr_models = {}
_asr_lock = threading.Lock()

try:
    from faster_whisper import WhisperModel
    HAVE_FASTER_WHISPER = True
except ImportError:
    _safe_print("[B] [!] faster-whisper no instalado. pip install faster-whisper")


def _find_wav_data_offset(audio_bytes):
    data_pos = audio_bytes.find(b'data', 12)
    return data_pos + 8 if data_pos >= 0 else 44


def _get_asr_model(model_name):
    if model_name in _asr_models:
        return _asr_models[model_name]
    with _asr_lock:
        if model_name in _asr_models:
            return _asr_models[model_name]
        try:
            t0 = time.time()
            # Fix: usar GPU explícitamente (antes estaba hardcoded a "cpu")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = WhisperModel(model_name, device=device, compute_type="int8")
            _safe_print(f"[B] [ASR] faster-whisper {model_name} cargado en GPU en {(time.time()-t0)*1000:.0f}ms")
            _asr_models[model_name] = model
            return model
        except Exception as e:
            _safe_print(f"[B] [!] Error cargando {model_name}: {e}")
            return None


def _asr_transcribe(audio_bytes, language="auto", model_name="small"):
    model = _get_asr_model(model_name)
    if model is None:
        return None, None, f"Modelo '{model_name}' no disponible"
    try:
        sr = 16000
        if len(audio_bytes) >= 44:
            sr = struct.unpack_from('<I', audio_bytes, 24)[0]
        offset = _find_wav_data_offset(audio_bytes)
        pcm_data = audio_bytes[offset:] if len(audio_bytes) > offset else audio_bytes[44:]
        pcm_int16 = np.frombuffer(pcm_data, dtype=np.int16)
        if len(pcm_int16) == 0:
            return None, None, "Audio vacío"
        pcm_float32 = pcm_int16.astype(np.float32) / 32768.0

        # Silero VAD: pre-filtrar segmentos de silencio antes de ASR
        try:
            from silero_vad import load_silero_vad as _load_vad, get_speech_timestamps as _get_speech_ts
            if not hasattr(_asr_transcribe, '_vad_model'):
                _asr_transcribe._vad_model = _load_vad()
                _asr_transcribe._vad_lock = threading.Lock()
            with _asr_transcribe._vad_lock:
                wav_tensor = torch.tensor(pcm_float32, dtype=torch.float32)
                speech_timestamps = _get_speech_ts(
                    wav_tensor, _asr_transcribe._vad_model,
                    sampling_rate=16000,
                    min_silence_duration_ms=500,
                    speech_pad_ms=200,
                )
                if speech_timestamps:
                    # Extraer solo segmentos con habla
                    speech_segments = []
                    for ts in speech_timestamps:
                        speech_segments.append(pcm_float32[ts['start']:ts['end']])
                    if speech_segments:
                        pcm_float32 = np.concatenate(speech_segments)
                        _safe_print(f"[B] [VAD] {len(speech_timestamps)} segmentos de habla detectados")
        except Exception:
            pass  # Silero VAD es opcional — si falla, usar Whisper VAD interno

        lang_code = None if language == "auto" else language
        with _asr_lock:
            segments, info = model.transcribe(
                pcm_float32, language=lang_code, beam_size=1,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200,
                                    threshold=0.5, neg_threshold=0.35, min_speech_duration_ms=200),
            )
        seg_list = []
        full_text = []
        for seg in segments:
            seg_list.append({"text": seg.text.strip(), "start": round(seg.start, 2), "end": round(seg.end, 2)})
            full_text.append(seg.text.strip())
        text = " ".join(full_text).strip()
        return text, seg_list, None
    except Exception as e:
        return None, None, str(e)[:300]
