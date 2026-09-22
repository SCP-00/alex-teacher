"""alex-teacher — Handler HTTP.

Rutas del servidor: chat (proxy LLM), TTS (kokoro/piper), ASR (whisper),
stats, caché y la feature de Cuentos. El estado mutable (state.cache,
state.stats, state.piper) se asigna en server.main().
"""

import base64
import json
import os
import re
import shutil
import struct
import time
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler

import numpy as np
from pathlib import Path

import alexlog
import state
import tts
import asr
from config import (
    LLAMA_HOST, NUM_CTX, LLAMA_MODEL,
    SERVER_MODE, FRONTEND_DIR, KOKORO_CONFIG, STORY_DIR,
)
from prompts import parse_multi_output, get_tts_text, get_system_prompt
from sanitize import (
    detect_language, _sanitize_tts_text, _split_sentences,
    _make_silence_pcm, _fix_word_spacing,
)
from tts import _kokoro_synthesize, _kokoro_synthesize_stream, _run_piper_file
from asr import _get_asr_model, _asr_transcribe
from stories import (
    _story_load, _story_delete, _story_generate, _story_search,
    _story_load_index, _safe_story_id,
)
from util import _safe_print


def _strip_think_stream(text, in_think):
    """Filtra bloques <think>...</think> de un chunk de streaming, con estado.

    Retorna (texto_limpio, nuevo_estado). El estado in_think se mantiene
    entre chunks porque llama.cpp (modelos de razonamiento) puede partir el
    bloque <think> en varios chunks de red.
    """
    out = []
    remaining = text
    while remaining:
        if in_think:
            idx = remaining.find("</think>")
            if idx == -1:
                remaining = ""  # seguimos dentro del bloque de razonamiento
            else:
                in_think = False
                remaining = remaining[idx + len("</think>"):]
        else:
            idx = remaining.find("<think>")
            if idx == -1:
                out.append(remaining)
                remaining = ""
            else:
                out.append(remaining[:idx])
                in_think = True
                remaining = remaining[idx + len("<think>"):]
    return "".join(out).strip(), in_think


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def do_GET(self):
        rid = None
        with alexlog.request_ctx("http.get", "handler") as _rid:
            rid = _rid
            self._do_get(rid)

    def _do_get(self, rid):
        if self.path == "/api/stats":
            self._json_response(state.stats.get_stats())
        elif self.path == "/api/cache/stats":
            self._json_response(state.cache.stats())
        elif self.path == "/api/cache/clear":
            state.cache.invalidate()
            self._json_response({"ok": True})
        elif self.path == "/api/stories":
            self._json_response({"stories": _story_load_index()})
        elif self.path.startswith("/api/stories/"):
            self._serve_story(self.path)
        elif self.path in ("/", "/index.html", "/conv.html"):
            # Solo existe index.html (conv.html nunca se creó) — servir siempre index.html
            self._serve_file("index.html")
        else:
            super().do_GET()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b"{}"
        with alexlog.request_ctx("http.post", "handler") as rid:
            self._do_post(body, rid)

    def _do_post(self, body, rid):
        if self.path == "/api/chat":
            self._handle_chat(body, rid)
        elif self.path == "/api/tts":
            self._handle_tts(body, rid)
        elif self.path == "/api/tts/stream":
            self._handle_tts_stream(body, rid)
        elif self.path == "/api/asr":
            self._handle_asr(body, rid)
        elif self.path == "/api/stories/generate":
            self._handle_stories_generate(body, rid)
        elif self.path == "/api/stories/search":
            self._handle_stories_search(body, rid)
        elif self.path.startswith("/api/stories/") and self.path.endswith("/delete"):
            self._handle_stories_delete(body, rid)
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_chat(self, body, rid=None):
        """Proxy a llama-server con caché LRU y multi-output parsing.

        Acepta un campo 'mode' (teacher/conversation) para usar
        system prompts optimizados con formato multi-output (【TEXT】/【PRONUNCIATION】/【TRANSLATION】).
        """
        try:
            data = json.loads(body)
            mode = data.get('mode', 'conversation')
            target_lang = data.get('target_lang', '')
            t0 = time.time()
            stream_chars = 0

            # Validación: sin un mensaje de usuario con contenido no tiene
            # sentido gastar tokens del LLM (curl con body vacío = dinero/quemado)
            _has_user_text = False
            if isinstance(data.get("messages"), list):
                for m in data["messages"]:
                    if m.get("role") == "user" and str(m.get("content", "")).strip():
                        _has_user_text = True
                        break
            elif isinstance(data.get("text"), str) and data["text"].strip():
                _has_user_text = True
            if not _has_user_text:
                self._json_response({"error": "No hay mensaje de usuario para responder"})
                return

            if "messages" not in data and "prompt" in data:
                prompt = data["prompt"]
                parts = re.split(r'<\|im_start\|>(\w+)\n', prompt)
                messages = []
                for i in range(1, len(parts)-1, 2):
                    role = parts[i]
                    content = parts[i+1].replace('<|im_end|>', '').strip()
                    if role == 'system':
                        messages.insert(0, {"role": "system", "content": content})
                    elif role in ('user', 'assistant'):
                        messages.append({"role": role, "content": content})
                chat_data = {
                    "messages": messages,
                    "model": LLAMA_MODEL,
                    "n_predict": min(int(data.get("n_predict", 512)), 1024),  # cap anti-DoS (QA)
                    "temperature": data.get("temperature", 0.7),
                    "stream": data.get("stream", False),
                    "options": {"num_ctx": NUM_CTX},
                    "keep_alive": "2m",
                    "reasoning_format": "none",
                    # Qwen3.5 razona por defecto: sin esto, agota n_predict en <think> y devuelve content="" (QA 2026-09-22)
                    "chat_template_kwargs": {"enable_thinking": os.environ.get("ALEX_ENABLE_THINKING", "0") == "1"},
                }
            elif "messages" in data:
                messages = list(data.get("messages") or [])
                has_system = bool(messages and messages[0].get("role") == "system")
                if not has_system:
                    messages.insert(0, {"role": "system", "content": get_system_prompt(mode)})
                if target_lang:
                    for msg in reversed(messages):
                        if msg.get("role") == "user":
                            msg["content"] = f"{msg.get('content', '')}\n[Target language: {target_lang}]"
                            break
                # Teacher: lower temperature (0.3) for accuracy, lower n_predict for speed
                # Conversation: 0.7 for naturalness
                chat_data = {
                    "model": LLAMA_MODEL,
                    "messages": messages,
                    "n_predict": min(int(data.get("n_predict", 384 if mode == 'teacher' else 512)), 1024),  # cap anti-DoS
                    "temperature": data.get("temperature", {
                        'teacher': 0.3, 'conversation': 0.7
                    }.get(mode, 0.7)),
                    "stream": data.get("stream", False),
                    "options": {"num_ctx": NUM_CTX},
                    "keep_alive": "2m",
                    "reasoning_format": "none",
                    # Qwen3.5 razona por defecto: sin esto, agota n_predict en <think> y devuelve content="" (QA 2026-09-22)
                    "chat_template_kwargs": {"enable_thinking": os.environ.get("ALEX_ENABLE_THINKING", "0") == "1"},
                }
            else:
                # Build messages with proper system prompt from translator module
                user_text = data.get('text', '')
                sys_prompt = get_system_prompt(mode)
                messages = [{'role': 'system', 'content': sys_prompt}]
                # Add target language instruction for translator mode
                user_content = user_text
                if target_lang:
                    user_content = f"{user_text}\n[Target language: {target_lang}]"
                messages.append({'role': 'user', 'content': user_content})
                chat_data = {
                    "model": LLAMA_MODEL,
                    "messages": messages,
                    "n_predict": min(int(data.get("n_predict", 384 if mode == 'teacher' else 1024)), 1024),  # cap anti-DoS
                    "temperature": data.get("temperature", {
                        'teacher': 0.3, 'conversation': 0.7
                    }.get(mode, 0.7)),
                    "stream": data.get("stream", False),
                    "options": {"num_ctx": NUM_CTX},
                    "keep_alive": "2m",
                    "reasoning_format": "none",
                    # Qwen3.5 razona por defecto: sin esto, agota n_predict en <think> y devuelve content="" (QA 2026-09-22)
                    "chat_template_kwargs": {"enable_thinking": os.environ.get("ALEX_ENABLE_THINKING", "0") == "1"},
                }

            # Cache check (non-streaming only)
            if not chat_data.get("stream"):
                cached = state.cache.get(chat_data.get("messages", []))
                if cached:
                    _safe_print(f"[B] [proxy] Cache HIT! {len(cached)} chars")
                    result = {
                        "choices": [{"message": {"content": cached}}],
                        "usage": {"completion_tokens": 0, "prompt_tokens": 0},
                        "cached": True,
                    }
                    self._json_response(result)
                    return

            target = f"{LLAMA_HOST}/chat/completions"
            req = urllib.request.Request(
                target,
                data=json.dumps(chat_data).encode(),
                headers={"Content-Type": "application/json"},
            )

            if chat_data.get("stream"):
                with urllib.request.urlopen(req, timeout=200) as resp:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("X-Mode", mode)
                    self.end_headers()
                    buf = b''
                    in_think = False
                    while True:
                        chunk = resp.read(4096)
                        if not chunk:
                            break
                        buf += chunk
                        lines = buf.split(b'\n')
                        buf = lines.pop()
                        for line in lines:
                            if line.startswith(b'data: '):
                                try:
                                    data_str = line[6:].decode('utf-8', errors='replace').strip()
                                    if data_str and data_str != '[DONE]':
                                        parsed = json.loads(data_str)
                                        choices = parsed.get('choices', [])
                                        if choices and 'delta' in choices[0]:
                                            delta = choices[0]['delta']
                                            if 'content' in delta and delta['content']:
                                                # Filtrar razonamiento <think> con estado entre chunks
                                                filtered, in_think = _strip_think_stream(delta['content'], in_think)
                                                if filtered:
                                                    delta['content'] = filtered
                                                    stream_chars += len(filtered)
                                                else:
                                                    delta.pop('content', None)
                                        line = b'data: ' + json.dumps(parsed).encode()
                                except Exception:
                                    pass
                            try:
                                self.wfile.write(line + b'\n')
                                self.wfile.flush()
                            except Exception:
                                break
                    # Métricas reales: tokens ≈ chars/4 (estimación para streaming)
                    if state.stats and stream_chars:
                        state.stats.record_request(
                            prompt_tokens=0, completion_tokens=max(1, stream_chars // 4),
                            elapsed=time.time() - t0)
            else:
                with urllib.request.urlopen(req, timeout=200) as resp:
                    result = json.loads(resp.read().decode())
                # Métricas reales del usage que devuelve llama-server
                if state.stats:
                    usage = result.get("usage") or {}
                    state.stats.record_request(
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        elapsed=time.time() - t0)
                content = ""
                choices = result.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                # Strip thinking tags
                cleaned = re.sub(r'<think>[\s\S]*?</think>\s*', '', content).strip()
                if not cleaned:
                    cleaned = content
                # Si el modelo se cortó a mitad del razonamiento (<think> sin cerrar),
                # descartar todo desde el <think> abierto (llama.cpp Qwen3.5 razona)
                if "<think>" in cleaned:
                    cleaned = cleaned.split("<think>", 1)[0].strip()
                # Debug: log the raw LLM response (first 200 chars) + RAW completo a alexlog
                _safe_print(f"[B] [teacher] Raw LLM response ({len(cleaned)} chars): {cleaned[:200]}...")
                alexlog.raw("llm.raw", "handler",
                            f"LLM {len(cleaned)} chars (mode={mode})",
                            payload={"mode": mode, "content": cleaned},
                            request_id=rid)

                # Parse multi-output for teacher mode
                parsed = parse_multi_output(cleaned) if mode == 'teacher' else {}

                # Debug: log parsed fields
                if parsed:
                    for k, v in parsed.items():
                        if v:
                            _safe_print(f"[B] [teacher] Parsed {k}: {v[:80]}...")
                else:
                    _safe_print(f"[B] [teacher] WARNING: parse_multi_output returned empty dict!")

                # Fix word spacing: apply to explanation/exercise (prose) only
                # Skip tts_reading and pronunciation — they contain romaji with hyphens/parentheses
                if parsed:
                    for key in ('explanation', 'exercise', 'translation'):
                        if parsed.get(key):
                            parsed[key] = _fix_word_spacing(parsed[key])
                if choices:
                    choices[0]["message"]["content"] = cleaned
                # Always include parsed in response (even if empty)
                result['parsed'] = parsed
                result['mode'] = mode
                if cleaned and len(cleaned) > 20:
                    state.cache.put(chat_data.get("messages", []), cleaned)
                self._json_response(result)

        except urllib.error.HTTPError as e:
            err = e.read().decode(errors='replace')[:500]
            _safe_print(f"[B] [proxy] llama-server HTTP {e.code}")
            self._json_response({"error": f"HTTP {e.code}: {err}"})
        except Exception as e:
            _safe_print(f"[B] [proxy] {type(e).__name__}: {e}")
            self._json_response({"error": str(e)[:300]})

    def _handle_tts(self, body, rid=None):
        """TTS híbrido: Kokoro-82M (primario, streaming real) → Piper (fallback).

        Estrategia:
        1. Kokoro: mejor calidad, streaming real (~2.5x EN, 1.0x ES)
        2. Piper: ~45ms, siempre disponible como respaldo
        3. subprocess Piper: último recurso
        """
        t0 = time.time()
        try:
            data = json.loads(body)
            text = data.get("text", "").strip()
            lang = data.get("lang", "auto")
            mode = data.get("mode", "conversation")

            # If mode is teacher, extract only the TTS-relevant text
            if mode == 'teacher':
                text = get_tts_text(text, mode)
            if not text:
                self._json_response({"error": "texto vacio"})
                return

            # Sanitizar: eliminar caracteres no pronunciables (CJK, etc.)
            # Importante: pasar lang para preservar CJK en japonés
            if lang not in KOKORO_CONFIG:
                detected = detect_language(text)
                lang = detected if detected in KOKORO_CONFIG else 'es'
            text = _sanitize_tts_text(text, lang)
            if not text or len(text) < 2:
                self._json_response({"error": "Texto sin caracteres pronunciables"})
                return

            wav_data = None
            method = "kokoro"

            # ── 1. Kokoro-82M ONNX (primario, CPU) ──
            # Para textos largos: secuencializar por oraciones con silencio entre ellas
            # Para textos cortos (<100 chars): sintetizar de una vez
            if tts.HAVE_KOKORO:
                try:
                    sr_val = None
                    if len(text) > 100:
                        # Sequential TTS: split sentences, add silence gaps
                        sentences = _split_sentences(text, lang)
                        all_pcm = []
                        for sent in sentences:
                            s, audio = _kokoro_synthesize(sent, lang)
                            if audio is not None and len(audio) > 0:
                                sr_val = s
                                a = np.asarray(audio, dtype=np.float32)
                                a = np.clip(a, -1.0, 1.0)
                                all_pcm.append((a * 32767).astype(np.int16))
                                # Add 300ms silence between sentences
                                if sent != sentences[-1]:
                                    all_pcm.append(np.frombuffer(
                                        _make_silence_pcm(sr_val, 300), dtype=np.int16))
                        if all_pcm:
                            full_int16 = np.concatenate(all_pcm)
                            method = f"kokoro_sequential({len(sentences)})"
                        else:
                            full_int16 = None
                    else:
                        s, audio = _kokoro_synthesize(text, lang)
                        sr_val = s
                        if audio is not None and len(audio) > 0:
                            a = np.asarray(audio, dtype=np.float32)
                            full_int16 = (np.clip(a, -1.0, 1.0) * 32767).astype(np.int16)
                            method = "kokoro_onnx"
                        else:
                            full_int16 = None

                    if full_int16 is not None and len(full_int16) > 0:
                        data_size = len(full_int16) * 2
                        buf = bytearray(44 + data_size)
                        buf[0:4] = b'RIFF'
                        struct.pack_into('<I', buf, 4, data_size + 36)
                        buf[8:12] = b'WAVE'
                        buf[12:16] = b'fmt '
                        struct.pack_into('<I', buf, 16, 16)
                        struct.pack_into('<H', buf, 20, 1)
                        struct.pack_into('<H', buf, 22, 1)
                        struct.pack_into('<I', buf, 24, sr_val)
                        struct.pack_into('<I', buf, 28, sr_val * 2)
                        struct.pack_into('<H', buf, 32, 2)
                        struct.pack_into('<H', buf, 34, 16)
                        buf[36:40] = b'data'
                        struct.pack_into('<I', buf, 40, data_size)
                        buf[44:44+data_size] = full_int16.tobytes()
                        wav_data = bytes(buf)
                    if sr_val is None:
                        sr_val = 24000  # Kokoro default
                except Exception as e:
                    _safe_print(f"[B] [TTS] Kokoro-ONNX falló: {e}")
                    method = "kokoro_fallback"

            # ── 2. Piper Python API (fallback) ──
            if wav_data is None or len(wav_data) < 100:
                if state.piper and state.piper.available:
                    wav_data = state.piper.synthesize(text, lang)
                    method = "piper_python_api"

            # ── 3. Piper subprocess (último recurso) ──
            if wav_data is None or len(wav_data) < 100:
                # En Linux el binario está en PATH; en Windows en models/piper/
                piper_exe = shutil.which("piper") or Path.home() / ".local/share/alex/models/piper/piper.exe"
                es_model = Path.home() / ".local/share/alex/models/piper/es_ES-sharvard-medium.onnx"
                en_model = Path.home() / ".local/share/alex/models/piper/en_US-lessac-medium.onnx"
                mp = en_model if lang == 'en' and en_model.exists() else es_model
                wav_data = _run_piper_file(text, mp, piper_exe)
                method = "subprocess_fallback"

            if not wav_data or len(wav_data) < 100:
                self._json_response({"error": "No se genero audio"})
                return

            dur = round((time.time() - t0) * 1000)
            _safe_print(f"[B] [TTS] {len(wav_data)}B en {dur}ms ({method})")
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(wav_data)))
            self.end_headers()
            self.wfile.write(wav_data)

        except Exception as e:
            _safe_print(f"[B] [TTS] Error: {e}")
            self._json_response({"error": str(e)[:200]})

    def _handle_tts_stream(self, body, rid=None):
        """Streaming TTS híbrido: Kokoro-82M (primario) → Piper (fallback).

        Kokoro genera audio chunk por chunk (cada segmento de texto).
        Cada chunk se envía al cliente en cuanto está listo.
        Si Kokoro no está disponible, cae en Piper streaming.
        """
        t0 = time.time()
        headers_sent = False
        engine_used = ""
        try:
            data = json.loads(body)
            text = data.get("text", "").strip()
            lang = data.get("lang", "auto")
            if not text:
                self._json_response({"error": "texto vacio"})
                return
            # Sanitizar: eliminar caracteres no pronunciables (CJK, etc.)
            # Importante: pasar lang para preservar CJK en japonés
            if lang not in KOKORO_CONFIG:
                detected = detect_language(text)
                lang = detected if detected in KOKORO_CONFIG else 'es'
            text = _sanitize_tts_text(text, lang)
            if not text or len(text) < 2:
                self._json_response({"error": "Texto sin caracteres pronunciables"})
                return

            total_pcm = 0
            stream = None

            # ── 1. Kokoro-ONNX streaming ──
            if tts.HAVE_KOKORO:
                stream = _kokoro_synthesize_stream(text, lang)
                engine_used = "kokoro_onnx"

            # ── 2. Piper streaming (fallback si Kokoro no disponible) ──
            if stream is None:
                if state.piper and state.piper.available:
                    stream = state.piper.synthesize_stream(text, lang)
                    engine_used = "piper"
                else:
                    self._json_response({"error": "TTS no disponible"})
                    return

            for sr, pcm in stream:
                if not headers_sent:
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/L16")
                    self.send_header("X-Sample-Rate", str(sr))
                    self.send_header("X-Channels", "1")
                    self.send_header("X-Bits-Per-Sample", "16")
                    self.send_header("X-TTS-Engine", engine_used)
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    headers_sent = True
                self.wfile.write(pcm)
                self.wfile.flush()
                total_pcm += len(pcm)
            dur = round((time.time() - t0) * 1000)
            _safe_print(f"[B] [TTS-stream] {total_pcm}B en {dur}ms ({engine_used})")
        except Exception as e:
            _safe_print(f"[B] [TTS-stream] Error: {e}")
            if not headers_sent:
                self._json_response({"error": str(e)[:200]})

    def _handle_asr(self, body, rid=None):
        """ASR con faster-whisper. Auto-switch: base (ES/EN), small (JA)."""
        t0 = time.time()
        lang = "auto"
        try:
            data = json.loads(body)
            audio_b64 = data.get("audio", "")
            lang = data.get("lang", "auto")
            if not audio_b64:
                self._json_response({"error": "No audio"})
                return
            raw_audio = base64.b64decode(audio_b64)
        except Exception as e:
            self._json_response({"error": f"Error: {str(e)[:100]}"})
            return

        if len(raw_audio) < 100:
            self._json_response({"error": "Audio demasiado pequeño"})
            return

        text_result = None
        segments_result = None
        error_msg = None
        method_used = "faster-whisper"
        model_ok = asr.HAVE_FASTER_WHISPER and _get_asr_model("small") is not None

        if model_ok:
            try:
                text_result, segments_result, error_msg = _asr_transcribe(raw_audio, lang)
            except Exception as e:
                error_msg = str(e)[:200]

        # Distinguir casos: modelo no disponible vs silencio (texto vacío válido)
        if not model_ok:
            self._json_response({"error": error_msg or "ASR no disponible"})
            return
        if text_result is None:
            self._json_response({"error": error_msg or "Error transcribiendo audio"})
            return

        if not text_result.strip():
            # Silencio / sin voz detectable: no es error del modelo
            self._json_response({"text": "", "segments": [], "language": lang,
                                 "duration_ms": round((time.time() - t0) * 1000),
                                 "empty": True})
            return  # sin return: se escribía una 2ª respuesta HTTP encima (JSON corrupto)

        dur = round((time.time() - t0) * 1000)
        _safe_print(f"[B] [ASR] {len(text_result)} chars en {dur}ms ({method_used})")
        self._json_response({
            "text": text_result,
            "segments": segments_result or [],
            "language": lang,
            "duration_ms": dur,
        })

    # ── Cuentos: handlers ───────────────────────────────────────
    def _serve_story(self, path):
        """Sirve meta o audio de un cuento (el audio cacheado se sirve al instante)."""
        parts = path.split("/")
        if len(parts) < 4:
            self.send_response(404)
            self.end_headers()
            return
        sid = parts[3]
        if not _safe_story_id(sid):
            self.send_response(404)
            self.end_headers()
            return
        if path.endswith("/audio"):
            wav_path = STORY_DIR / sid / "audio.wav"
            story = _story_load(sid)
            if story and wav_path.exists():
                data = wav_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "public, max-age=31536000, immutable")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()
        else:
            story = _story_load(sid)
            if story:
                self._json_response(story)
            else:
                self.send_response(404)
                self.end_headers()

    def _handle_stories_generate(self, body, rid=None):
        try:
            data = json.loads(body) if body else {}
            # Validar límites: el frontend permite 4-24; un curl puede pedir 9999
            # y colgar el servidor (el LLM escribiría durante horas)
            try:
                lines = int(data.get("lines", 10))
            except (TypeError, ValueError):
                lines = 10
            lines = max(4, min(lines, 24))
            meta = _story_generate(
                lang=data.get("language", "es"),
                theme=data.get("theme", ""),
                level=data.get("level", "A1"),
                lines=lines,
            )
            alexlog.info("stories.generate", "handler",
                         f"Cuento generado: {meta.get('title', '?')}",
                         request_id=rid, lang=data.get("language", "es"),
                         lines=data.get("lines"), audio=meta.get("audio"))
            self._json_response(meta)
        except Exception as e:
            _safe_print(f"[B] [stories] generate error: {e}")
            alexlog.error("stories.generate", "handler", str(e)[:300], request_id=rid)
            self._json_response({"error": str(e)[:300]})

    def _handle_stories_search(self, body, rid=None):
        try:
            data = json.loads(body) if body else {}
            q = data.get("query", "")
            res = _story_search(q)
            alexlog.info("stories.search", "handler",
                         f"Búsqueda '{q[:40]}' -> {len(res)} resultados",
                         request_id=rid, query=q[:120])
            self._json_response({"results": res})
        except Exception as e:
            self._json_response({"error": str(e)[:300]})

    def _handle_stories_delete(self, body, rid=None):
        try:
            parts = self.path.split("/")
            sid = parts[3] if len(parts) > 3 else ""
            if not _story_delete(sid):
                self.send_response(404)
                self.end_headers()
                return
            alexlog.info("stories.delete", "handler", f"Eliminado {sid}",
                         request_id=rid, sid=sid)
            self._json_response({"ok": True})
        except Exception as e:
            self._json_response({"error": str(e)[:300]})

    def _json_response(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _serve_file(self, rel_path):
        fp = FRONTEND_DIR / rel_path
        if fp.exists():
            self.path = f"/{rel_path}"
            super().do_GET()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")

    def log_message(self, fmt, *args):
        msg = fmt % args
        if "/api/stats" not in msg:
            _safe_print(f"[B] {msg}")
