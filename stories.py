"""alex-teacher — Cuentos (Stories).

El LLM escribe la historia; el TTS le pone voz a cada personaje
(Kokoro seeds + Qwen3-TTS con clonación para voces 100% únicas);
el audio + embedding se cachean para reproducir al instante sin regenerar.
"""

import json
import os
import re
import time

from config import (
    STORY_DIR, STORY_INDEX, STORY_CAST, STORY_CAST_IDS, STORY_CAST_HINT,
    KOKORO_CONFIG,
)
from llm import _llm_complete
from tts import _get_kokoro_instance
from util import _safe_print
from qwen3tts import _qwen3_generate_refs

_STORY_EMBEDDER = None


def _get_story_embedder():
    """Embeddings multilingües (fastembed, CPU) para búsqueda semántica."""
    global _STORY_EMBEDDER
    if _STORY_EMBEDDER is None:
        try:
            from fastembed import TextEmbedding
            _STORY_EMBEDDER = TextEmbedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            _safe_print("[B] [stories] Embedder multilingüe listo")
        except Exception as e:
            _safe_print(f"[B] [stories] Embedder no disponible: {e}")
            _STORY_EMBEDDER = False
    return _STORY_EMBEDDER or None


def _story_embed(text):
    """Vector semántico del cuento (None si no hay embedder)."""
    emb = _get_story_embedder()
    if emb is None or not text:
        return None
    try:
        return [float(x) for x in next(emb.embed([text[:800]]))]
    except Exception:
        return None


def _story_synth_line(text, cast_id, lang):
    """Sintetiza una línea con la voz del personaje. Retorna (sr, np_audio) o (None, None).

    Prioridad: Qwen3-TTS (voz clonada única por personaje) -> Kokoro (fallback).
    """
    if not text:
        return None, None
    # 1) Qwen3-TTS con clonación de voz (si el modelo está descargado)
    try:
        from qwen3tts import _qwen3_synth_line
        sr, audio = _qwen3_synth_line(text, cast_id, lang)
        if sr is not None and audio is not None:
            return sr, audio
    except ImportError:
        pass
    except Exception as e:
        _safe_print(f"[B] [stories] Qwen3-TTS error ({cast_id}/{lang}): {e}")
    # 2) Fallback: Kokoro ONNX (CPU, 0 VRAM)
    k = _get_kokoro_instance()
    if k is None:
        return None, None
    voice = None
    for c in STORY_CAST:
        if c["id"] == cast_id:
            voice = c["voices"].get(lang)
            break
    if not voice:
        voice = KOKORO_CONFIG.get(lang, KOKORO_CONFIG["es"])["voice"]
    cfg = KOKORO_CONFIG.get(lang, KOKORO_CONFIG["es"])
    try:
        import numpy as np
        audio, sr = k.create(text, voice=voice, speed=cfg.get("speed", 1.0), lang=cfg["lang"])
        if audio is None or len(audio) == 0:
            return None, None
        return sr, np.asarray(audio, dtype=np.float32)
    except Exception as e:
        _safe_print(f"[B] [stories] TTS error ({cast_id}/{lang}): {e}")
        return None, None


def _story_render_wav(lines, lang, gap_ms=350):
    """Concatena las líneas con pausas y devuelve bytes WAV."""
    import io
    import wave

    import numpy as np
    sr = None
    chunks = []
    for line in lines:
        speaker = line.get("speaker", "narrator")
        text = line.get("text", "").strip()
        if not text:
            continue
        s, audio = _story_synth_line(text, speaker, lang)
        if audio is None:
            continue
        if sr is None and s is not None:
            sr = s
        chunks.append(np.clip(audio, -1.0, 1.0).astype(np.float32))
    if not chunks:
        return None
    sr = sr or 24000
    gap = np.zeros(int(sr * gap_ms / 1000), dtype=np.float32)
    parts = []
    for c in chunks:
        parts.append(c)
        parts.append(gap)
    all_audio = np.concatenate(parts)
    pcm = (all_audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _story_save(sid, story, wav_bytes, embedding):
    """Persiste cuento + audio + embedding en ~/.local/share/alex/data/stories/<sid>/"""
    story_dir = STORY_DIR / sid
    story_dir.mkdir(parents=True, exist_ok=True)
    (story_dir / "story.json").write_text(json.dumps(story, ensure_ascii=False, indent=2), encoding="utf-8")
    if wav_bytes:
        (story_dir / "audio.wav").write_bytes(wav_bytes)
    (story_dir / "embedding.json").write_text(json.dumps(embedding or []))
    meta = {
        "id": sid,
        "title": story.get("title", "Sin título"),
        "language": story.get("language", ""),
        "theme": story.get("theme", ""),
        "level": story.get("level", ""),
        "created": story.get("created", ""),
        "characters": story.get("characters", []),
        "audio": wav_bytes is not None,
    }
    index = [i for i in _story_load_index() if i["id"] != sid]
    index.append(meta)
    index.sort(key=lambda i: i.get("created", ""), reverse=True)
    STORY_INDEX.parent.mkdir(parents=True, exist_ok=True)
    STORY_INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _story_load_index():
    if not STORY_INDEX.exists():
        return []
    try:
        return json.loads(STORY_INDEX.read_text(encoding="utf-8"))
    except Exception:
        return []


def _safe_story_id(sid):
    """Valida un id de cuento: solo alfanuméricos, guiones y guion bajo (evita path traversal)."""
    return bool(sid) and re.fullmatch(r"[A-Za-z0-9_-]+", sid) is not None


def _story_load(sid):
    if not _safe_story_id(sid):
        return None
    d = STORY_DIR / sid
    if not (d / "story.json").exists():
        return None
    story = json.loads((d / "story.json").read_text(encoding="utf-8"))
    story["has_audio"] = (d / "audio.wav").exists()
    return story


def _story_delete(sid):
    if not _safe_story_id(sid):
        return False
    d = STORY_DIR / sid
    if not d.exists():
        return False
    import shutil
    shutil.rmtree(d)
    index = [i for i in _story_load_index() if i["id"] != sid]
    STORY_INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def _extract_json(text):
    """Extrae el primer objeto JSON válido de un texto (robusto ante prosa).

    Maneja prosa ANTES y DESPUÉS del objeto JSON: recorta desde el primer
    "{" y prueba cada cierre "}" hasta encontrar un objeto parseable.
    """
    try:
        return json.loads(text)
    except Exception:
        pass
    idx = text.find("{")
    while idx != -1:
        end = text.find("}", idx)
        while end != -1:
            candidate = text[idx:end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                pass
            end = text.find("}", end + 1)
        idx = text.find("{", idx + 1)
    raise ValueError(f"JSON no encontrado en: {text[:200]}")


def _story_generate(lang="es", theme="", level="A1", lines=10):
    """Genera un cuento: LLM escribe, TTS narra por personaje, se cachea audio + embedding."""
    lang_names = {"es": "Spanish", "ja": "Japanese", "en": "English"}
    lang_name = lang_names.get(lang, lang)
    system = (
        "You are a children's story writer and voice director for an offline AI language-learning app. "
        "You write short, engaging stories for language learners.\n"
        f"Write the story in {lang_name} at CEFR level {level or 'A1'} (simple vocabulary, short sentences).\n"
        f"Available characters (use ONLY these ids as 'speaker'): {STORY_CAST_HINT}.\n"
        "Rule 1: the narrator sets the scene first.\n"
        "Rule 2: dialogue lines use the character ids above; the narrator never speaks dialogue.\n"
        "Rule 3: keep each line to a maximum of two short sentences.\n"
        "Respond with ONLY valid JSON, no markdown, no code fences, exactly this schema:\n"
        '{"title":"...","theme":"...","language":"' + lang + '","level":"' + level + '",'
        '"characters":[{"id":"zorro","name":"Zorro","role":"protagonist"}],'
        '"lines":[{"speaker":"narrator","text":"..."},{"speaker":"zorro","text":"..."}]}'
    )
    user = f"Write a {lines}-line story about: {theme or 'a curious animal discovering a new place'}."
    _safe_print(f"[B] [stories] LLM escribiendo cuento ({lang}/{level}/{lines} líneas)...")
    raw = _llm_complete(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        n_predict=min(max(int(lines) * 120, 600), 3000),
        temperature=0.8,
    )
    story = _extract_json(raw)
    if isinstance(story, str):
        try:
            story = json.loads(story)
        except Exception:
            pass
    if not isinstance(story, dict):
        raise ValueError(f"El LLM no devolvió un objeto JSON: {raw[:200]}")
    story.setdefault("language", lang)
    story.setdefault("theme", theme or "general")
    story.setdefault("level", level or "A1")
    story.setdefault("characters", [])
    story.setdefault("lines", [])
    if not story.get("title"):
        story["title"] = f"{theme or 'Cuento'} · {lang.upper()}"
    # Normalizar líneas: aceptar strings o dicts (el modelo a veces simplifica)
    _lines = []
    for ln in story.get("lines", []):
        if isinstance(ln, str):
            _lines.append({"speaker": "narrator", "text": ln.strip()})
        elif isinstance(ln, dict):
            sp = str(ln.get("speaker", "narrator") or "narrator")
            if sp not in STORY_CAST_IDS:
                sp = "narrator"
            _lines.append({"speaker": sp, "text": str(ln.get("text", "")).strip()})
    story["lines"] = [l for l in _lines if l["text"]]
    # Normalizar personajes
    _chars = []
    for c in story.get("characters", []):
        if isinstance(c, str):
            _chars.append({"id": c, "name": c, "role": ""})
        elif isinstance(c, dict):
            _chars.append(c)
    story["characters"] = _chars
    sid = time.strftime("%Y%m%d-%H%M%S") + "-" + os.urandom(3).hex()
    story["id"] = sid
    story["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _safe_print(f"[B] [stories] Generando audio ({len(story['lines'])} líneas)...")
    # Asegura los clips de referencia de voz por personaje (una sola vez)
    try:
        _qwen3_generate_refs()
    except Exception as e:
        _safe_print(f"[B] [stories] WARN refs de voz: {e}")
    wav = _story_render_wav(story["lines"], lang)
    if wav is None:
        _safe_print("[B] [stories] WARN: sin audio (¿Kokoro disponible?)")
    embed_text = story.get("title", "") + ". " + " ".join(l.get("text", "") for l in story["lines"][:4])
    embedding = _story_embed(embed_text)
    meta = _story_save(sid, story, wav, embedding)
    meta["lines"] = len(story["lines"])
    meta["text"] = story
    return meta


def _story_search(query, k=5):
    """Búsqueda semántica por similitud coseno sobre los embeddings guardados."""
    import math
    q_emb = _story_embed(query)
    if q_emb is None:
        return []
    results = []
    for meta in _story_load_index():
        sid = meta["id"]
        if not _safe_story_id(sid):
            continue
        emb_path = STORY_DIR / sid / "embedding.json"
        if not emb_path.exists():
            continue
        try:
            e = json.loads(emb_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not e or len(e) != len(q_emb):
            continue
        dot = sum(a * b for a, b in zip(q_emb, e))
        na = math.sqrt(sum(x * x for x in q_emb))
        nb = math.sqrt(sum(x * x for x in e))
        sim = dot / (na * nb) if na and nb else 0.0
        results.append((sim, meta))
    results.sort(key=lambda t: -t[0])
    return [m for s, m in results[:k]]
