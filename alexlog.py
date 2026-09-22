"""alexlog — Sistema de logging agent-friendly para ALEX.

Formato: JSON Lines estricto (1 evento = 1 línea JSON). Un agente (LLM)
puede leer el log sin regex: `for line in f: json.loads(line)`.

Línea de ejemplo:
{"ts":"2026-08-03T01:30:00.123Z","level":"INFO","event":"llm.request",
 "module":"handler","msg":"chat completions","request_id":"a1b2c3",
 "dur_ms":1234,"ram_mb":5120,"vram_mb":2800,
 "data":{"model":"qwen","n_predict":384,"raw":{...}}}

Principios de diseño para agentes:
  1. JSONL estricto — parseable línea a línea, sin heurística.
  2. `raw()` — guarda el payload COMPLETO (respuestas LLM, prompts, salidas
     de pipeline) nunca truncado. Es la fuente de verdad para debugging.
  3. `request_id` — correlaciona todos los eventos de una petición HTTP.
  4. Nombres de evento con namespace: `llm.raw`, `stories.generate`,
     `tts.synth`, `asr.transcribe`, `http.request`.
  5. CLI de consulta: `python3 alexlog.py tail`, `--event X`, `--level Y`,
     `--request <id>` para que el agente extraiga contexto rápidamente.

Uso en código:
    from alexlog import log, raw, request_ctx

    with request_ctx() as rid:
        log("INFO", "stories.generate", "stories", "generando cuento",
            lang="es", lines=10)
        raw("llm.raw", "handler", "respuesta completa del LLM",
            payload={"content": llm_raw}, request_id=rid)
"""

import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────
LOG_DIR = Path(os.environ.get("ALEX_LOG_DIR", Path.home() / ".local/share/alex/logs"))
APP_NAME = os.environ.get("ALEX_APP", "alex")
LOG_FILE = LOG_DIR / f"{APP_NAME}.jsonl"

LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}
_LEVEL_ORDER = ("DEBUG", "INFO", "WARN", "ERROR")

_fh = None  # file handle del log (abierto lazy)


def _open():
    """Abre el fichero de log en modo append (lazy, thread-safe-ish)."""
    global _fh
    if _fh is None:
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            _fh = open(LOG_FILE, "a", encoding="utf-8")
        except Exception:
            _fh = False
    return _fh


def _sys_meta():
    """Metadatos de sistema livianos (RAM/VRAM) — nunca bloquean el request."""
    meta = {}
    try:
        import psutil
        meta["ram_mb"] = int(psutil.Process().memory_info().rss / 1e6)
    except Exception:
        pass
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        meta["vram_mb"] = int(pynvml.nvmlDeviceGetMemoryInfo(h).used / 1e6)
    except Exception:
        pass
    return meta


def _emit(level, event, module, msg, request_id=None, dur_ms=None, data=None):
    """Escribe una línea JSONL. Nunca lanza excepciones en el caller."""
    if level not in LEVELS:
        level = "INFO"
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()) +
             f".{int(time.time() % 1 * 1000):03d}Z",
        "level": level,
        "event": event,
        "module": module or "?",
        "msg": msg,
        "app": APP_NAME,
    }
    if request_id:
        rec["request_id"] = request_id
    if dur_ms is not None:
        rec["dur_ms"] = int(dur_ms)
    rec.update(_sys_meta())
    if data is not None:
        rec["data"] = data
    f = _open()
    if not f:
        return
    try:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        f.flush()
    except Exception:
        pass


# ── API pública ───────────────────────────────────────────────────
def log(level, event, module, msg, request_id=None, dur_ms=None, **data):
    """Registra un evento con metadatos adicionales (keyword args → data)."""
    _emit(level, event, module, msg, request_id=request_id, dur_ms=dur_ms, data=data or None)


def info(event, module, msg, request_id=None, dur_ms=None, **data):
    log("INFO", event, module, msg, request_id, dur_ms, **data)


def warn(event, module, msg, request_id=None, dur_ms=None, **data):
    log("WARN", event, module, msg, request_id, dur_ms, **data)


def error(event, module, msg, request_id=None, dur_ms=None, **data):
    log("ERROR", event, module, msg, request_id, dur_ms, **data)


def raw(event, module, msg, payload, request_id=None):
    """Guarda un payload COMPLETO (salida RAW del LLM/pipeline) sin truncar."""
    _emit("DEBUG", event, module, msg, request_id=request_id,
          data={"raw": payload})


@contextmanager
def request_ctx(event="http.request", module="handler"):
    """Contexto: genera un request_id y registra start/end del request."""
    rid = uuid.uuid4().hex[:10]
    _emit("INFO", event, module, f"{module} request", request_id=rid)
    t0 = time.time()
    try:
        yield rid
    finally:
        _emit("INFO", event, module, f"{module} request done",
              request_id=rid, dur_ms=(time.time() - t0) * 1000)


def tail(n=50, level=None, event=None, request_id=None, path=None):
    """Consulta el log: últimas n líneas filtradas. Devuelve lista de dicts."""
    path = Path(path) if path else LOG_FILE
    if not path.exists():
        return []
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return []
    for line in lines[-max(n * 4, 200):]:  # ventana de lectura amplia
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if level and rec.get("level") != level:
            continue
        if event and rec.get("event") != event:
            continue
        if request_id and rec.get("request_id") != request_id:
            continue
        out.append(rec)
        if len(out) >= n:
            break
    return out[-n:]


def stats(path=None):
    """Resumen: cuenta eventos por nivel y por tipo de evento."""
    path = Path(path) if path else LOG_FILE
    counts = {}
    by_event = {}
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    counts[rec.get("level", "?")] = counts.get(rec.get("level", "?"), 0) + 1
                    e = rec.get("event", "?")
                    by_event[e] = by_event.get(e, 0) + 1
        except Exception:
            pass
    return {"file": str(path), "total": sum(counts.values()),
            "levels": counts, "events": by_event}


# ── CLI de consulta (para agentes) ────────────────────────────────
def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog="alexlog",
        description="Consulta el log JSONL de ALEX (agent-friendly).")
    p.add_argument("--tail", type=int, default=30, help="últimas N líneas")
    p.add_argument("--event", help="filtrar por evento (e.g. llm.raw)")
    p.add_argument("--level", choices=_LEVEL_ORDER, help="filtrar por nivel")
    p.add_argument("--request", help="filtrar por request_id")
    p.add_argument("--path", default=None, help="ruta alternativa del log")
    p.add_argument("--stats", action="store_true", help="resumen de eventos")
    args = p.parse_args(argv)

    if args.stats:
        print(json.dumps(stats(args.path), indent=2, ensure_ascii=False))
        return

    recs = tail(args.tail, level=args.level, event=args.event,
                request_id=args.request, path=args.path)
    for r in recs:
        line = {k: r.get(k) for k in ("ts", "level", "event", "module",
                                      "msg", "request_id", "dur_ms")}
        if "data" in r:
            line["data"] = r["data"]
        print(json.dumps(line, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
