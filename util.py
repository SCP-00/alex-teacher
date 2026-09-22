"""alex-teacher — Utilidades compartidas."""

import alexlog


def _safe_print(msg):
    """Imprime a consola Y emite al log JSONL agent-friendly (evento app.print)."""
    try:
        print(msg)
    except UnicodeEncodeError:
        try:
            print(msg.encode('ascii', errors='replace').decode('ascii'))
        except Exception:
            pass
    # Emisión no invasiva: si alexlog falla, el print ya ocurrió
    try:
        alexlog.info("app.print", "util", str(msg)[:300])
    except Exception:
        pass
