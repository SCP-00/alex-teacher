"""alex-teacher — Recolector de estadísticas del sistema.

Monitorea CPU/RAM (psutil), GPU (pynvml) y llama-server (llama.cpp) en
segundo plano para exponerlas vía /api/stats. Las métricas de tokens se
acumulan con record_request() desde el handler (usage real de la API
OpenAI-compatible de llama-server).
"""

import json
import threading
import time
import urllib.request

from config import LLAMA_HOST
from util import _safe_print


class StatsCollector:
    def __init__(self):
        self.lock = threading.Lock()
        self.stats = {
            "cpu_percent": 0.0, "ram_used_gb": 0.0, "ram_total_gb": 0.0, "ram_percent": 0.0,
            "vram_used_mb": 0, "vram_total_mb": 0, "vram_percent": 0.0,
            "gpu_percent": 0, "gpu_temp": 0,
            "tokens_per_sec": 0.0, "context_used": 0, "context_max": 4096, "context_percent": 0.0,
            "tokens_generated": 0, "prompt_tokens": 0, "is_processing": False, "n_slots": 0,
            "llama_connected": False, "slots_detail": [],
        }
        self._last_decoded = 0
        self._last_time = time.time()
        self._tok_times = []
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _poll_loop(self):
        while self._running:
            if HAVE_PSUTIL:
                self._collect_system_stats()
            if HAVE_NVML:
                self._collect_gpu_stats()
            self._collect_llama_stats()
            time.sleep(1.8)

    def _collect_system_stats(self):
        try:
            s = {}
            s["cpu_percent"] = round(psutil.cpu_percent(interval=0), 1)
            m = psutil.virtual_memory()
            s["ram_used_gb"] = round(m.used / (1024**3), 1)
            s["ram_total_gb"] = round(m.total / (1024**3), 1)
            s["ram_percent"] = round(m.percent, 1)
            with self.lock:
                self.stats.update(s)
        except Exception:
            pass

    def _collect_gpu_stats(self):
        try:
            s = {}
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            mi = pynvml.nvmlDeviceGetMemoryInfo(h)
            s["vram_used_mb"] = round(mi.used / (1024**2), 0)
            s["vram_total_mb"] = round(mi.total / (1024**2), 0)
            s["vram_percent"] = round(mi.used / mi.total * 100, 1)
            s["gpu_percent"] = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            s["gpu_temp"] = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            with self.lock:
                self.stats.update(s)
        except Exception:
            pass

    def _collect_llama_stats(self):
        """Consulta llama-server (llama.cpp) para conectividad y slots reales.

        llama-server expone /slots con is_processing y n_ctx por slot (métricas
        reales del motor, no estimadas). LLAMA_HOST incluye /v1 (API
        OpenAI-compatible); /slots vive en la raíz, así que quitamos el sufijo.
        """
        base = LLAMA_HOST.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        try:
            req = urllib.request.Request(f"{base}/slots")
            with urllib.request.urlopen(req, timeout=3) as resp:
                slots = json.loads(resp.read().decode())
                is_alive = True
        except Exception:
            slots = []
            is_alive = False

        with self.lock:
            self.stats["llama_connected"] = is_alive
            if not is_alive:
                self.stats["n_slots"] = 0
                self.stats["slots_detail"] = []
                return
            self.stats["n_slots"] = len(slots)
            self.stats["slots_detail"] = [
                {"id": s.get("id"), "is_processing": s.get("is_processing", False)}
                for s in slots
            ]
            self.stats["is_processing"] = any(
                s.get("is_processing", False) for s in slots
            )
            ctx = [s.get("n_ctx") for s in slots if s.get("n_ctx")]
            if ctx:
                self.stats["context_max"] = ctx[0]
            # Decaimiento del tokens_per_sec cuando no hay requests recientes
            if time.time() - self._last_time > 5.0:
                self.stats["tokens_per_sec"] = 0.0

    def record_request(self, prompt_tokens=0, completion_tokens=0, elapsed=0.0):
        """Acumula el usage real de una request LLM (llamado desde el handler)."""
        with self.lock:
            self.stats["prompt_tokens"] += int(prompt_tokens or 0)
            self.stats["tokens_generated"] += int(completion_tokens or 0)
            if completion_tokens and elapsed > 0:
                tps = completion_tokens / elapsed
                self.stats["tokens_per_sec"] = round(tps, 1)
            self._last_time = time.time()

    def get_stats(self):
        with self.lock:
            return dict(self.stats)

    def stop(self):
        self._running = False


# ── Dependencias opcionales ────────────────────────────────
try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False
    _safe_print("[B] [!] psutil no instalado. Stats no disponibles.")

HAVE_NVML = False
try:
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
        name = name.decode('utf-8', errors='replace')
    HAVE_NVML = True
    _safe_print(f"[B] [OK] GPU: {str(name).replace(chr(0),'').strip()}")
except Exception:
    HAVE_NVML = False
    _safe_print("[B] [!] pynvml no disponible. Stats GPU no disponibles.")
