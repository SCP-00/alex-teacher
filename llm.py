"""alex-teacher — LLM: caché LRU y completado contra llama-server."""

import json
import os
import re
import threading
import urllib.request
from collections import OrderedDict

from config import LLAMA_MODEL, NUM_CTX, LLAMA_HOST


class ResponseCache:
    """LRU cache thread-safe para respuestas del LLM."""

    def __init__(self, maxsize=50):
        self._cache = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, messages):
        key_parts = []
        for msg in messages[-4:]:
            role = msg.get('role', '')
            content = msg.get('content', '')[:200]
            key_parts.append(f"{role}:{content}")
        return hash('|'.join(key_parts))

    def get(self, messages):
        key = self._make_key(messages)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, messages, response):
        if not response or len(response) < 20:
            return
        key = self._make_key(messages)
        with self._lock:
            self._cache[key] = response
            self._cache.move_to_end(key)
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def invalidate(self):
        with self._lock:
            self._cache.clear()

    def stats(self):
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 1),
            }


def _llm_complete(messages, n_predict=800, temperature=0.7):
    """Completado no-streaming contra llama-server (reutiliza LLAMA_HOST)."""
    data = {
        "model": LLAMA_MODEL,
        "messages": messages,
        "n_predict": n_predict,
        "temperature": temperature,
        "stream": False,
        "options": {"num_ctx": NUM_CTX},
        "keep_alive": "2m",
        "reasoning_format": "none",
        # Qwen3.5 razona por defecto: sin esto agota n_predict en <think> (QA 2026-09-22)
        "chat_template_kwargs": {"enable_thinking": os.environ.get("ALEX_ENABLE_THINKING", "0") == "1"},
    }
    req = urllib.request.Request(
        f"{LLAMA_HOST}/chat/completions",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=240) as resp:
        result = json.loads(resp.read().decode())
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    # Limpiar razonamiento (puede quedar sin cerrar si el modelo se cortó)
    content = re.sub(r"<think>[\s\S]*?</think>", "", content)
    if "<think>" in content:
        content = content.split("<think>", 1)[0]
    return content.strip()
