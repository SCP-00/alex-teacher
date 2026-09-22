"""alex-teacher — Estado compartido del servidor.

Singleton holders que `server.py` reasigna durante el arranque y que los
handlers leen en cada request. Centralizar aquí el estado mutable evita
`global` repartidos por todos los módulos.
"""

from llm import ResponseCache

# Caché LRU de respuestas del LLM (creado al importar)
cache = ResponseCache(maxsize=50)

# Se asignan en server.main():
stats = None   # StatsCollector
piper = None   # PiperTTS
