"""alex-teacher — Sanitización y procesamiento de texto para TTS.

Funciones puras (sin estado global): limpieza de texto para síntesis,
división de oraciones, detección de idioma y espaciado natural.
"""

import re

# Regex compilada: caracteres seguros para TTS por idioma
# Base: letras Latin + español (para ES/EN)
# Solo se permiten caracteres pronunciables: letras, dígitos, espacios y
# puntuación de oración. Los símbolos (@ # & * + = / < > ~ ^ \\ | _) se
# eliminan porque el TTS los leería como nombres ("arroba", "numeral").
# Nota: NO usar \x00-\x7F (cubre TODO el ASCII, incluidos los símbolos).
_TTS_SAFE_RE = re.compile(
    r"[^A-Za-z0-9\u00C0-\u024F\u1E00-\u1EFF"
    r"\u0300-\u036F\s.,!?;:'\"¡¿\(\)\[\]\{\}\-–—…€£¥°%$]"
)
# Japonés: Latin + hiragana + katakana + kanji + puntuación JP
_TTS_SAFE_JA_RE = re.compile(
    r"[^A-Za-z0-9\u00C0-\u024F\u1E00-\u1EFF"
    r"\u0300-\u036F\s.,!?;:'\"¡¿\(\)\[\]\{\}\-–—…€£¥°%$"
    r"\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF"  # hiragana + katakana + CJK
    r"\u3000-\u303F\uFF00-\uFFEF"  # CJK punctuation + fullwidth
    r"]"
)

# Regex para thinking tags del LLM (Qwen thinking mode)
_THINKING_RE = re.compile(r'<think>[\s\S]*?</think>', re.DOTALL)

# Simple pattern: strip ALL emojis (Unicode ranges: dingbats, misc symbols, all supplementary)
_SIMPLE_EMOJI_RE = re.compile(
    '['
    '\u2600-\u27BF'    # misc symbols, dingbats
    '\uFE00-\uFE0F'    # variation selectors
    '\U0001F000-\U0001FFFF'  # all supplementary emojis (15K+)
    '\u200D'            # zero-width joiner (ZWJ sequences)
    ']+',
    re.UNICODE,
)

# Common Spanish/English emoji description words that TTS reads aloud as 'carita sonriente'
_EMOJI_WORD_RE = re.compile(
    r'\b(carita[s]?|emojis?|sonrisa[s]?|guiño[s]?|corazón|estrella[s]?|fuego|clap|wave|'
    r'smiley|smile|frown|wink|heart[s]?|star[s]?|fire)\b',
    re.IGNORECASE,
)


def _sanitize_tts_text(text, lang='es'):
    """Elimina todo lo que no debe ser leído en voz alta por TTS.

    Ahora es language-aware:
    - Para japonés: preserva hiragana, katakana y kanji
    - Para español/inglés: elimina CJK (Kokoro no los soporta en ES/EN)

    Pasos:
    1. Eliminar thinking tags (reasoning chain del LLM)
    2. Eliminar emojis Unicode
    3. Eliminar descripciones de emojis ("carita sonriente", "emoji")
    4. Strip caracteres no pronunciables (CJK para ES/EN, Latin para otros)
    5. Normalizar espacios y agregar pausas naturales
    """
    if not text:
        return ""

    # 1. Strip thinking tags (<think>...</think>)
    clean = _THINKING_RE.sub('', text)

    # 2. Strip all emojis
    clean = _SIMPLE_EMOJI_RE.sub('', clean)

    # 3. Strip emoji description words
    clean = _EMOJI_WORD_RE.sub('', clean)

    # 4. Strip non-pronounceable characters (language-aware)
    if lang == 'ja':
        # Para japonés: preservar hiragana, katakana, kanji + Latin
        clean = _TTS_SAFE_JA_RE.sub(' ', clean)
    else:
        # Para español/inglés: eliminar CJK (Kokoro solo soporta Latin)
        clean = _TTS_SAFE_RE.sub(' ', clean)

    # 5. Normalizar espacios y agregar pausas
    clean = _normalize_tts_spacing(clean)

    # 6. Remove orphaned punctuation at start/end
    clean = re.sub(r'^[,;:\s]+', '', clean)
    clean = re.sub(r'[,;:\s]+$', '', clean)

    return clean


def _fix_word_spacing(text):
    """Inserta espacios entre palabras Latin concatenadas.

    El LLM de 4B a veces produce texto sin espacios entre palabras:
      'Sensei,ohayōgozaimasu.' → 'Sensei, ohayō gozaimasu.'
      'SEN-shee(like)+o-ha' → 'SEN-shee (like) + o-ha'

    Estrategia:
    1. Insertar espacio después de .,;:!? antes de letra Latin
    2. Insertar espacio antes de (/[+]= cuando precedido de letra
    3. Insertar espacio después de )/] antes de letra Latin
    4. Colapsar múltiples espacios
    """
    if not text:
        return ""
    # Latin chars: ASCII + Latin-1 Supplement + Latin Extended-A/B (covers macrons: ō, ū)
    _LAT = 'A-Za-zÀ-ÿ\u0100-\u024F'
    # After sentence-ending or comma punctuation, before Latin letter
    clean = re.sub(r'([.!?;:,])([' + _LAT + '])', r'\1 \2', text)
    # Before opening paren/bracket when preceded by letter (no space)
    clean = re.sub(r'([' + _LAT + '])(\\()', r'\\1 \\2', clean)
    clean = re.sub(r'([' + _LAT + '])(\\[)', r'\\1 \\2', clean)
    # After closing paren/bracket, before Latin letter
    clean = re.sub(r'(\))(\w)', r'\1 \2', clean)
    clean = re.sub(r'(\])([' + _LAT + '])', r'\1 \2', clean)
    # Before +, =, / operators when surrounded by letters (not in paths)
    clean = re.sub(r'([' + _LAT + '])([+=])([' + _LAT + '])', r'\1 \2 \3', clean)
    # Collapse multiple spaces
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean


def _normalize_tts_spacing(text):
    """Normaliza texto para TTS: agrega pausas naturales entre oraciones."""
    if not text:
        return ""
    clean = _fix_word_spacing(text)
    return clean


def _split_sentences(text, lang='es'):
    """Divide texto en oraciones para TTS secuencial.

    Kokoro funciona mejor con utterances de 100-200 tokens.
    Textos largos suenan apresurados. Esta función:
    1. Divide por puntuación de fin de oración (.!?)
    2. Para japonés: también divide por 。！？
    3. Filtra segmentos vacíos o muy cortos
    """
    if not text or len(text) < 10:
        return [text] if text else []

    # Split by sentence-ending punctuation
    if lang == 'ja':
        # Japanese sentence endings
        parts = re.split(r'([。！？!?])', text)
    else:
        parts = re.split(r'([.!?])', text)

    # Rebuild sentences: attach punctuation to previous segment
    sentences = []
    current = ''
    for i, part in enumerate(parts):
        if re.match(r'^[.!?。！？]$', part):
            current += part
            sent = current.strip()
            if len(sent) > 3:
                sentences.append(sent)
            current = ''
        else:
            current += part.strip()  # strip leading spaces from next segment
    final = current.strip()
    if final and len(final) > 3:
        sentences.append(final)

    return sentences if sentences else [text]


def _make_silence_pcm(sr, duration_ms=300):
    """Genera silencio PCM (int16) de la duración especificada."""
    import numpy as np
    num_samples = int(sr * duration_ms / 1000)
    return np.zeros(num_samples, dtype=np.int16).tobytes()


# ── Detección de idioma ────────────────────────────────────
def detect_language(text):
    """Detecta es/en/ja/zh/ko en un texto.

    Diferencia correctamente:
    - JA: tiene hiragana (3040-309F) o katakana (30A0-30FF)
    - ZH: solo kanji/hanzi (4E00-9FFF) sin kana
    - KO: hangul (AC00-D7AF)
    - ES: acentos + palabras comunes
    - EN: todo lo demas
    """
    if not text.strip():
        return 'en'

    has_hiragana = any('\u3040' <= c <= '\u309f' for c in text)
    has_katakana = any('\u30a0' <= c <= '\u30ff' for c in text)
    has_kanji = any('\u4e00' <= c <= '\u9fff' for c in text)
    has_hangul = any('\uac00' <= c <= '\ud7af' for c in text)

    # Silabarios japoneses → JA (incluso si mezcla con kanji)
    if has_hiragana or has_katakana:
        return 'ja'

    # Hangul coreano → KO
    if has_hangul:
        return 'ko'

    # Solo kanji/hanzi sin kana → ZH (chino)
    if has_kanji:
        return 'zh'

    # Español: acentos + palabras clave
    es_chars = sum(1 for c in text if '\u00e1' <= c <= '\u00fa' or c in 'ñçüöéèêëàâîôùû¿¡')
    es_words = {'hola','gracias','como','estas','muy','bien','que','el','la','los','las','por','para','con','sin','es','son','del','más','todo','casa','agua','vida','mundo','día','noche','hoy','ayer','mañana','adios','luego','entonces','también','solo','cada','bienvenido','amigo','señor','señora','hablar','tener','hacer','poder','saber','querer','bueno','buena','grande','mejor','siempre','nunca','pronto','tarde','donde','cuando','porque','cual','quien','cuanto','año','semana','mes','lunes','martes','miercoles','jueves','viernes','sabado','domingo','saludos','gracias','muchas','favor','permiso','disculpa','siento','perdon','feliz','contento','cansado','bienvenido','nosotros','ellos','este','esta','ese','esa','aquel','leer','escribir','correr','comer','beber','dormir','vivir','morir','nacer','crecer','pensar','creer','recordar','olvidar','gustar','esperar','viajar','jugar','trabajar','estudiar','enseñar','aprender','entender','conocer','buscar','encontrar','perder','ganar','pagar','vender','comprar','llevar','traer','dejar','entrar','salir','subir','bajar','abrir','cerrar','empezar','terminar','cambiar','mejorar','empeorar','arreglar','romper','limpiar','ensuciar','cocinar','cantar','bailar','pintar','escribir','leer','correr','saltar','nadar','volar','caminar','sentar','parar'}
    words = [w.strip('.,!?;:\'"()[]{}') for w in text.lower().split() if w.strip('.,!?;:\'"()[]{}')]
    if not words:
        return 'en'
    es_count = sum(1 for w in words if w in es_words)
    if es_count > 0 or es_chars > 0:
        return 'es'

    return 'en'
