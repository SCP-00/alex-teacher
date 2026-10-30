<div align="center">

# 🎓 Alex Teacher

**An AI language tutor that teaches with voice — Japanese, Spanish and English.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![llama.cpp](https://img.shields.io/badge/LLM-llama.cpp-8A2BE2)](https://github.com/ggml-org/llama.cpp)
[![Kokoro TTS](https://img.shields.io/badge/TTS-Kokoro%20v1.0-F69220)](https://github.com/thewh1teagle/kokoro-onnx)
[![Whisper](https://img.shields.io/badge/ASR-faster--whisper-40E0D0)](https://github.com/SYSTRAN/faster-whisper)
[![Tests](https://img.shields.io/badge/Tests-66%20%E2%9C%85-brightgreen)](tests/)

</div>

You **speak**, it **listens**, **corrects**, **translates**, and **answers with its own voice**. Everything runs locally: no cloud, no API keys, no audio leaving the machine.

The interesting part is not that a model can answer a question. It is that the answer comes back in a **fixed teaching format** the interface can parse and turn into something you can practise: the text, a reading, a pronunciation guide, a translation, an explanation and an exercise. A free-form LLM reply cannot do that, so most of the work here is in the prompts and the sanitising between model and screen.

## My role in this project

The code is written with AI coding agents; the product decisions and the quality bar are mine:

- **Designed the response contract.** Six tagged sections per turn. The tutor is only useful if the interface can rely on the shape of the reply, so the format is enforced in the prompt and validated before rendering.
- **Solved the Japanese problem.** Kanji read by an English-trained TTS model sound wrong, so output goes through kanji → hiragana conversion before synthesis.
- **Defined the story mode**: illustrated audio stories with a multi-voice cast, generated from the same structured output and saved for replay.
- **Kept it offline by design.** No API keys, no accounts, no telemetry. If it needs the network to teach you a sentence, it was built wrong.
- **Criterion:** 66 tests covering config, prompt structure, sanitising, stories and utilities.

## How a turn works

```
microphone
   │
   ▼
faster-whisper (ASR)          speech → text
   │
   ▼
prompts.py                    builds the tutor prompt + the six-section contract
   │
   ▼
Qwen3.5-2B via llama.cpp      ~89 tok/s on an RTX 3050 laptop GPU
   │
   ▼
sanitize + parse              strip thinking, validate sections, kanji → hiragana
   │
   ▼
Kokoro TTS                    text → audio, correct readings for Japanese
   │
   ▼
browser                       text, reading, translation, explanation, exercise
```

## Features

- 🗣 **Speech-to-speech tutor** — talk to it, it talks back (autoTTS on by default).
- 📖 **Structured teaching format** — `【TEXT】` · `【TTS_READING】` · `【PRONUNCIATION】` · `【TRANSLATION】` · `【EXPLANATION】` · `【EXERCISE】`.
- 🈶 **Japanese-aware TTS** — kanji converted to hiragana before synthesis.
- 📚 **Story mode** — illustrated audio stories with a multi-voice cast, saved and replayable.
- ⚡ **Local LLM** — Qwen3.5-2B through llama.cpp.
- 📊 **Live metrics** — VRAM and tokens/s in the UI.
- 🎨 Minimal Swiss-style interface with dark mode.

## Stack

`Python` `Flask` `llama.cpp` `Qwen3.5-2B (GGUF)` `faster-whisper` `Kokoro TTS` `HTML/CSS/JS`

## Quick start

```bash
git clone https://github.com/SCP-00/alex-teacher.git
cd alex-teacher
./setup.sh    # instala llama.cpp, venv de Python, Kokoro y Whisper (Linux/macOS)
./run.sh      # arranca llama-server + la app y abre el navegador
```

En Windows, sigue la instalación manual de abajo (los scripts `setup.bat` / `run.bat` no siempre están al día).

Abre **http://localhost:3000** y empieza a hablar.

## Manual setup

<details>
<summary>Paso a paso</summary>

**1. Entorno de Python**

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**2. llama.cpp** — [releases oficiales](https://github.com/ggml-org/llama.cpp/releases) o compilado desde fuente.

**3. Modelos** (se descargan a `~/.local/share/alex/models/`):

```bash
mkdir -p ~/.local/share/alex/models/llm ~/.local/share/alex/models/onnx

# LLM (Qwen3.5 2B, ~1.4 GB)
curl -L -o ~/.local/share/alex/models/llm/Qwen3.5-2B-Q4_K_M.gguf \
"https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF/resolve/main/Qwen_Qwen3.5-2B-Q4_K_M.gguf"

# Kokoro TTS
curl -L -o ~/.local/share/alex/models/onnx/kokoro-v1.0.onnx \
"https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
curl -L -o ~/.local/share/alex/models/onnx/voices-v1.0.bin \
"https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
```

**4. Arrancar**

```bash
llama-server -m ~/.local/share/alex/models/llm/Qwen3.5-2B-Q4_K_M.gguf \
  --port 8080 -ngl 99 -c 8192 --jinja &

LLAMA_HOST=http://localhost:8080/v1 .venv/bin/python server.py
# → http://localhost:3000
```

</details>

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLAMA_HOST` | `http://localhost:8080/v1` | llama-server endpoint |
| `LLAMA_MODEL` | `Qwen3.5-2B-Q4_K_M` | Model alias sent to llama-server |
| `PLAN_B_PORT` | `3000` | Web server port |
| `ALEX_ENABLE_THINKING` | `0` | `1` = enable Qwen3.5 reasoning mode (slower, better at maths) |
| `QWEN3_TTS_MODEL` | `0.6b` | `1.7b` = higher quality voice cloning |

## Tests

```bash
python3 -m pytest tests/ -q    # 66 tests
```

## What it needs to run

A machine that can run llama.cpp with a 2B model and Whisper at the same time. The throughput in the badges was measured on a laptop with an RTX 3050; on CPU it still works, it is just slower. Without a GPU the first response can take several seconds.

## License

MIT
