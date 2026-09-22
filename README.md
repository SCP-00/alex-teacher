<div align="center">

# 🎓 Alex Teacher

**An AI language tutor that teaches with voice — Japanese, Spanish and English.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![llama.cpp](https://img.shields.io/badge/LLM-llama.cpp-8A2BE2)](https://github.com/ggml-org/llama.cpp)
[![Kokoro TTS](https://img.shields.io/badge/TTS-Kokoro%20v1.0-F69220)](https://github.com/thewh1teagle/kokoro-onnx)
[![Whisper](https://img.shields.io/badge/ASR-faster--whisper-40E0D0)](https://github.com/SYSTRAN/faster-whisper)
[![Tests](https://img.shields.io/badge/Tests-66%20%E2%9C%85-brightgreen)](tests/)

</div>

An offline-first AI language tutor: you **speak**, it **listens**, **corrects**, **translates**, and **answers with its own voice**. Everything runs 100% locally — no cloud, no API keys, no data leaves your machine.

**The full speech loop works end-to-end:** microphone → Whisper ASR → Qwen3.5 LLM (with a structured tutor format) → Kokoro TTS → speaker. Japanese output is read with correct Japanese readings (kanji → hiragana conversion before TTS).

## ✨ Features

- 🗣 **Speech-to-speech tutor** — talk to it, it talks back (autoTTS on by default)
- 📖 Structured teaching format: `【TEXT】` · `【TTS_READING】` (romaji) · `【PRONUNCIATION】` · `【TRANSLATION】` · `【EXPLANATION】` · `【EXERCISE】`
- 🈶 **Japanese-aware TTS** — kanji are converted to hiragana readings before synthesis (no more "Chinese-sounding" kanji)
- 📚 **Story mode** — generates illustrated audio stories with multi-voice cast, saved and replayable
- ⚡ Local LLM via llama.cpp — 89 tok/s generation on an RTX 3050 laptop GPU
- 📊 Live system metrics (VRAM, tokens/s) directly in the UI
- 🎨 Minimal Swiss-style interface, dark mode

## 🚀 Quick start

```bash
git clone https://github.com/SCP-00/alex-teacher.git
cd alex-teacher

# 1) Install system deps (llama.cpp, Python venv, Kokoro, Whisper)
./setup.sh          # Linux / macOS

# 2) Start (launches llama-server + the web app, opens your browser)
./run.sh
```

Windows: use `setup.bat` / `run.bat` if present, or follow [Manual setup](#-manual-setup).

Open **http://localhost:3000** and start talking.

## ⚙️ Manual setup

<details>
<summary>Step by step</summary>

**1. Python venv**
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**2. llama.cpp** — [official releases](https://github.com/ggml-org/llama.cpp/releases) or build from source.

**3. Models** (downloaded to `~/.local/share/alex/models/`):
```bash
mkdir -p ~/.local/share/alex/models/llm ~/.local/share/alex/models/onnx

# LLM (Qwen3.5 2B, ~1.4GB)
curl -L -o ~/.local/share/alex/models/llm/Qwen3.5-2B-Q4_K_M.gguf \
  "https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF/resolve/main/Qwen_Qwen3.5-2B-Q4_K_M.gguf"

# Kokoro TTS
curl -L -o ~/.local/share/alex/models/onnx/kokoro-v1.0.onnx \
  "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
curl -L -o ~/.local/share/alex/models/onnx/voices-v1.0.bin \
  "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
```

**4. Run**
```bash
llama-server -m ~/.local/share/alex/models/llm/Qwen3.5-2B-Q4_K_M.gguf \
  --port 8080 -ngl 99 -c 8192 --jinja &

LLAMA_HOST=http://localhost:8080/v1 .venv/bin/python server.py
# → http://localhost:3000
```
</details>

## 🔧 Configuration

| Env var | Default | Description |
|---|---|---|
| `LLAMA_HOST` | `http://localhost:8080/v1` | llama-server endpoint |
| `LLAMA_MODEL` | `Qwen3.5-2B-Q4_K_M` | Model alias sent to llama-server |
| `PLAN_B_PORT` | `3000` | Web server port |
| `ALEX_ENABLE_THINKING` | `0` | `1` = enable Qwen3.5 reasoning mode (slower, better at math) |
| `QWEN3_TTS_MODEL` | `0.6b` | `1.7b` = higher quality voice cloning |

## 🧪 Tests

```bash
python3 -m pytest tests/ -q    # 66 tests
```

## 📜 License

MIT
