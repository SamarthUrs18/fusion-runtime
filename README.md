# fusion-runtime

**Low-latency voice AI inference runtime — STT → LLM → TTS in <500ms**

Self-hosted, model-agnostic, production-ready. Built for real-time voice applications.

## 🎯 Why fusion-runtime?

STT, LLM and TTS run co-located in a single worker: zero network hops between models, shared GPU memory, a true streaming pipeline, and dynamic batching.

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Single Worker Container                   │
│  ┌─────────────┐    <5ms    ┌─────────────┐   Shared GPU  ┌─────────────┐
│  │   STT       │ ──────────► │   LLM       │ ◄───────────► │   TTS       │
│  │ faster-     │  (queues)   │  llama.cpp  │   Memory      │  Kokoro     │
│  │ whisper     │             │  Qwen2.5-7B │  (KV cache +  │  ONNX       │
│  └─────────────┘             └─────────────┘  audio tokens)└─────────────┘
│         │                            │                         │
│         └────────────────────────────┼────────────────────────┘
│                                      ▼
│                         ┌─────────────────────┐
│                         │   Orchestrator      │
│                         │  • Latency budget   │
│                         │  • Streaming coord  │
│                         │  • Dynamic batching │
│                         └─────────────────────┘
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                 ┌──────────────────────┐
                 │   Voice clients      │
                 │   (WebSocket)        │
                 └──────────────────────┘
```

## 🚀 Quickstart

### Prerequisites
- NVIDIA GPU (CUDA 12.4+) for production
- Docker + NVIDIA Container Toolkit
- 8GB+ VRAM recommended

### 1. Clone & Download Models
```bash
git clone https://github.com/fusion-runtime/fusion-runtime.git
cd fusion-runtime

# Download models (one-time)
python scripts/download_models.py --all
```

### 2. Run with Docker Compose
```bash
cd docker
docker-compose up -d
```

### 3. Test
```bash
# Health check
curl http://localhost:8000/health

# Single-turn voice chat
curl -X POST http://localhost:8000/v1/voice/chat \
  -H "Content-Type: application/json" \
  -d '{"audio_base64": "'$(base64 -w0 test_audio.wav)'"}'

# WebSocket (real-time)
# See examples/websocket_client.py
```

## ⚙️ Configuration

### Default (Production - All Self-Hosted)
```python
PipelineConfig(
    stt=STTConfig(provider="faster_whisper", model="tiny.en", device="cuda"),
    llm=LLMConfig(provider="llama_cpp", model="Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
    tts=TTSConfig(provider="kokoro", model="kokoro-v1.0.onnx"),
    target_latency_ms=500,
)
```

### Swappable LLM
```python
LLMConfig(provider="llama_cpp", model="llm/your-model.gguf")                 # Local GGUF
LLMConfig(provider="openai", model="...", api_base="http://localhost:8080/v1")  # Any OpenAI-compatible server
```

## 📦 Model Support

| Component | Engine | Format |
|-----------|--------|--------|
| **STT** | faster-whisper (tiny.en default) | CTranslate2 |
| **LLM** | llama.cpp (Qwen2.5 default), or any OpenAI-compatible endpoint | GGUF |
| **TTS** | Kokoro | ONNX |
| **VAD** | Silero | — |
| **Turn Detection** | Punctuation + silence | — |


## 📊 Performance Targets

| Metric | Target (P50) | Target (P99) |
|--------|-------------|-------------|
| Time to First Audio | 200ms | 350ms |
| End-to-End Latency | 300ms | 500ms |
| Concurrent Users/GPU | 15-25 | - |
| GPU Memory | 8-10 GB | - |

## 🔧 Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run CPU-only (no GPU needed)
docker-compose -f docker/docker-compose.yml up fusion-runtime-cpu

# Run tests
pytest tests/

# Lint
ruff check fusion_runtime/
mypy fusion_runtime/
```

## 📁 Project Structure

```
fusion-runtime/
├── fusion_runtime/
│   ├── config.py        # Settings, profiles, model directory
│   ├── server.py        # FastAPI + WebSocket server
│   ├── stt/             # base.py + one file per engine (whisper.py)
│   ├── llm/             # base.py, llama_cpp.py, openai_compat.py
│   ├── tts/             # base.py, kokoro.py
│   ├── vad/             # base.py, silero.py, turn.py (turn detection)
│   ├── engine/          # orchestrator.py (conversation loop), barge_in.py, metrics.py
│   └── audio/           # echo_canceller.py, duplex_audio.py
├── docker/
│   ├── Dockerfile       # CUDA production image
│   ├── Dockerfile.cpu   # CPU-only dev image
│   ├── docker-compose.yml
│   └── requirements.cpu.txt
├── scripts/
│   └── download_models.py
├── tests/
├── pyproject.toml
└── README.md
```

## 🎯 Roadmap

- [ ] Modal deployment template (free tier → RunPod)
- [ ] Python SDK (`pip install fusion-runtime-sdk`)
- [ ] TypeScript SDK for frontend integration
- [ ] Prometheus metrics + Grafana dashboards
- [ ] Multi-language STT/TTS routing
- [ ] Custom model fine-tuning pipeline

## 📄 License

**AGPL-3.0-or-later** (core runtime)  
**Apache-2.0** (SDKs, client libraries)

Commercial licenses available — contact for enterprise.

---

**Built for developers who need voice AI that actually feels real-time.**