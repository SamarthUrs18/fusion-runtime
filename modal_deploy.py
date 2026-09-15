"""
Modal Deployment for fusion-runtime

Deploy to Modal (free tier) for testing, then scale to RunPod.
"""
import modal

# Modal app
app = modal.App("fusion-runtime")

# GPU-enabled image
image = (
    modal.Image.from_registry("nvidia/cuda:12.4-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libsndfile1", "ffmpeg", "build-essential", "cmake")
    .pip_install(
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.29.0",
        "pydantic>=2.7.0",
        "pydantic-settings>=2.3.0",
        "numpy>=1.26.0",
        "scipy>=1.13.0",
        "soundfile>=0.12.1",
        "websockets>=12.0",
        "aiohttp>=3.9.0",
        "python-multipart>=0.0.9",
        "httpx>=0.27.0",
        "faster-whisper>=1.1.0",
        "llama-cpp-python>=0.2.80",
        "onnxruntime-gpu>=1.18.0",
        "torch>=2.3.0",
        "torchaudio>=2.3.0",
        "huggingface-hub>=0.23.0",
        extra_index_url="https://abetlen.github.io/llama-cpp-python/whl/cu124",
    )
    .run_commands(
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
    )
)

# Model volume (persists across deployments)
model_volume = modal.Volume.from_name("fusion-runtime-models", create_if_missing=True)
MODEL_DIR = "/models"

# Model download function
@app.function(
    image=image,
    volumes={MODEL_DIR: model_volume},
    timeout=1800,
    gpu="T4",  # Free tier GPU
)
def download_models():
    """Download all models to persistent volume."""
    from huggingface_hub import hf_hub_download, snapshot_download
    from pathlib import Path
    
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    
    # Whisper
    from faster_whisper import WhisperModel
    WhisperModel("tiny.en", device="cuda", compute_type="float16", download_root=MODEL_DIR)
    
    # LLM
    hf_hub_download(
        repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
        filename="qwen2.5-7b-instruct-q4_k_m.gguf",
        local_dir=f"{MODEL_DIR}/llm",
    )
    
    # Kokoro
    hf_hub_download(
        repo_id="hexgrad/Kokoro-82M",
        filename="kokoro-v1.0.onnx",
        local_dir=f"{MODEL_DIR}/tts",
    )
    for voice in ["af_heart", "af_bella", "am_michael", "bf_emma", "bm_lewis"]:
        try:
            hf_hub_download(
                repo_id="hexgrad/Kokoro-82M",
                filename=f"voices/{voice}.bin",
                local_dir=f"{MODEL_DIR}/tts",
            )
        except:
            pass
    
    # Silero VAD
    import torch
    torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False, trust_repo=True)
    
    print("✅ All models downloaded to volume")
    return "done"


# Fusion-runtime server
@app.function(
    image=image,
    volumes={MODEL_DIR: model_volume},
    gpu="T4",  # Free tier: T4 (16GB)
    scaledown_window=300,  # Scale to zero after 5 min idle
    max_containers=3,      # Free tier limit
    timeout=300,
    concurrency_limit=10,
)
@modal.asgi_app()
def fusion_runtime_app():
    """ASGI app for Modal."""
    import os
    os.environ["FUSION_MODEL_DIR"] = MODEL_DIR
    os.environ["FUSION_CONFIG"] = "production"
    
    from fusion_runtime.server import app as fastapi_app
    return fastapi_app


# CLI commands
@app.local_entrypoint()
def deploy():
    """Deploy to Modal."""
    print("🚀 Deploying fusion-runtime to Modal...")
    # Models downloaded on first run automatically


@app.local_entrypoint()
def setup_models():
    """Pre-download models to volume."""
    print("📥 Downloading models to Modal volume...")
    download_models.remote()


if __name__ == "__main__":
    # For local testing: modal run modal_deploy.py::setup_models
    # For deploy: modal deploy modal_deploy.py
    pass