#!/usr/bin/env python3
"""
Model Download Script

Downloads all required models for fusion-runtime.
Run once during container build or manual setup.
"""
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).parent.parent / "models"))

def download_whisper():
    """Download faster-whisper tiny.en model."""
    print("📥 Downloading faster-whisper tiny.en...")
    # faster-whisper downloads automatically on first use
    # This just triggers the download
    from faster_whisper import WhisperModel
    model = WhisperModel("tiny.en", device="cpu", compute_type="int8", download_root=str(MODEL_DIR))
    print("✅ faster-whisper ready")

def download_llama_cpp():
    """Download Qwen2.5-7B-Instruct GGUF."""
    print("📥 Downloading Qwen2.5-7B-Instruct-Q4_K_M...")
    model_path = hf_hub_download(
        repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
        filename="qwen2.5-7b-instruct-q4_k_m.gguf",
        local_dir=MODEL_DIR / "llm",
        local_dir_use_symlinks=False,
    )
    print(f"✅ LLM downloaded to {model_path}")

def download_kokoro():
    """Download Kokoro ONNX model and voices."""
    print("📥 Downloading Kokoro v1.0 ONNX...")
    repo = "onnx-community/Kokoro-82M-v1.0-ONNX"
    
    # Model (quantized uint8 = smaller/faster on CPU; use model.onnx for GPU)
    model_file = os.getenv("KOKORO_ONNX_FILE", "onnx/model.onnx")
    model_path = hf_hub_download(
        repo_id=repo,
        filename=model_file,
        local_dir=MODEL_DIR / "tts",
    )
    
    # Also grab the tokenizer (Kokoro needs it for phoneme -> token IDs)
    hf_hub_download(
        repo_id=repo,
        filename="tokenizer.json",
        local_dir=MODEL_DIR / "tts",
    )
    
    # Voices (.bin format for the ONNX port)
    voices = ["af_heart", "af_bella", "am_michael", "bf_emma", "bm_lewis"]
    for voice in voices:
        try:
            hf_hub_download(
                repo_id=repo,
                filename=f"voices/{voice}.bin",
                local_dir=MODEL_DIR / "tts",
            )
        except Exception:
            pass  # Voice file might not exist
    
    # Build the kokoro-onnx voice pack (npz: {voice_name: (510,256) style vectors})
    # kokoro-onnx expects one voices-v1.0.bin file; assemble it from per-voice bins.
    import numpy as np
    tts_dir = MODEL_DIR / "tts"
    pack = {}
    for vf in sorted((tts_dir / "voices").glob("*.bin")):
        arr = np.fromfile(vf, dtype=np.float32)
        if arr.size == 510 * 256:  # valid Kokoro style pack for this voice
            pack[vf.stem] = arr.reshape(510, 256)
    if pack:
        out = tts_dir / "voices-v1.0.bin"
        with open(out, "wb") as f:
            np.savez(f, **pack)  # open file object → no .npz extension appended
        print(f"✅ Voice pack built: {out} ({len(pack)} voices)")
    
    print(f"✅ Kokoro downloaded to {model_path}")

def download_silero_vad():
    """Pre-download Silero VAD model."""
    print("📥 Downloading Silero VAD...")
    import torch
    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        trust_repo=True,
        source='github',
    )
    print("✅ Silero VAD ready")

def download_fire_red_asr():
    """Download FireRedASR models (optional)."""
    print("📥 Downloading FireRedASR...")
    try:
        snapshot_download(
            repo_id="FireRedTeam/FireRedASR-AED-L",
            local_dir=MODEL_DIR / "fireredasr" / "FireRedASR-AED-L",
            local_dir_use_symlinks=False,
        )
        snapshot_download(
            repo_id="FireRedTeam/FireRedChat-punc",
            local_dir=MODEL_DIR / "fireredasr" / "PUNC-BERT",
            local_dir_use_symlinks=False,
        )
        print("✅ FireRedASR downloaded")
    except Exception as e:
        print(f"⚠️ FireRedASR download failed (optional): {e}")

def download_fire_red_tts():
    """Download FireRedTTS (optional, non-commercial)."""
    print("📥 Downloading FireRedTTS...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="FireRedTeam/FireRedTTS-1S",
            revision="fireredtts1s_4_chat",
            local_dir=MODEL_DIR / "fireredtts",
            local_dir_use_symlinks=False,
        )
        print("✅ FireRedTTS downloaded (NON-COMMERCIAL ONLY)")
    except Exception as e:
        print(f"⚠️ FireRedTTS download failed (optional): {e}")

def download_fire_red_eot():
    """Download FireRedChat EoT turn detector (optional)."""
    print("📥 Downloading FireRedChat EoT...")
    try:
        snapshot_download(
            repo_id="FireRedTeam/FireRedChat-EoT",
            local_dir=MODEL_DIR / "firered_eot",
            local_dir_use_symlinks=False,
        )
        print("✅ FireRedChat EoT downloaded")
    except Exception as e:
        print(f"⚠️ FireRedChat EoT download failed (optional): {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download fusion-runtime models")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--whisper", action="store_true", help="Download Whisper")
    parser.add_argument("--llm", action="store_true", help="Download LLM")
    parser.add_argument("--kokoro", action="store_true", help="Download Kokoro TTS")
    parser.add_argument("--vad", action="store_true", help="Download Silero VAD")
    parser.add_argument("--firered-asr", action="store_true", help="Download FireRedASR")
    parser.add_argument("--firered-tts", action="store_true", help="Download FireRedTTS (non-commercial)")
    parser.add_argument("--firered-eot", action="store_true", help="Download FireRedChat EoT")
    
    args = parser.parse_args()
    
    # Default to core models if none specified
    if not any(vars(args).values()):
        args.whisper = args.llm = args.kokoro = args.vad = True
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.whisper:
        download_whisper()
    if args.llm:
        download_llama_cpp()
    if args.kokoro:
        download_kokoro()
    if args.vad:
        download_silero_vad()
    if args.firered_asr:
        download_fire_red_asr()
    if args.firered_tts:
        download_fire_red_tts()
    if args.firered_eot:
        download_fire_red_eot()
    
    print("\n🎉 All requested models downloaded!")
    print(f"📁 Model directory: {MODEL_DIR}")

if __name__ == "__main__":
    main()