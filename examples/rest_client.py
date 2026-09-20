#!/usr/bin/env python3
"""
Example: REST API client for single-turn voice chat.
"""
import asyncio
import base64
import sys
from pathlib import Path

import httpx


async def voice_chat(audio_path: str, url: str = "http://localhost:8000/v1/voice/chat"):
    """Send audio file, get audio response."""

    # Read and encode audio
    audio_bytes = Path(audio_path).read_bytes()
    audio_b64 = base64.b64encode(audio_bytes).decode()

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json={
            "audio_base64": audio_b64,
            "system_prompt": "You are a helpful voice assistant. Keep responses concise.",
        })
        response.raise_for_status()
        result = response.json()

    # Decode and save response audio
    output_audio = base64.b64decode(result["audio_base64"])
    output_path = Path(audio_path).with_suffix(".response.wav")
    output_path.write_bytes(output_audio)

    print(f"✅ Response saved to {output_path}")
    print(f"   Latency: {result['latency_ms']:.0f}ms")
    print(f"   Transcript: {result['transcript']}")
    print(f"   Response: {result['response_text']}")


async def health_check(url: str = "http://localhost:8000/health"):
    """Check server health."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        print(f"Health: {response.json()}")


async def main():
    if len(sys.argv) < 2:
        print("Usage: python rest_client.py <audio_file.wav> [url]")
        print("       python rest_client.py --health [url]")
        return

    url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"

    if sys.argv[1] == "--health":
        await health_check(f"{url}/health")
    else:
        await voice_chat(sys.argv[1], f"{url}/v1/voice/chat")


if __name__ == "__main__":
    asyncio.run(main())
