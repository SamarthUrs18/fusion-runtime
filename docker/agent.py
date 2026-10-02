"""The GPU image's agent, used when you don't mount your own over /app/agent.py.

Speech runs on the GPU in this container; the LLM runs on the vLLM container beside
it (docker/docker-compose.gpu.yml), which fusion reaches at http://vllm:8002/v1.
Without that container, frun up says it can't reach the server rather than falling
back to a slow model on the CPU.

To change the model, edit the vllm service's --model in the compose file and the
LLM line here to match. FUSION_LLM_URL moves the server without editing this file.
"""
from fusion_runtime import LLM, STT, TTS, Agent

agent = Agent(
    name="default",
    prompt="You are a helpful voice assistant. Keep answers to one or two short sentences.",
    greeting="Hi! How can I help?",
    stt=STT("whisper-small"),
    llm=LLM("vllm:hf:Qwen/Qwen2.5-7B-Instruct-AWQ", url="http://vllm:8002/v1", max_tokens=200),
    tts=TTS("kokoro-v1.0", voice="af_heart"),
    profile="production",
)
