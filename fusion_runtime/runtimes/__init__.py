"""Model runtimes: one adapter per engine, file format or protocol, never per model.

    llama_cpp     any GGUF LLM (chat template read from the file)
    ctranslate2   any Whisper-family STT model
    onnx          ONNX TTS, with a small family spec per model family (Kokoro)
    openai_http   any OpenAI-compatible chat endpoint (vLLM, llama-server, hosted APIs)

Each implements the contract in fusion_runtime.contract and passes
fusion_runtime.testing.conformance. The resolver picks one from the model
reference; the registry imports it by name.
"""
