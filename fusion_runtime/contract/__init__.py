"""The runtime contract: what every STT, LLM and TTS runtime implements.

Runtimes are written per runtime or protocol (llama.cpp, CTranslate2, ONNX,
OpenAI-compatible HTTP), never per model. Check a runtime with
`fusion_runtime.testing.conformance`.
"""
from fusion_runtime.contract.common import (
    RUNTIME_STAGES,
    STAGES,
    AdapterError,
    AuthFailed,
    Cancelled,
    CancelToken,
    Capabilities,
    Health,
    InvalidRequest,
    ModelNotFound,
    ModelRuntime,
    ModelSpec,
    Overloaded,
    RateLimited,
    Request,
    RuntimeFailure,
    Stage,
    UnsupportedModel,
)
from fusion_runtime.contract.llm import LLMChunk, LLMRequest, LLMRuntime, Message, ToolCall, ToolSpec
from fusion_runtime.contract.stt import STTRequest, STTResult, STTRuntime, Transcript
from fusion_runtime.contract.tts import AudioChunk, TTSRequest, TTSRuntime
from fusion_runtime.contract.turn import TurnDetector, TurnPrediction, TurnRequest
