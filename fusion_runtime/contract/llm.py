"""LLM runtime contract."""
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple

from fusion_runtime.contract.common import ModelRuntime, Request


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None  # for role="tool": which call this answers


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema object


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON text, as produced by the model


@dataclass(kw_only=True)
class LLMRequest(Request):
    messages: Sequence[Message]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Tuple[str, ...] = ()
    tools: Tuple[ToolSpec, ...] = ()


@dataclass
class LLMChunk:
    text: str = ""
    tool_calls: Tuple[ToolCall, ...] = ()
    finish_reason: Optional[str] = None  # set on the last chunk: "stop" | "length" | "tool_calls" | "cancelled"
    usage: Dict[str, int] = field(default_factory=dict)  # e.g. prompt_tokens, completion_tokens (last chunk)


class LLMRuntime(ModelRuntime):
    stage = "llm"

    @abstractmethod
    def generate(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        """Stream the reply as an async iterator.

        Pull-driven: decode the next token only when the caller asks for the
        next chunk. Decoding ahead while the caller is busy (for example while
        TTS synthesizes a sentence) competes for the same CPU/GPU and measurably
        slows time to first audio. The last chunk has `finish_reason` set.
        On cancellation, raise Cancelled promptly.
        """
