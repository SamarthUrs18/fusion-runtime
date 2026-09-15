"""LLM interface shared by every LLM engine."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional, List


@dataclass
class LLMResult:
    text: str
    is_final: bool
    tokens_used: int
    latency_ms: float
    finish_reason: Optional[str] = None


@dataclass
class ChatMessage:
    role: str  # system, user, assistant
    content: str


class LLMBase(ABC):
    """Base class for all LLM providers."""
    
    def __init__(self, config):
        self.config = config
        self._warm = False
    
    @abstractmethod
    async def generate_stream(
        self,
        messages: List[ChatMessage],
        budget_ms: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[LLMResult]:
        """Stream generation token by token."""
        pass
    
    @abstractmethod
    async def generate(self, messages: List[ChatMessage], **kwargs) -> LLMResult:
        """Non-streaming generation."""
        pass
    
    async def warmup(self):
        if not self._warm:
            await self._warmup_impl()
            self._warm = True
    
    @abstractmethod
    async def _warmup_impl(self):
        pass
