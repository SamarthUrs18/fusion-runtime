"""
LLM Base Classes and Implementations
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional, List
import asyncio


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


class LlamaCppLLM(LLMBase):
    """llama.cpp implementation with streaming."""
    
    async def _warmup_impl(self):
        from llama_cpp import Llama
        self.llm = Llama(
            model_path=self.config.model,
            n_ctx=self.config.n_ctx,
            n_gpu_layers=self.config.n_gpu_layers,
            n_batch=self.config.n_batch,
            n_threads=self.config.n_threads,
            verbose=False,
        )
        # Warmup
        await self.generate([ChatMessage(role="user", content="Hi")])
    
    async def generate_stream(
        self,
        messages: List[ChatMessage],
        budget_ms: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[LLMResult]:
        import time
        
        # Convert to llama.cpp format
        prompt = self._format_prompt(messages)
        
        start = time.perf_counter()
        first_token = True
        token_count = 0
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        
        def _generate():
            return self.llm(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                stream=True,
            )
        
        stream = await loop.run_in_executor(None, _generate)
        
        for chunk in stream:
            if "choices" in chunk and chunk["choices"]:
                delta = chunk["choices"][0].get("text", "")
                if delta:
                    token_count += 1
                    latency = (time.perf_counter() - start) * 1000
                    
                    yield LLMResult(
                        text=delta,
                        is_final=False,
                        tokens_used=token_count,
                        latency_ms=latency,
                    )
                    first_token = False
        
        # Final chunk
        yield LLMResult(
            text="",
            is_final=True,
            tokens_used=token_count,
            latency_ms=(time.perf_counter() - start) * 1000,
            finish_reason="stop",
        )
    
    async def generate(self, messages: List[ChatMessage], **kwargs) -> LLMResult:
        import time
        
        prompt = self._format_prompt(messages)
        start = time.perf_counter()
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.llm(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                stream=False,
            )
        )
        
        text = result["choices"][0]["text"]
        tokens = result["usage"]["completion_tokens"]
        
        return LLMResult(
            text=text,
            is_final=True,
            tokens_used=tokens,
            latency_ms=(time.perf_counter() - start) * 1000,
            finish_reason=result["choices"][0].get("finish_reason"),
        )
    
    def _format_prompt(self, messages: List[ChatMessage]) -> str:
        """Format messages for Qwen2.5 chat template."""
        formatted = []
        for msg in messages:
            if msg.role == "system":
                formatted.append(f"<|im_start|>system\n{msg.content}<|im_end|>")
            elif msg.role == "user":
                formatted.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
            elif msg.role == "assistant":
                formatted.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")
        formatted.append("<|im_start|>assistant\n")
        return "\n".join(formatted)


class OpenAILLM(LLMBase):
    """OpenAI API implementation."""
    
    async def _warmup_impl(self):
        import openai
        self.client = openai.AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base,
        )
    
    async def generate_stream(
        self,
        messages: List[ChatMessage],
        budget_ms: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[LLMResult]:
        import time
        
        openai_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]
        
        start = time.perf_counter()
        token_count = 0
        
        stream = await self.client.chat.completions.create(
            model=self.config.model,
            messages=openai_messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                token_count += 1
                yield LLMResult(
                    text=chunk.choices[0].delta.content,
                    is_final=False,
                    tokens_used=token_count,
                    latency_ms=(time.perf_counter() - start) * 1000,
                )
        
        yield LLMResult(
            text="",
            is_final=True,
            tokens_used=token_count,
            latency_ms=(time.perf_counter() - start) * 1000,
            finish_reason="stop",
        )
    
    async def generate(self, messages: List[ChatMessage], **kwargs) -> LLMResult:
        import time
        
        openai_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]
        
        start = time.perf_counter()
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=openai_messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=False,
        )
        
        return LLMResult(
            text=response.choices[0].message.content,
            is_final=True,
            tokens_used=response.usage.completion_tokens,
            latency_ms=(time.perf_counter() - start) * 1000,
            finish_reason=response.choices[0].finish_reason,
        )


def create_llm(config) -> LLMBase:
    """Factory function to create LLM instance from config."""
    from fusion_runtime.config import Provider
    
    if config.provider == Provider.LLAMA_CPP:
        return LlamaCppLLM(config)
    elif config.provider == Provider.OPENAI:
        return OpenAILLM(config)
    # Add vLLM, Ollama, Anthropic...
    raise ValueError(f"Unknown LLM provider: {config.provider}")