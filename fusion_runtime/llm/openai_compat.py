"""LLM engine for any OpenAI-compatible endpoint (llama-server, vLLM, OpenAI)."""
from typing import AsyncIterator, Optional, List

from fusion_runtime.llm.base import ChatMessage, LLMBase, LLMResult


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
