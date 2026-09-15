"""In-process llama.cpp LLM engine (GGUF models)."""
from typing import AsyncIterator, Optional, List
import asyncio

from fusion_runtime.llm.base import ChatMessage, LLMBase, LLMResult


class LlamaCppLLM(LLMBase):
    """llama.cpp implementation with streaming."""
    
    async def _warmup_impl(self):
        from llama_cpp import Llama
        from fusion_runtime.config import resolve_model_path

        model_path = resolve_model_path(self.config.model)
        if not model_path.exists():
            raise FileNotFoundError(
                f"LLM model not found at {model_path}. "
                "Run: python3 scripts/download_models.py --llm"
            )
        self.llm = Llama(
            model_path=str(model_path),
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
