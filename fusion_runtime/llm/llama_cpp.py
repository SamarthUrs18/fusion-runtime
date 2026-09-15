"""In-process llama.cpp LLM engine (GGUF models)."""
from typing import AsyncIterator, Optional, List
import asyncio
import threading

from fusion_runtime.llm.base import ChatMessage, LLMBase, LLMResult

_DONE = object()  # end-of-stream marker from the decode thread


class LlamaCppLLM(LLMBase):
    """llama.cpp implementation with streaming.

    Decoding runs on a worker thread, never on the event loop: a decode step
    is a blocking C call, and running it on the loop froze audio input,
    barge-in detection and every other connection for the whole reply.

    The thread decodes a token only when the caller asks for one. While TTS is
    synthesizing a sentence it isn't asking, so decoding pauses. Letting the
    thread decode ahead, even one token, slowed Kokoro down badly on CPU:
    median time to first audio went from ~650 ms to 900-1600 ms (8 GB MacBook
    Air), because llama.cpp's threads keep the cores busy after each token.
    """

    def __init__(self, config):
        super().__init__(config)
        # One llama.cpp context decodes one prompt at a time. Concurrent callers
        # wait here instead of mixing their tokens into each other's replies.
        self._decode_lock = threading.Lock()

    async def _warmup_impl(self):
        from llama_cpp import Llama
        from fusion_runtime.config import resolve_model_path

        model_path = resolve_model_path(self.config.model)
        if not model_path.exists():
            raise FileNotFoundError(
                f"LLM model not found at {model_path}. "
                "Run: frun models pull --llm"
            )
        self.llm = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: Llama(
                model_path=str(model_path),
                n_ctx=self.config.n_ctx,
                n_gpu_layers=self.config.n_gpu_layers,
                n_batch=self.config.n_batch,
                n_threads=self.config.n_threads,
                verbose=False,
            ),
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

        prompt = self._format_prompt(messages)
        start = time.perf_counter()
        loop = asyncio.get_running_loop()
        tokens: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()  # set when the caller stops listening (e.g. barge-in)
        requests = threading.Semaphore(0)  # released once each time the caller wants a token

        def wait_for_request() -> bool:
            while not requests.acquire(timeout=0.05):
                if stop.is_set():
                    return False
            return not stop.is_set()

        def send(item):
            try:
                loop.call_soon_threadsafe(tokens.put_nowait, item)
            except RuntimeError:  # event loop already closed: nobody is listening
                stop.set()

        def decode():
            try:
                with self._decode_lock:
                    if stop.is_set():
                        return
                    stream = self.llm(
                        prompt,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        stream=True,
                    )
                    try:
                        while wait_for_request():
                            text = ""
                            while not text:  # decode until there's text to hand over
                                chunk = next(stream, None)
                                if chunk is None:
                                    return
                                choices = chunk.get("choices")
                                text = choices[0].get("text", "") if choices else ""
                            send(text)
                    finally:
                        stream.close()
            except Exception as e:
                send(e)
            finally:
                send(_DONE)

        loop.run_in_executor(None, decode)
        token_count = 0
        try:
            while True:
                requests.release()  # ask the decode thread for the next token
                item = await tokens.get()
                if item is _DONE:
                    break
                if isinstance(item, Exception):
                    raise item
                token_count += 1
                yield LLMResult(
                    text=item,
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
        finally:
            stop.set()  # finished, failed, or abandoned mid-reply: stop decoding

    async def generate(self, messages: List[ChatMessage], **kwargs) -> LLMResult:
        import time

        prompt = self._format_prompt(messages)
        start = time.perf_counter()

        def decode():
            with self._decode_lock:
                return self.llm(
                    prompt,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    stream=False,
                )

        result = await asyncio.get_running_loop().run_in_executor(None, decode)

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
