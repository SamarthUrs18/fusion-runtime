"""Any GGUF language model, in process, through llama.cpp.

The prompt format comes from the chat template stored in the GGUF file
(llama-cpp-python renders it), so Llama, Mistral, Gemma, Qwen and their
fine-tunes all work without model-specific code. A file without a template
can be given one through the `chat_template` option.

Decoding runs on a worker thread, never on the event loop: a decode step is a
blocking C call, and running it on the loop froze audio input, barge-in
detection and every other connection for the whole reply.

The thread decodes a token only when the caller asks for one. While TTS is
synthesizing a sentence it isn't asking, so decoding pauses. Letting the
thread decode ahead, even one token, slowed Kokoro down badly on CPU: median
time to first audio went from ~650 ms to 900-1600 ms (8 GB MacBook Air),
because llama.cpp's threads keep the cores busy after each token.

Options (from config): n_ctx, n_gpu_layers, n_batch, n_threads, chat_template,
warmup (default true).
"""
import asyncio
import threading
from typing import Any, AsyncIterator, Dict, Optional

from fusion_runtime.contract import (
    Cancelled,
    Capabilities,
    Health,
    InvalidRequest,
    LLMChunk,
    LLMRequest,
    LLMRuntime,
    ModelNotFound,
    RuntimeFailure,
)

_DONE = object()  # end-of-stream marker from the decode thread


class LlamaCppLLM(LLMRuntime):
    def __init__(self, spec):
        super().__init__(spec)
        self.llm = None
        # One llama.cpp context decodes one prompt at a time. Concurrent callers
        # wait here instead of mixing their tokens into each other's replies.
        self._decode_lock = threading.Lock()

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(max_concurrency=1, languages=None, tools=False)

    def health(self) -> Health:
        return Health("ok") if self.llm is not None else Health("down", "model not loaded")

    async def load(self) -> None:
        from pathlib import Path

        options = self.spec.options
        path = Path(self.spec.model)
        if not path.is_file():
            raise ModelNotFound(f"GGUF model not found at {path}. Run: frun models pull")

        def build():
            from llama_cpp import Llama

            kwargs: Dict[str, Any] = dict(
                model_path=str(path),
                n_ctx=options.get("n_ctx", 4096),
                n_gpu_layers=options.get("n_gpu_layers", -1),
                n_batch=options.get("n_batch", 512),
                verbose=False,
            )
            if options.get("n_threads"):
                kwargs["n_threads"] = options["n_threads"]
            llm = Llama(**kwargs)
            template = options.get("chat_template")
            if template:
                from llama_cpp import llama_chat_format

                llm.chat_handler = llama_chat_format.Jinja2ChatFormatter(
                    template=template,
                    eos_token=llm._model.token_get_text(llm.token_eos()),
                    bos_token=llm._model.token_get_text(llm.token_bos()),
                    stop_token_ids=[llm.token_eos()],
                ).to_chat_handler()
            return llm

        self.llm = await asyncio.get_running_loop().run_in_executor(None, build)
        if options.get("warmup", True):
            async for _ in self.generate(LLMRequest(messages=[_user("Hi")], max_tokens=4)):
                pass

    async def close(self) -> None:
        llm, self.llm = self.llm, None
        if llm is not None:
            with self._decode_lock:  # never free the model under a running decode
                close = getattr(llm, "close", None)
                if close is not None:
                    close()

    async def generate(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        if not request.messages:
            raise InvalidRequest("messages must not be empty")
        if request.tools:
            raise InvalidRequest("tool calling isn't supported by the llama_cpp runtime yet")
        request.cancel.raise_if_cancelled()
        if self.llm is None:
            raise RuntimeFailure("model not loaded")

        llm = self.llm
        messages = [_to_openai(m) for m in request.messages]
        loop = asyncio.get_running_loop()
        tokens: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()  # set when the caller stops listening (barge-in, cancel, close)
        wanted = threading.Semaphore(0)  # released once each time the caller wants a token
        request.cancel.add_callback(stop.set)

        def wait_for_request() -> bool:
            while not wanted.acquire(timeout=0.05):
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
                    stream = llm.create_chat_completion(
                        messages=messages,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        stop=list(request.stop) or None,
                        stream=True,
                    )
                    try:
                        while wait_for_request():
                            text, finish = "", None
                            while not text and finish is None:  # decode until there's something to hand over
                                chunk = next(stream, None)
                                if chunk is None:
                                    finish = "stop"
                                    break
                                choice = (chunk.get("choices") or [{}])[0]
                                text = (choice.get("delta") or {}).get("content") or ""
                                finish = choice.get("finish_reason")
                            send((text, finish))
                            if finish is not None:
                                return
                    finally:
                        close = getattr(stream, "close", None)
                        if close is not None:
                            close()
            except Exception as e:
                send(e)
            finally:
                send(_DONE)

        loop.run_in_executor(None, decode)
        produced = 0
        try:
            while True:
                wanted.release()  # ask the decode thread for the next token
                item = await tokens.get()
                if request.cancel.cancelled:
                    raise Cancelled(request.cancel.reason or "cancelled")
                if item is _DONE:
                    yield LLMChunk(finish_reason="stop", usage={"completion_tokens": produced})
                    return
                if isinstance(item, Exception):
                    raise RuntimeFailure(f"llama.cpp decode failed: {item}") from item
                text, finish = item
                if text:
                    produced += 1
                if finish is not None:
                    yield LLMChunk(text=text, finish_reason=finish, usage={"completion_tokens": produced})
                    return
                yield LLMChunk(text=text)
        finally:
            stop.set()  # finished, failed, or abandoned mid-reply: stop decoding


def _user(text: str):
    from fusion_runtime.contract import Message

    return Message(role="user", content=text)


def _to_openai(message) -> Dict[str, Optional[str]]:
    out = {"role": message.role, "content": message.content}
    if message.name:
        out["name"] = message.name
    if message.tool_call_id:
        out["tool_call_id"] = message.tool_call_id
    return out
