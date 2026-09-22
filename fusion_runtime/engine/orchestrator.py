"""The voice conversation loop: audio → VAD → STT → turn detection → LLM → TTS → audio."""
import asyncio
import contextlib
import copy
import difflib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, List, Optional

from fusion_runtime.config import PipelineConfig, TurnDetectionConfig
from fusion_runtime.contract import (
    Cancelled,
    LLMRequest,
    LLMRuntime,
    STTRequest,
    STTRuntime,
    TTSRequest,
    TTSRuntime,
    TurnDetector,
    TurnRequest,
)
from fusion_runtime.engine.barge_in import BargeInState
from fusion_runtime.engine.conversation import Conversation
from fusion_runtime.engine.scheduler import ModelScheduler
from fusion_runtime.engine.streaming import PartialTranscript, TurnTranscriber, raise_if_error
from fusion_runtime.engine.text import END_OF_REPLY, REPLY_CUT_OFF, speakable_segments, words
from fusion_runtime.telemetry import SessionTrace, describe_error, tag_stage, telemetry
from fusion_runtime.vad import TurnState, VADBase, create_vad

# Silence after which the user's last words are transcribed, ahead of the turn ending
FINALIZE_AFTER_SILENCE_MS = 150


class PipelineOrchestrator:
    """
    Main orchestrator coordinating STT → LLM → TTS with:
    - Streaming pipeline (token-by-token)
    - Latency budgeting per stage
    - Dynamic batching
    - Graceful degradation
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        # Model runtimes, shared by every conversation. Resolved and loaded by initialize().
        self.stt: Optional[STTRuntime] = None
        self.llm: Optional[LLMRuntime] = None
        self.tts: Optional[TTSRuntime] = None
        self.ready = False
        self.vad: VADBase = create_vad(config.vad)
        self.turn_detector: Optional[TurnDetector] = None  # loaded by initialize(); None means silence only

        # Batching
        self._batch_queue: asyncio.Queue = asyncio.Queue()
        self._batch_task: Optional[asyncio.Task] = None

        # What each stage's model resolved to (runtime, format, metadata), filled in by initialize()
        self.resolved_models: dict = {}

    def _turn_detector_name(self) -> str:
        turn_config = getattr(getattr(self, "config", None), "turn_detection", None)
        if turn_config is None:
            return "silence"
        return turn_config.runtime or getattr(turn_config.provider, "value", None) or "silence"

    def scheduler(self, stage: str) -> ModelScheduler:
        """The queue in front of a stage's shared model, created on first use.

        Its concurrency comes from the loaded runtime's capabilities (the
        in-process llama.cpp model decodes one reply at a time; an HTTP
        endpoint takes many).
        """
        schedulers = self.__dict__.setdefault("_schedulers", {})
        if stage not in schedulers:
            runtime = getattr(self, stage, None)
            capabilities = getattr(runtime, "capabilities", None)
            resolved = self.__dict__.get("resolved_models", {}).get(stage)
            stage_config = getattr(getattr(self, "config", None), stage, None)
            model = _model_label(resolved) if resolved else getattr(stage_config, "model", "")
            schedulers[stage] = ModelScheduler(
                stage, model=str(model or ""), max_concurrency=getattr(capabilities, "max_concurrency", 1) or 1)
        return schedulers[stage]

    async def _load_runtime(self, stage: str):
        """Resolve a stage's model to a runtime, then load and warm it."""
        from fusion_runtime.registry import create_runtime
        from fusion_runtime.resolver import resolve_stage_config

        stage_config = getattr(self.config, stage)
        labels = {"model": getattr(stage_config, "model", None)}
        t0 = time.perf_counter()
        try:
            await self._ensure_downloaded(stage, stage_config)
            # reads file headers (a GGUF vocabulary takes tens of ms): keep it off the event loop
            resolved = await asyncio.get_running_loop().run_in_executor(None, resolve_stage_config, stage, stage_config)
            self.resolved_models[stage] = resolved
            labels = {"runtime": resolved.spec.runtime, "model": _model_label(resolved)}
            reserved = {"stage", "name", "level", "session_id", "turn_id", "request_id", "duration_ms", "error", "model", "runtime"}
            telemetry.emit("model.resolved", level="debug", stage=stage,
                           **{k: v for k, v in resolved.describe().items() if k not in reserved}, **labels)
            runtime = create_runtime(resolved.spec)
            await runtime.load()
        except Exception as e:
            tag_stage(e, stage)
            telemetry.emit("model.load_failed", level="error", stage=stage, error=describe_error(e, stage), **labels)
            raise
        setattr(self, stage, runtime)
        telemetry.emit("model.loaded", stage=stage, duration_ms=(time.perf_counter() - t0) * 1000, **labels)

    async def _load_turn_detector(self) -> None:
        """Load the configured turn detector (the built-in silence detector by default)."""
        from fusion_runtime.registry import create_runtime
        from fusion_runtime.turns import turn_detector_spec

        spec = turn_detector_spec(self.config.turn_detection)
        labels = {"runtime": spec.runtime, "model": spec.model or None}
        t0 = time.perf_counter()
        try:
            detector = create_runtime(spec)
            await detector.load()
        except Exception as e:
            tag_stage(e, "turn")
            telemetry.emit("model.load_failed", level="error", stage="turn", error=describe_error(e, "turn"), **labels)
            raise
        self.turn_detector = detector
        telemetry.emit("model.loaded", stage="turn", duration_ms=(time.perf_counter() - t0) * 1000,
                       uses_audio=detector.uses_audio, uses_history=detector.uses_history, **labels)

    async def _ensure_downloaded(self, stage: str, stage_config) -> None:
        """Fetch an hf: model the config asks for but the machine doesn't have.

        Deployments start from an empty disk, so the server downloads what the
        agent names instead of expecting someone to run `frun models pull`
        first. Set FUSION_AUTO_DOWNLOAD=0 to require it to be there already.
        """
        import os

        from fusion_runtime.catalog import (
            hf_expected_bytes,
            hf_reference,
            is_hf_downloaded,
            pull_hf,
        )
        from fusion_runtime.config import model_dir

        ref = getattr(stage_config, "model", "") or ""
        if not ref.startswith("hf:") or os.getenv("FUSION_AUTO_DOWNLOAD", "1") in ("0", "false", "False"):
            return
        root = model_dir()
        repo, revision, filename = hf_reference(ref)
        if is_hf_downloaded(root, repo, revision):
            return
        from fusion_runtime.contract import ModelNotFound
        from fusion_runtime.resolver import resolve_stage_config

        try:  # already here under another name (a catalog model is the same file), so nothing to fetch
            await asyncio.get_running_loop().run_in_executor(None, resolve_stage_config, stage, stage_config)
            return
        except ModelNotFound:
            pass
        total = await asyncio.get_running_loop().run_in_executor(
            None, hf_expected_bytes, repo, revision, os.getenv("HF_TOKEN"), filename)
        size = f"{total / 1e9:.1f} GB" if total and total >= 1e9 else (f"{total / 1e6:.0f} MB" if total else "unknown size")
        telemetry.emit("model.downloading", stage=stage, model=ref, size=size, destination=str(root / "hf"),
                       hint="first run with this model — this can take a few minutes; later starts reuse it")
        started = time.perf_counter()
        await asyncio.get_running_loop().run_in_executor(
            None, lambda: pull_hf(ref, root, log=lambda line: telemetry.emit(
                "model.download_detail", level="debug", stage=stage, detail=line.strip())))
        telemetry.emit("model.downloaded", stage=stage, model=ref, size=size,
                       duration_ms=(time.perf_counter() - started) * 1000)

    async def initialize(self):
        """Resolve, load and warm up all models, reporting each one's load time."""
        started = time.perf_counter()
        telemetry.emit("models.loading", stage="server")
        await asyncio.gather(self._load_runtime("stt"), self._load_runtime("llm"), self._load_runtime("tts"),
                             self._load_turn_detector())
        self.__dict__.pop("_schedulers", None)  # size queues from the loaded runtimes' capabilities
        self.ready = True
        # Load Silero once now, so sessions only copy it (see _load_vad_frame_model).
        await self._load_vad_frame_model()
        telemetry.emit("models.ready", stage="server", duration_ms=(time.perf_counter() - started) * 1000)

        if self.config.enable_batching:
            self._batch_task = asyncio.create_task(self._batch_worker())

    async def shutdown(self):
        pool = self.__dict__.pop("_vad_pool", None)
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        self.ready = False
        for stage in ("stt", "llm", "tts", "turn_detector"):
            runtime = self.__dict__.get(stage)
            if runtime is not None:
                with contextlib.suppress(Exception):
                    await runtime.close()
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass

    async def run_pipeline(
        self,
        audio_stream: AsyncIterator[bytes],
        system_prompt: str = "You are a helpful voice assistant.",
        on_event=None,
        barge_in: Optional["BargeInState"] = None,
        trace: Optional[SessionTrace] = None,
    ) -> AsyncIterator[bytes]:
        """
        Main pipeline: Audio → STT → LLM → TTS → Audio
        Yields audio chunks for playback.

        on_event(dict) — optional callback receiving live events:
          {"type": "transcript", "text": ..., "is_final": bool}
          {"type": "response", "text": <chunk so far>}
          {"type": "turn.trace", "turn_id": ..., "summary": {...}, "timeline": [...]}

        `trace` collects the per-turn timeline and telemetry for this
        conversation; one is created if not given.
        """
        pipeline_start = time.perf_counter()

        def emit(event: dict):
            if on_event is not None:
                try:
                    on_event(event)
                except Exception:
                    pass

        trace = trace if trace is not None else SessionTrace()
        if trace.on_turn_trace is None:
            trace.on_turn_trace = lambda turn_trace: emit({"type": "turn.trace", **turn_trace})

        # Reset state
        await self.vad.reset()
        turn_state = TurnState()
        stt_reset = asyncio.Event()
        # Callers can pass their own so they can trigger interruption from
        # outside the pipeline — e.g. the websocket layer, when a client
        # reports the user talking over the bot. The client is better placed
        # to detect that than we are: it can compare its mic level against
        # the audio it's currently playing, which we never see.
        barge_in = barge_in if barge_in is not None else BargeInState()

        # Fan the raw audio out to two independent consumers: the normal
        # STT/VAD chain below, and a barge-in watcher that keeps scanning
        # for real user speech even while that chain is blocked generating
        # a reply (it has to be a separate tap — the main chain doesn't
        # pull any further audio until the current turn's LLM/TTS call
        # returns, so it can't itself notice an interruption arriving).
        main_q: asyncio.Queue = asyncio.Queue()
        watch_q: asyncio.Queue = asyncio.Queue()
        tee_task = asyncio.create_task(self._tee_audio(audio_stream, [main_q, watch_q], trace))
        watcher_task = asyncio.create_task(
            self._barge_in_watcher(self._drain(watch_q), barge_in, emit, trace)
        )

        try:
            # Stage 1: VAD + STT Streaming
            stt_stream = self._stt_stage(
                self._drain(main_q), emit, turn_state, stt_reset, trace
            )

            # Stage 2: LLM Streaming (consumes STT partials)
            llm_stream = self._llm_stage(
                stt_stream, system_prompt, emit, turn_state, stt_reset, barge_in, trace
            )

            # Stage 3: TTS Streaming (consumes LLM tokens)
            tts_stream = self._tts_stage(llm_stream, trace)

            # Yield audio chunks
            async for audio_chunk in tts_stream:
                turn = trace.responding
                if turn is not None and "audio_first_sent" not in turn.marks:
                    turn.mark("audio_first_sent")
                    trace.event(
                        "audio.first_sent", turn=turn, stage="audio",
                        response_ms=turn.between_ms("turn_end_detected", "audio_first_sent"),
                        ttfa_ms=turn.between_ms("speech_end", "audio_first_sent") if trace.realtime_audio else None,
                    )
                yield audio_chunk
        except Exception as e:
            info = describe_error(e)
            turn = trace.responding or trace.listening
            if turn is not None:
                turn.add("errors")
            trace.event("pipeline.error", turn=turn, level="error", stage=info.stage or "pipeline", error=info)
            with contextlib.suppress(Exception):
                e.fusion_reported = True  # callers shouldn't log (or count) it again
            raise
        finally:
            tee_task.cancel()
            watcher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await tee_task
            with contextlib.suppress(asyncio.CancelledError):
                await watcher_task
            trace.finish()

        trace.event("pipeline.done", level="debug", stage="pipeline",
                    duration_ms=(time.perf_counter() - pipeline_start) * 1000,
                    turns=trace.turn_count)

    @staticmethod
    async def _tee_audio(source: AsyncIterator[bytes], queues: List[asyncio.Queue],
                         trace: Optional[SessionTrace] = None):
        """Fan one audio stream out to several queues so it can have more
        than one independent consumer (see `run_pipeline`). Also feeds the
        trace's audio clock with each chunk's arrival time."""
        try:
            async for chunk in source:
                if trace is not None:
                    trace.audio_received(len(chunk))
                for q in queues:
                    q.put_nowait(chunk)
        finally:
            for q in queues:
                q.put_nowait(None)  # sentinel: source ended

    @staticmethod
    async def _drain(q: asyncio.Queue) -> AsyncIterator[bytes]:
        while True:
            item = await q.get()
            if item is None:
                return
            yield item

    @staticmethod
    def _normalize_words(text: str) -> List[str]:
        """Lowercase, strip punctuation, split on whitespace — puts STT
        output and our own known LLM text on equal footing for comparison
        (STT never emits punctuation the same way twice; we don't want a
        stray comma to break an otherwise-exact word match)."""
        return words(text)

    @classmethod
    def _looks_like_self_echo(cls, candidate: str, bot_text: str, config: TurnDetectionConfig) -> bool:
        """True if `candidate` (a freshly transcribed 'user' turn) is
        actually the bot's own recent speech bleeding back through the mic,
        rather than something the user said.

        Deliberately text-only, not audio-domain: we don't need to model an
        acoustic echo path at all, because we already know exactly what the
        bot said, with zero noise — unlike a real reference-signal problem,
        there's no delay estimation or adaptive filtering involved, just a
        string comparison. See TurnDetectionConfig.echo_min_match_words for
        why a short/coincidental overlap doesn't trip this.
        """
        if not candidate or not bot_text:
            return False
        cand_words = cls._normalize_words(candidate)
        bot_words = cls._normalize_words(bot_text)
        if not cand_words or not bot_words:
            return False
        match = difflib.SequenceMatcher(None, cand_words, bot_words, autojunk=False).find_longest_match(
            0, len(cand_words), 0, len(bot_words)
        )
        if match.size < config.echo_min_match_words:
            return False
        return (match.size / len(cand_words)) >= config.echo_containment_ratio

    async def _vad_probabilities(self, model, frames: List[bytes], sample_rate: int) -> List[float]:
        """Speech probability per 32 ms frame, computed off the event loop.

        Silero runs ~1.5 ms per frame on CPU (50 ms on its first call), for
        every frame of every stream, twice (VAD filter and barge-in watcher).
        On the event loop that added up to stalls that froze audio input and
        interruptions for every conversation. A small dedicated pool keeps it
        from queueing behind STT, LLM and TTS work in the default executor.
        Frames of one stream go through in order, one awaited call at a time,
        so each stream's stateful model is never used concurrently.
        """
        pool = self.__dict__.get("_vad_pool")
        if pool is None:
            pool = self.__dict__["_vad_pool"] = ThreadPoolExecutor(max_workers=4, thread_name_prefix="fusion-vad")

        def infer() -> List[float]:
            import numpy as np
            import torch
            probs = []
            with torch.no_grad():
                for frame in frames:
                    audio_np = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
                    probs.append(float(model(torch.from_numpy(audio_np).unsqueeze(0), sample_rate).item()))
            return probs

        return await asyncio.get_running_loop().run_in_executor(pool, infer)

    async def _load_vad_frame_model(self):
        """A frame-level Silero VAD model for ONE audio stream.

        Silero carries context from each frame into the next, so two streams
        must never share an instance. `_apply_vad` and `_barge_in_watcher`
        used to share one, and each then saw the other's frames as context —
        the STT side routinely lags the watcher while Whisper runs. Measured
        on the hello.wav fixture with a 40-frame lag, the longest detected
        speech run fell from 1152 ms to 704 ms: enough to miss the
        sustained-speech bar for a real interruption. So every caller gets
        its own instance. A model injected on `self.vad._frame_model` that
        keeps no per-stream state (tests inject plain functions) is returned
        as-is.
        """
        injected = getattr(self.vad, "_frame_model", None)
        if injected is not None and not hasattr(injected, "reset_states"):
            return injected

        # Loading Silero from disk takes ~150-200 ms and holds Python's GIL for
        # much of it, stalling the event loop 50-70 ms even from a worker
        # thread; it used to happen twice per session. Instead it's loaded and
        # warmed once, and each stream gets a copy with fresh state (~10 ms).
        # Copies were verified to match a freshly loaded model exactly and to
        # stay independent when used alternately.
        template = self.__dict__.get("_vad_template")
        if template is None:
            lock = self.__dict__.setdefault("_vad_template_lock", asyncio.Lock())
            async with lock:
                template = self.__dict__.get("_vad_template")
                if template is None:
                    template = await self._load_vad_template()
                    if template is None:
                        return None
                    self.__dict__["_vad_template"] = template

        def fresh_copy():
            model = copy.deepcopy(template)
            model.reset_states()
            return model

        return await asyncio.get_running_loop().run_in_executor(None, fresh_copy)

    async def _load_vad_template(self):
        def load():
            import torch

            from fusion_runtime.catalog.entries import use_model_dir_for_torch_hub

            use_model_dir_for_torch_hub('snakers4/silero-vad')  # keep it beside the other models
            model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True,
                verbose=False,
            )
            with torch.no_grad():  # the first calls are slow; pay for them once, here
                for _ in range(3):
                    model(torch.zeros(1, 512), 16000)
            model.reset_states()
            return model

        t0 = time.perf_counter()
        try:
            model = await asyncio.get_running_loop().run_in_executor(None, load)
        except Exception as e:
            telemetry.emit(
                "model.load_failed", level="warning", stage="vad", error=describe_error(e, "vad"),
                runtime="silero", model="silero-vad",
                impact="no speech detection: turns end on a timer and interruptions don't work",
                hint="run: frun doctor",
            )
            return None
        telemetry.emit("model.loaded", stage="vad", duration_ms=(time.perf_counter() - t0) * 1000,
                       runtime="silero", model="silero-vad")
        return model

    async def _barge_in_watcher(
        self,
        audio_stream: AsyncIterator[bytes],
        barge_in: BargeInState,
        emit=None,
        trace: Optional[SessionTrace] = None,
    ):
        """Watches raw mic audio for genuine interruption, independent of
        whatever the main STT/LLM/TTS chain is doing.

        This is the piece that chain structurally can't do itself: once a
        turn starts generating, `_llm_stage` blocks inside the LLM/TTS call
        and doesn't pull the next STT result until that returns, so it
        can't notice new speech arriving mid-reply. This watcher taps the
        raw audio independently and, while `barge_in.speaking` is true,
        requires a sustained run of speech-probability frames (not a single
        blip) before declaring a real interruption — since our mic audio
        isn't guaranteed echo-free, a one-frame threshold would trigger on
        residual echo of the bot's own voice as readily as on a real user.
        """

        model = await self._load_vad_frame_model()
        if model is None:
            return  # no VAD available — can't detect barge-in at all

        threshold = getattr(self.vad.config, "threshold", 0.5)
        sample_rate = getattr(self.vad, "sample_rate", 16000)
        chunk_samples = 512
        bytes_per_frame = chunk_samples * 2
        frame_ms = chunk_samples / sample_rate * 1000
        min_speech_ms = getattr(self.config.turn_detection, "barge_in_min_speech_ms", 300)

        buffer = bytearray()
        speech_run_ms = 0.0
        samples_seen = 0
        async for chunk in audio_stream:
            buffer.extend(chunk)
            frames = []  # (frame, first sample index) that need a speech check
            while len(buffer) >= bytes_per_frame:
                frame = bytes(buffer[:bytes_per_frame])
                buffer = buffer[bytes_per_frame:]
                frames.append((frame, samples_seen))
                samples_seen += chunk_samples
            if not frames:
                continue

            if not barge_in.speaking:
                speech_run_ms = 0.0
                continue  # nothing to interrupt right now
            watched = []
            for frame, frame_start_sample in frames:
                arrived = trace.arrival_time(frame_start_sample) if trace is not None else None
                if arrived is not None and barge_in.speaking_since is not None and arrived < barge_in.speaking_since:
                    speech_run_ms = 0.0  # backlog from before the bot spoke: the user's own turn, not an interruption
                    continue
                watched.append(frame)
            if not watched:
                continue
            probs = await self._vad_probabilities(model, watched, sample_rate)

            for prob in probs:
                if not barge_in.speaking:
                    break  # fired (or the reply ended) while these frames were being checked

                if prob >= threshold:
                    speech_run_ms += frame_ms
                    if speech_run_ms >= min_speech_ms:
                        barge_in.fire()  # also stops re-firing until the next reply starts
                        if trace is not None:
                            turn = trace.responding
                            if turn is not None:
                                turn.mark("barge_in_fired")
                                turn.info["barge_in_speech_ms"] = speech_run_ms
                            trace.event("barge_in.fired", turn=turn, stage="barge_in",
                                        speech_over_bot_ms=round(speech_run_ms), needed_ms=min_speech_ms)
                        speech_run_ms = 0.0
                        if emit:
                            emit({"type": "interrupted"})
                else:
                    if speech_run_ms >= min_speech_ms / 2 and trace is not None:
                        # Logged so a missed interruption leaves evidence: speech
                        # was heard over the bot, but broke off before the bar.
                        trace.event("barge_in.heard", turn=trace.responding, stage="barge_in",
                                    speech_over_bot_ms=round(speech_run_ms), needed_ms=min_speech_ms,
                                    hint="speech over the bot stopped before it counted as an interruption")
                    speech_run_ms = 0.0

    async def _stt_stage(
        self,
        audio_stream: AsyncIterator[bytes],
        emit=None,
        turn_state: Optional[TurnState] = None,
        stt_reset: Optional[asyncio.Event] = None,
        trace: Optional[SessionTrace] = None,
    ) -> AsyncIterator[PartialTranscript]:
        """VAD + STT with streaming partial results."""
        stt_config = getattr(self.config, "stt", None)
        stt_scheduler = self.scheduler("stt")
        session_id = trace.session_id if trace is not None else None

        async def transcribe(pcm: bytes):
            request = STTRequest(audio=pcm, language=getattr(stt_config, "language", None), session_id=session_id)
            async with stt_scheduler.slot(request):
                results = await self.stt.transcribe([request])
            return raise_if_error(results[0])

        # Apply VAD filter
        vad_filtered = self._apply_vad(audio_stream, turn_state, trace)
        detector = self.__dict__.get("turn_detector")
        if turn_state is not None and getattr(detector, "uses_audio", False):
            vad_filtered = self._keep_turn_audio(vad_filtered, turn_state)

        def note(result: PartialTranscript) -> None:
            if result.text and trace is not None:
                turn = trace.listening_turn()
                turn.add("stt_windows")
                turn.add("stt_transcribe_ms", result.latency_ms or 0.0)
                if "stt_first_partial" not in turn.marks:
                    turn.mark("stt_first_partial")
                    trace.event("stt.first_partial", turn=turn, stage="stt", duration_ms=result.latency_ms,
                                **telemetry.content(result.text))
                else:
                    trace.event("stt.partial", turn=turn, level="debug", stage="stt", duration_ms=result.latency_ms,
                                **telemetry.content(result.text))
                turn.mark("stt_last_partial", overwrite=True)
            if result.text:
                if emit:
                    # Always partial: the one "is_final" transcript event for a turn is sent by
                    # _llm_stage once turn detection has decided the turn is over.
                    emit({"type": "transcript", "text": result.text, "is_final": False})

        # Re-transcribe the turn so far as speech arrives (see engine/streaming.py). Turn
        # detection reaches the transcriber through turn_state, to transcribe the last
        # words during a pause and to start the next turn.
        transcriber = TurnTranscriber(transcribe)
        if turn_state is not None:
            async def finalize():
                try:
                    result = await transcriber.finalize()
                except Exception as e:
                    raise tag_stage(e, "stt")
                if result is not None:
                    note(result)
                return result

            turn_state.finalize_transcript = finalize
            turn_state.start_next_turn = transcriber.reset
        stream = transcriber.stream(vad_filtered, reset_signal=stt_reset)
        while True:
            try:
                result = await stream.__anext__()
            except StopAsyncIteration:
                break
            except Exception as e:
                raise tag_stage(e, "stt")
            note(result)
            yield result

    async def _keep_turn_audio(self, chunks: AsyncIterator[bytes], turn_state: TurnState) -> AsyncIterator[bytes]:
        """Keep the latest seconds of this turn's speech for a turn detector that uses audio."""
        limit = int(self.config.turn_detection.detector_audio_s * 16000) * 2
        async for chunk in chunks:
            turn_state.speech_audio.extend(chunk)
            if len(turn_state.speech_audio) > limit:
                del turn_state.speech_audio[:len(turn_state.speech_audio) - limit]
            yield chunk

    async def _apply_vad(
        self,
        audio_stream: AsyncIterator[bytes],
        turn_state: Optional[TurnState] = None,
        trace: Optional[SessionTrace] = None,
    ) -> AsyncIterator[bytes]:
        """Filter audio through VAD, only forward speech.

        Uses frame-level Silero VAD (512-sample chunks = 32ms) to decide
        per-frame speech probability; speech frames pass through, silence
        is dropped. Keeps STT from wasting cycles on non-speech.

        Also accumulates real trailing-silence duration into `turn_state`
        (when provided) so turn detection can require an actual pause
        instead of trusting the STT's own per-window punctuation alone.
        """

        model = await self._load_vad_frame_model()
        if model is None:
            # VAD unavailable → pass everything through. Leave
            # turn_state.vad_active False so turn detection knows not
            # to gate on silence it has no way of measuring.
            async for chunk in audio_stream:
                yield chunk
            return

        threshold = getattr(self.vad.config, "threshold", 0.5)
        sample_rate = getattr(self.vad, "sample_rate", 16000)
        chunk_samples = 512  # Silero's expected frame size at 16kHz
        bytes_per_frame = chunk_samples * 2
        frame_ms = chunk_samples / sample_rate * 1000

        # Speech segment tracking for the timeline, on the audio clock (sample
        # position), so queueing behind STT doesn't distort when speech happened.
        vad_config = getattr(self.config, "vad", None)
        segment_end_silence_ms = getattr(vad_config, "min_silence_ms", 100)
        samples_seen = 0
        speaking = False
        segment_start_sample = 0
        silence_start_sample = 0

        buffer = bytearray()
        async for chunk in audio_stream:
            buffer.extend(chunk)
            frames = []
            while len(buffer) >= bytes_per_frame:
                frames.append(bytes(buffer[:bytes_per_frame]))
                buffer = buffer[bytes_per_frame:]
            if not frames:
                continue
            probs = await self._vad_probabilities(model, frames, sample_rate)

            for frame, prob in zip(frames, probs, strict=True):
                frame_start_sample = samples_seen
                samples_seen += chunk_samples
                if trace is not None:
                    if prob >= threshold:
                        if not speaking:
                            speaking = True
                            segment_start_sample = frame_start_sample
                            turn = trace.listening_turn()
                            turn.add("speech_segments")
                            first = "speech_start" not in turn.marks
                            turn.mark("speech_start", at_mono=trace.arrival_time(frame_start_sample))
                            trace.event("vad.speech_start", turn=turn, stage="vad",
                                        level="info" if first else "debug", probability=round(prob, 2),
                                        audio_offset_ms=round(frame_start_sample / sample_rate * 1000))
                        silence_start_sample = samples_seen
                    elif speaking and (samples_seen - silence_start_sample) / sample_rate * 1000 >= segment_end_silence_ms:
                        speaking = False
                        turn = trace.listening_turn()
                        turn.mark("speech_end", at_mono=trace.arrival_time(silence_start_sample), overwrite=True)
                        turn.add("speech_audio_ms", (silence_start_sample - segment_start_sample) / sample_rate * 1000)
                        trace.event("vad.speech_end", turn=turn, stage="vad", level="debug",
                                    segment_ms=round((silence_start_sample - segment_start_sample) / sample_rate * 1000),
                                    audio_offset_ms=round(silence_start_sample / sample_rate * 1000))

                if turn_state is not None:
                    turn_state.vad_active = True
                    if prob >= threshold:
                        turn_state.silence_ms = 0.0
                    else:
                        turn_state.silence_ms += frame_ms

                if prob >= threshold:
                    yield frame

        if trace is not None and speaking:
            # Audio ended mid-speech (a file, or the client hung up while talking): close the segment.
            turn = trace.listening_turn()
            turn.mark("speech_end", at_mono=trace.arrival_time(samples_seen), overwrite=True)
            turn.add("speech_audio_ms", (samples_seen - segment_start_sample) / sample_rate * 1000)
            trace.event("vad.speech_end", turn=turn, stage="vad", level="debug", reason="audio_ended",
                        segment_ms=round((samples_seen - segment_start_sample) / sample_rate * 1000))

    async def _llm_stage(
        self,
        stt_stream: AsyncIterator[PartialTranscript],
        system_prompt: str,
        emit=None,
        turn_state: Optional[TurnState] = None,
        stt_reset: Optional[asyncio.Event] = None,
        barge_in: Optional[BargeInState] = None,
        trace: Optional[SessionTrace] = None,
        conversation: Optional[Conversation] = None,
    ) -> AsyncIterator[str]:
        """LLM streaming, gated by real (forward-measured) silence rather
        than reacting only when new STT text happens to arrive.

        Turn-completion used to be decided the instant a new STT result
        showed up: check accumulated silence *right then*. But STT only
        emits a new result once new speech has accumulated — VAD drops
        silence before it ever reaches STT — so that silence reading was
        always "the pause *before* this new speech started", never "the
        pause *after* the user actually stopped talking". Any ordinary
        mid-sentence breath satisfied it just as well as a real pause,
        cutting turns off before they were finished.

        Fix: an independent watcher polls `turn_state.silence_ms`
        continuously (not just when stt_stream happens to produce
        something) and fires the instant *forward* silence crosses the
        threshold, against whatever's been transcribed so far. A second,
        lightweight task keeps that transcript current by draining
        `stt_stream` concurrently, so consuming one never blocks the other
        the way a single sequential loop would.
        """
        if conversation is None:
            llm_config = getattr(self.config, "llm", None)
            conversation = Conversation.for_context(
                system_prompt, getattr(llm_config, "n_ctx", None), getattr(llm_config, "max_tokens", None))
        llm_scheduler = self.scheduler("llm")
        first_token = True
        response_buffer = ""
        # The bot's own most recently spoken text (partial, if it was cut
        # off mid-reply) — kept around purely so a fresh "user" transcript
        # can be checked against it before we act on it. See
        # `_looks_like_self_echo` and TurnDetectionConfig.echo_min_match_words.
        last_bot_text = ""
        pending_transcript = ""
        turn_ready = asyncio.Event()
        turn_config = self.config.turn_detection
        detector = self.__dict__.get("turn_detector")
        detector_name = self._turn_detector_name()
        ask_detector = detector is not None and getattr(detector, "refines_wait", True)
        # The detector's latest answer, and the transcript (plus audio length) it was about
        prediction = {"key": None, "p": None, "ms": None}
        predict_task: Optional[asyncio.Task] = None
        detector_failed = False
        last_stt_activity = time.monotonic()
        last_turn_started: Optional[float] = None

        heard_language: Optional[str] = None  # what STT detected most recently (when not fixed in config)

        def turn_language() -> Optional[str]:
            return getattr(getattr(self.config, "stt", None), "language", None) or heard_language

        pending_audio_bytes = 0  # how much of the turn's audio pending_transcript covers

        def take_transcript(result) -> None:
            """Use a transcript unless one covering more of the turn is already in hand."""
            nonlocal pending_transcript, pending_audio_bytes, last_stt_activity
            covers = getattr(result, "audio_bytes", 0)
            if result.text and (covers >= pending_audio_bytes or not covers):
                pending_transcript = result.text
                pending_audio_bytes = covers
                last_stt_activity = time.monotonic()

        async def accumulate_stt():
            nonlocal pending_transcript, last_stt_activity, heard_language
            async for stt_result in stt_stream:
                if stt_result.language:
                    heard_language = stt_result.language
                if stt_result.text and getattr(stt_result, "audio_bytes", 0):
                    take_transcript(stt_result)
                elif stt_result.text:
                    # Replace, don't append — each STTResult restates the
                    # whole turn so far (see STTResult's docstring), so
                    # appending would stack overlapping re-transcriptions
                    # of the same speech on top of each other.
                    pending_transcript = stt_result.text
                    last_stt_activity = time.monotonic()

        def prediction_key(text: str):
            audio_len = len(turn_state.speech_audio) if turn_state is not None and detector.uses_audio else 0
            return (text, audio_len)

        async def predict(text: str, key) -> None:
            nonlocal detector_failed
            history = conversation.messages_for("")[1:-1] if detector.uses_history else ()
            request = TurnRequest(
                transcript=text,
                history=tuple(history[-turn_config.detector_history_messages:]) if history else (),
                audio=bytes(turn_state.speech_audio) if detector.uses_audio and turn_state is not None else None,
                silence_ms=turn_state.silence_ms if turn_state is not None else 0.0,
                session_id=trace.session_id if trace is not None else None,
                language=turn_language(),
            )
            started = time.perf_counter()
            try:
                async with self.scheduler("turn").slot(request):
                    result = await detector.predict(request)
                p = min(1.0, max(0.0, float(result.end_of_turn)))
                prediction.update(key=key, p=p, ms=(time.perf_counter() - started) * 1000)
            except (asyncio.CancelledError, Cancelled):
                raise
            except Exception as e:
                prediction.update(key=key, p=None, ms=None)
                if not detector_failed:  # once per call: the default wait keeps working meanwhile
                    detector_failed = True
                    telemetry.emit("turn.detector_failed", level="warning", stage="turn",
                                   error=describe_error(e, "turn"), detector=detector_name,
                                   impact=f"turns end after the default {turn_config.min_silence_ms} ms pause")

        def wait_ms(text: str):
            """How much silence ends the turn, given the detector's latest answer for this text."""
            p = prediction["p"] if ask_detector and prediction["key"] is not None and prediction["key"][0] == text else None
            if p is None:
                return turn_config.min_silence_ms, None
            if p >= turn_config.likely_threshold:
                return turn_config.min_confident_silence_ms, p
            if p < turn_config.unlikely_threshold:
                return turn_config.max_silence_ms, p
            return turn_config.min_silence_ms, p

        def record_turn_end(reason: str, threshold_ms: float, text: str, p=None, **measured) -> None:
            if trace is None:
                return
            turn = trace.listening_turn()
            if "turn_end_detected" in turn.marks:
                return
            turn.mark("turn_end_detected")
            turn.info["turn_end_reason"] = reason
            if p is not None:
                turn.info["end_of_turn_probability"] = round(p, 3)
            trace.event(
                "turn.end_detected", turn=turn, stage="turn", reason=reason, detector=detector_name,
                end_of_turn_probability=round(p, 3) if p is not None else None,
                prediction_ms=round(prediction["ms"], 1) if p is not None and prediction["ms"] is not None else None,
                threshold_ms=threshold_ms,
                wait_ms=turn.between_ms("speech_end", "turn_end_detected") if trace.realtime_audio else None,
                speech_ms=turn.between_ms("speech_start", "speech_end"),
                **{k: round(v) for k, v in measured.items()},
            )

        finalize_task: Optional[asyncio.Task] = None

        async def last_words_transcribed() -> None:
            """Wait for the words said since the last partial transcript, if any are being transcribed."""
            nonlocal finalize_task
            finalize = getattr(turn_state, "finalize_transcript", None)
            if finalize is None:
                return
            if finalize_task is None:
                finalize_task = asyncio.create_task(finalize())
            try:
                result = await asyncio.shield(finalize_task)
            except Exception:
                finalize_task = None
                raise
            finalize_task = None
            if result is not None:
                take_transcript(result)

        async def watch_for_turn_end():
            nonlocal predict_task, finalize_task
            finalized_this_pause = False
            try:
                while True:
                    await asyncio.sleep(0.05)
                    paused_ms = turn_state.silence_ms if turn_state is not None and turn_state.vad_active else 0.0
                    if paused_ms == 0:
                        finalized_this_pause = False
                    if finalize_task is not None and finalize_task.done():
                        await last_words_transcribed()  # collect it (or raise its error)
                    elif (finalize_task is None and not finalized_this_pause and paused_ms >= FINALIZE_AFTER_SILENCE_MS
                          and getattr(turn_state, "finalize_transcript", None) is not None):
                        finalized_this_pause = True
                        # The user paused: transcribe their last words now, so the transcript is
                        # complete by the time the pause is long enough to end the turn.
                        finalize_task = asyncio.create_task(turn_state.finalize_transcript())
                    text = pending_transcript.strip()
                    if not text:
                        continue

                    if ask_detector:
                        # Ask again when the words change; audio-based detectors also
                        # when the user pauses with more speech than last time.
                        key, last = prediction_key(text), prediction["key"]
                        if last is None or last[0] != text:
                            stale = True
                        else:
                            paused = turn_state is not None and turn_state.silence_ms > 0
                            stale = detector.uses_audio and paused and key[1] > last[1]
                        if stale and (predict_task is None or predict_task.done()):
                            predict_task = asyncio.create_task(predict(text, key))

                    threshold, p = wait_ms(text)
                    vad_ok = turn_state is not None and turn_state.vad_active
                    if vad_ok and turn_state.silence_ms >= threshold:
                        await last_words_transcribed()
                        if turn_state.silence_ms < threshold:
                            continue  # they started talking again while the last words were transcribed
                        text = pending_transcript.strip()
                        record_turn_end("silence", threshold, text, p, silence_ms=turn_state.silence_ms)
                        turn_ready.set()
                        continue

                    # Fallback: elapsed wall-clock time since the last newly
                    # recognized word, independent of VAD. Without this, VAD
                    # being unavailable (e.g. failing to load — this exact
                    # failure mode silently broke a prior session's turn
                    # detection entirely) or just never reporting true silence
                    # in a noisy room would wait on a threshold that can never
                    # be reached and hang forever. When VAD *is* active this
                    # only acts as a distant safety net (extra grace period on
                    # top of the normal threshold), since VAD is the fast,
                    # accurate signal in that case.
                    idle_ms = (time.monotonic() - last_stt_activity) * 1000
                    idle_ceiling = threshold if not vad_ok else threshold + 1500
                    if idle_ms >= idle_ceiling:
                        await last_words_transcribed()
                        record_turn_end("no_new_words" if vad_ok else "no_vad_idle", threshold,
                                        pending_transcript.strip(), p, idle_ms=idle_ms)
                        turn_ready.set()
            finally:
                for task in (predict_task, finalize_task):
                    if task is not None and not task.done():
                        task.cancel()

        def start_next_turn() -> None:
            """This turn's words are taken: anything heard from now on belongs to the next turn."""
            nonlocal pending_audio_bytes
            pending_audio_bytes = 0
            if turn_state is not None:
                turn_state.speech_audio.clear()
                if turn_state.start_next_turn is not None:
                    turn_state.start_next_turn()

        def resume_previous_turn() -> None:
            """If the user cut off the last reply right after their turn ended, they weren't done:
            fold that turn into this one instead of answering it separately."""
            window_s = turn_config.resume_window_ms / 1000
            fired_at = getattr(barge_in, "fired_at", None)
            if not window_s or last_turn_started is None or fired_at is None:
                return
            gap_s = fired_at - last_turn_started
            if not 0 <= gap_s <= window_s:
                return
            retracted = conversation.retract_last_turn()
            if retracted is None and not conversation.has_carried_text:
                return
            if trace is not None:
                trace.event("turn.resumed", turn=trace.listening_turn(), stage="turn", gap_ms=round(gap_s * 1000),
                            window_ms=turn_config.resume_window_ms,
                            hint="the user kept talking right after a pause, so both parts are answered as one turn")
            if emit:
                emit({"type": "turn_resumed"})

        async def process_turn(transcript: str) -> AsyncIterator[str]:
            nonlocal first_token, response_buffer, last_bot_text, last_turn_started
            resume_previous_turn()
            last_turn_started = time.monotonic()
            turn = None
            if trace is not None:
                if "turn_end_detected" not in trace.listening_turn().marks:
                    record_turn_end("audio_ended", 0, transcript)
                turn = trace.start_responding()
                turn.info["language"] = turn_language()
                trace.event("stt.final", turn=turn, stage="stt", language=turn_language(), **telemetry.content(transcript))
            if emit:
                # The one authoritative "is_final" transcript event for
                # this turn — every event out of _stt_stage was a partial.
                emit({"type": "transcript", "text": transcript, "is_final": True})
            if stt_reset is not None:
                # Drop the STT's rolling window now that this turn is
                # done, so the next turn's transcript doesn't re-include
                # (and duplicate) speech we've already acted on.
                stt_reset.set()
            start_next_turn()

            messages = conversation.messages_for(transcript)

            if barge_in is not None:
                barge_in.mark_speaking()

            llm_config = getattr(self.config, "llm", None)
            resolved = self.__dict__.get("resolved_models", {}).get("llm")
            llm_labels = {
                "runtime": resolved.spec.runtime if resolved else getattr(getattr(llm_config, "provider", None), "value", None),
                "model": _model_label(resolved) if resolved else getattr(llm_config, "model", None),
            }
            if turn is not None:
                turn.mark("llm_request")
                trace.event("llm.request", turn=turn, stage="llm", messages=len(messages),
                            history_turns=conversation.turns, **llm_labels, **telemetry.content(transcript, "prompt"))
            finish_reason = None
            # tokens/s is only meaningful when the model decodes while we wait (see Capabilities)
            measure_decode = getattr(getattr(self.llm, "capabilities", None), "decodes_on_demand", True)

            # A new turn has nothing buffered to play, so it's due now. Playback-aware
            # scheduling will move this deadline as the call's unplayed audio changes.
            request = LLMRequest(
                messages=messages,
                max_tokens=getattr(llm_config, "max_tokens", 512),
                temperature=getattr(llm_config, "temperature", 0.7),
                top_p=getattr(llm_config, "top_p", 0.9),
                session_id=trace.session_id if trace is not None else None,
                deadline=time.monotonic(),
                language=turn_language(),
            )
            async with llm_scheduler.slot(request) as slot:
                if turn is not None:
                    turn.add("llm_queue_ms", slot.wait_ms)
                    if slot.queued_ahead:
                        trace.event("llm.queued", turn=turn, stage="llm", duration_ms=slot.wait_ms,
                                    queued_ahead=slot.queued_ahead)
                stream = self.llm.generate(request)
                while True:
                    waited_from = time.monotonic()
                    try:
                        llm_result = await stream.__anext__()
                    except StopAsyncIteration:
                        break
                    except Exception as e:
                        raise tag_stage(e, "llm")
                    if turn is not None and not first_token and measure_decode:
                        # Time actually spent waiting on the model for this token. Wall-clock
                        # time would also count TTS synthesis, since the LLM only decodes
                        # when the pipeline asks for the next token.
                        turn.add("llm_decode_ms", (time.monotonic() - waited_from) * 1000)
                    if barge_in is not None and barge_in.interrupted.is_set():
                        # The watcher caught real user speech starting while we
                        # were mid-reply — stop forwarding further tokens for
                        # this turn right away. The "interrupted" emit already
                        # went out from the watcher itself, not from here, so
                        # the client hears about it as early as possible.
                        barge_in.interrupted.clear()
                        finish_reason = "interrupted"
                        if turn is not None:
                            turn.mark("llm_stopped")
                            trace.event("llm.stopped", turn=turn, stage="llm", reason="barge_in",
                                        stop_ms=turn.between_ms("barge_in_fired", "llm_stopped"))
                        await stream.aclose()
                        if emit:
                            # A normal reply only gets printed client-side on
                            # its is_final event — without this, a cut-off
                            # reply would vanish from the transcript entirely
                            # (its audio played partially, but nothing shown).
                            emit({
                                "type": "response",
                                "text": response_buffer,
                                "is_final": True,
                                "interrupted": True,
                            })
                        yield REPLY_CUT_OFF  # drop words already generated but not yet spoken
                        break

                    if first_token:
                        first_token = False
                        if turn is not None:
                            turn.mark("llm_first_token")
                            trace.event("llm.first_token", turn=turn, stage="llm",
                                        duration_ms=turn.between_ms("llm_request", "llm_first_token"), **llm_labels)

                    is_final = llm_result.finish_reason is not None
                    if is_final:
                        finish_reason = llm_result.finish_reason

                    if llm_result.text:
                        response_buffer += llm_result.text
                        if turn is not None:
                            turn.add("llm_tokens")

                    # The terminal chunk from most providers carries no text
                    # (text="", is_final=True) — it must still be emitted, or
                    # the client's "is_final" completion event never arrives
                    # and nothing ever gets printed even though TTS already
                    # spoke the reply.
                    if emit and (llm_result.text or is_final):
                        emit({
                            "type": "response",
                            "text": response_buffer,
                            "is_final": is_final,
                        })

                    if llm_result.text:
                        yield llm_result.text
                    if is_final:
                        # Lets TTS speak the last sentence now, while this turn is still open,
                        # instead of waiting for a token after the final full stop that never comes
                        yield END_OF_REPLY

            if barge_in is not None:
                barge_in.mark_idle()
            if turn is not None:
                turn.mark("llm_done")
                turn.info["llm_finish"] = finish_reason or "stop"
                tokens = int(turn.counts.get("llm_tokens", 0))
                decode_ms = turn.counts.get("llm_decode_ms")
                trace.event(
                    "llm.done", turn=turn, stage="llm", duration_ms=turn.between_ms("llm_request", "llm_done"),
                    tokens=tokens, finish=turn.info["llm_finish"],
                    tokens_per_second=round((tokens - 1) / (decode_ms / 1000), 1) if decode_ms and tokens > 1 else None,
                    **telemetry.content(response_buffer, "reply"),
                )
                trace.end_turn(turn, "interrupted" if finish_reason == "interrupted" else "completed")
            # Recorded whether this reply finished naturally or was cut
            # off — either way it's what actually got spoken, and it's
            # what the next candidate "user" turn gets checked against.
            last_bot_text = response_buffer
            conversation.add_turn(transcript, response_buffer, interrupted=finish_reason == "interrupted")
            response_buffer = ""
            first_token = True

        async def maybe_process_turn(candidate: str) -> AsyncIterator[str]:
            """Guards process_turn with the self-echo check. A candidate
            that's really just the bot's own last utterance bleeding back
            through the mic (see _looks_like_self_echo) never reaches the
            LLM at all — it's discarded here, before it can turn into a
            reply to itself."""
            if self._looks_like_self_echo(candidate, last_bot_text, self.config.turn_detection):
                if trace is not None:
                    echo_turn = trace.listening_turn()
                    trace.event("echo.discarded", turn=echo_turn, level="warning", stage="turn",
                                hint="transcript matched the bot's own last reply, so it wasn't answered",
                                **telemetry.content(candidate))
                    trace.end_turn(echo_turn, "echo_discarded")
                if emit:
                    emit({"type": "echo_discarded", "text": candidate})
                if stt_reset is not None:
                    # process_turn() normally does this; since we're not
                    # calling it, do it ourselves so the STT's rolling
                    # buffer still gets flushed instead of dragging this
                    # echoed audio into whatever the user says next.
                    stt_reset.set()
                start_next_turn()
                return
            async for token in process_turn(candidate):
                yield token

        accumulate_task = asyncio.create_task(accumulate_stt())
        watcher_task = asyncio.create_task(watch_for_turn_end())
        try:
            while True:
                ready_wait = asyncio.create_task(turn_ready.wait())
                done, _ = await asyncio.wait(
                    {ready_wait, accumulate_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if ready_wait in done:
                    turn_ready.clear()
                    if turn_state is not None:
                        turn_state.silence_ms = 0.0
                    transcript = pending_transcript.strip()
                    pending_transcript = ""
                    if transcript:
                        async for token in maybe_process_turn(transcript):
                            yield token
                    continue

                # accumulate_stt() finished — the audio source ended (e.g.
                # a finite test fixture, or the connection closing). Flush
                # whatever's left as one final turn, then stop.
                ready_wait.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await ready_wait
                transcript = pending_transcript.strip()
                pending_transcript = ""
                if transcript:
                    async for token in maybe_process_turn(transcript):
                        yield token
                return
        finally:
            accumulate_task.cancel()
            watcher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await accumulate_task
            with contextlib.suppress(asyncio.CancelledError):
                await watcher_task

    async def _tts_stage(
        self,
        llm_stream: AsyncIterator[str],
        trace: Optional[SessionTrace] = None,
    ) -> AsyncIterator[bytes]:
        """Speak the reply sentence by sentence as the LLM streams it."""
        first_chunk = True
        tts_config = getattr(self.config, "tts", None)
        output_rate = getattr(tts_config, "sample_rate", 24000)
        tts_scheduler = self.scheduler("tts")
        session_id = trace.session_id if trace is not None else None

        async for segment in speakable_segments(llm_stream):
            if not segment.strip():
                continue
            request = TTSRequest(
                text=segment,
                language=getattr(tts_config, "language", None),
                voice=getattr(tts_config, "voice", None),
                speed=getattr(tts_config, "speed", 1.0),
                session_id=session_id,
            )
            stream = self.tts.synthesize(request)
            try:
                while True:
                    # A slot per chunk, never held while the audio goes downstream
                    started = time.perf_counter()
                    async with tts_scheduler.slot(request):
                        try:
                            chunk = await stream.__anext__()
                        except StopAsyncIteration:
                            break
                        except Exception as e:
                            raise tag_stage(e, "tts")
                    synth_ms = (time.perf_counter() - started) * 1000
                    pcm = _resample_pcm16(chunk.pcm, chunk.sample_rate, output_rate)
                    if trace is not None and trace.responding is not None and pcm:
                        turn = trace.responding
                        audio_s = len(pcm) / 2 / output_rate
                        turn.add("tts_chunks")
                        turn.add("tts_audio_s", audio_s)
                        turn.add("tts_synth_ms", synth_ms)
                        if "tts_first_chunk" not in turn.marks:
                            turn.mark("tts_first_chunk")
                            trace.event("tts.first_chunk", turn=turn, stage="tts", duration_ms=synth_ms,
                                        audio_ms=round(audio_s * 1000),
                                        since_first_token_ms=turn.between_ms("llm_first_token", "tts_first_chunk"))
                        else:
                            trace.event("tts.chunk", turn=turn, level="debug", stage="tts", duration_ms=synth_ms,
                                        audio_ms=round(audio_s * 1000))
                    if first_chunk:
                        first_chunk = False
                    yield pcm
            finally:
                await stream.aclose()

    # ============ Batching Support ============

    async def _batch_worker(self):
        """Background task for dynamic batching."""
        while True:
            try:
                batch = []
                # Collect requests up to max_batch_size or timeout
                while len(batch) < self.config.max_batch_size:
                    try:
                        item = await asyncio.wait_for(
                            self._batch_queue.get(),
                            timeout=self.config.batch_timeout_ms / 1000
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break

                if batch:
                    await self._process_batch(batch)

            except asyncio.CancelledError:
                break
            except Exception as e:
                telemetry.emit("batch_worker.error", level="error", stage="engine", error=describe_error(e))

    async def _process_batch(self, batch: List):
        """Process a batch of requests."""
        # Group by stage
        # This is where cross-request batching happens
        pass

def _model_label(resolved) -> str:
    """Short, stable model name for logs and metrics: catalog id, name on the server, or path."""
    return resolved.catalog_id or resolved.spec.options.get("model_name") or resolved.spec.model


def _resample_pcm16(pcm: bytes, rate: int, target: int) -> bytes:
    """Linear resampling of mono 16-bit PCM, when a TTS model's rate differs from the output rate."""
    if rate == target or not pcm:
        return pcm
    import numpy as np

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    count = max(1, round(len(samples) * target / rate))
    positions = np.linspace(0, len(samples) - 1, count)
    return np.interp(positions, np.arange(len(samples)), samples).astype(np.int16).tobytes()


# Convenience function for single-turn (non-streaming) usage
async def run_single_turn(
    orchestrator: PipelineOrchestrator,
    audio: bytes,
    system_prompt: str = "You are a helpful voice assistant."
) -> bytes:
    """Run pipeline on complete audio, return complete audio response."""
    async def audio_chunks():
        yield audio

    output = bytearray()
    async for chunk in orchestrator.run_pipeline(audio_chunks(), system_prompt):
        output.extend(chunk)
    return bytes(output)
