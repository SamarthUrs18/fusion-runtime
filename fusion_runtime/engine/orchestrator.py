"""The voice conversation loop: audio → VAD → STT → turn detection → LLM → TTS → audio."""
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Optional, List, Callable, Awaitable
import asyncio
import contextlib
import copy
import difflib
import re
import time
from collections import deque

from fusion_runtime.config import PipelineConfig, TurnDetectionConfig
from fusion_runtime.stt import STTBase, STTResult, create_stt
from fusion_runtime.llm import LLMBase, LLMResult, ChatMessage, create_llm
from fusion_runtime.tts import TTSBase, TTSResult, create_tts
from fusion_runtime.vad import VADBase, VADResult, TurnDetectorBase, TurnState, create_vad, create_turn_detector
from fusion_runtime.engine.barge_in import BargeInState
from fusion_runtime.engine.metrics import LatencyBudget, PipelineMetrics, StageBudget
from fusion_runtime.telemetry import SessionTrace, describe_error, tag_stage, telemetry


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
        self.stt: STTBase = create_stt(config.stt)
        self.llm: LLMBase = create_llm(config.llm)
        self.tts: TTSBase = create_tts(config.tts)
        self.vad: VADBase = create_vad(config.vad)
        self.turn_detector: TurnDetectorBase = create_turn_detector(config.turn_detection)
        
        # Batching
        self._batch_queue: asyncio.Queue = asyncio.Queue()
        self._batch_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.metrics_history: deque = deque(maxlen=1000)
    
    async def initialize(self):
        """Load and warm up all models, reporting each one's load time."""
        started = time.perf_counter()
        telemetry.emit("models.loading", stage="server")

        async def load(stage: str, warmup):
            stage_config = getattr(self.config, stage)
            t0 = time.perf_counter()
            try:
                await warmup()
            except Exception as e:
                tag_stage(e, stage)
                telemetry.emit("model.load_failed", level="error", stage=stage, error=describe_error(e, stage),
                               runtime=stage_config.provider.value, model=stage_config.model)
                raise
            telemetry.emit("model.loaded", stage=stage, duration_ms=(time.perf_counter() - t0) * 1000,
                           runtime=stage_config.provider.value, model=stage_config.model)

        await asyncio.gather(
            load("stt", self.stt.warmup),
            load("llm", self.llm.warmup),
            load("tts", self.tts.warmup),
        )
        # Load Silero once now, so sessions only copy it (see _load_vad_frame_model).
        await self._load_vad_frame_model()
        telemetry.emit("models.ready", stage="server", duration_ms=(time.perf_counter() - started) * 1000)
        
        if self.config.enable_batching:
            self._batch_task = asyncio.create_task(self._batch_worker())
    
    async def shutdown(self):
        pool = self.__dict__.pop("_vad_pool", None)
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
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
        budget = LatencyBudget(total_ms=self.config.target_latency_ms)
        metrics = PipelineMetrics()
        metrics.pipeline_start = pipeline_start

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
        await self.turn_detector.reset()
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
                self._drain(main_q), budget, metrics, emit, turn_state, stt_reset, trace
            )

            # Stage 2: LLM Streaming (consumes STT partials)
            llm_stream = self._llm_stage(
                stt_stream, system_prompt, budget, metrics, emit, turn_state, stt_reset, barge_in, trace
            )

            # Stage 3: TTS Streaming (consumes LLM tokens)
            tts_stream = self._tts_stage(llm_stream, budget, metrics, trace)

            # Yield audio chunks
            first_audio = True
            async for audio_chunk in tts_stream:
                now = time.perf_counter()
                if first_audio:
                    metrics.tts_first_chunk_ms = (now - pipeline_start) * 1000
                    first_audio = False
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

        metrics.e2e_latency_ms = (time.perf_counter() - pipeline_start) * 1000
        self.metrics_history.append(metrics)
        trace.event("pipeline.done", level="debug", stage="pipeline", duration_ms=metrics.e2e_latency_ms,
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
        return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()

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
        import numpy as np

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
        budget: LatencyBudget,
        metrics: PipelineMetrics,
        emit=None,
        turn_state: Optional[TurnState] = None,
        stt_reset: Optional[asyncio.Event] = None,
        trace: Optional[SessionTrace] = None,
    ) -> AsyncIterator[STTResult]:
        """VAD + STT with streaming partial results."""
        stt_start = time.perf_counter()
        stt_budget = budget.allocate("stt", 100)

        # Apply VAD filter
        vad_filtered = self._apply_vad(audio_stream, turn_state, trace)

        # Stream STT
        stream = self.stt.transcribe_stream(
            vad_filtered, budget_ms=stt_budget.remaining_ms, reset_signal=stt_reset
        )
        while True:
            try:
                result = await stream.__anext__()
            except StopAsyncIteration:
                break
            except Exception as e:
                raise tag_stage(e, "stt")
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
                    # Always partial here — `result.is_final` is just
                    # Whisper's own per-window guess (it ends windows in
                    # punctuation constantly, mid-sentence or not) and is no
                    # longer treated as authoritative. `_llm_stage`'s silence
                    # watcher emits the one real "is_final" transcript event
                    # once a turn has actually ended.
                    emit({"type": "transcript", "text": result.text, "is_final": False})
            if result.is_final:
                metrics.stt_latency_ms = (time.perf_counter() - stt_start) * 1000
            yield result
    
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
        import numpy as np

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

            for frame, prob in zip(frames, probs):
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
        stt_stream: AsyncIterator[STTResult],
        system_prompt: str,
        budget: LatencyBudget,
        metrics: PipelineMetrics,
        emit=None,
        turn_state: Optional[TurnState] = None,
        stt_reset: Optional[asyncio.Event] = None,
        barge_in: Optional[BargeInState] = None,
        trace: Optional[SessionTrace] = None,
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
        llm_start = time.perf_counter()
        first_token = True
        response_buffer = ""
        # The bot's own most recently spoken text (partial, if it was cut
        # off mid-reply) — kept around purely so a fresh "user" transcript
        # can be checked against it before we act on it. See
        # `_looks_like_self_echo` and TurnDetectionConfig.echo_min_match_words.
        last_bot_text = ""
        pending_transcript = ""
        turn_ready = asyncio.Event()
        min_confident_ms = self.config.turn_detection.min_confident_silence_ms
        min_silence_ms = self.config.turn_detection.min_silence_ms
        last_stt_activity = time.monotonic()

        async def accumulate_stt():
            nonlocal pending_transcript, last_stt_activity
            async for stt_result in stt_stream:
                if stt_result.text:
                    # Replace, don't append — each STTResult restates the
                    # whole turn so far (see STTResult's docstring), so
                    # appending would stack overlapping re-transcriptions
                    # of the same speech on top of each other.
                    pending_transcript = stt_result.text
                    last_stt_activity = time.monotonic()

        def record_turn_end(reason: str, threshold_ms: float, text: str, **measured) -> None:
            if trace is None:
                return
            turn = trace.listening_turn()
            if "turn_end_detected" in turn.marks:
                return
            turn.mark("turn_end_detected")
            turn.info["turn_end_reason"] = reason
            trace.event(
                "turn.end_detected", turn=turn, stage="turn", reason=reason,
                sounded_complete=self.turn_detector.looks_complete(text), threshold_ms=threshold_ms,
                wait_ms=turn.between_ms("speech_end", "turn_end_detected") if trace.realtime_audio else None,
                speech_ms=turn.between_ms("speech_start", "speech_end"),
                **{k: round(v) for k, v in measured.items()},
            )

        async def watch_for_turn_end():
            while True:
                await asyncio.sleep(0.05)
                text = pending_transcript.strip()
                if not text:
                    continue
                # A confidently-complete-sounding utterance only needs a
                # brief confirmation pause; anything ambiguous waits for
                # the longer, safer silence — so a clear "thanks, bye."
                # responds fast while a trailed-off "so I was..." doesn't
                # get answered mid-thought.
                threshold = (
                    min_confident_ms
                    if self.turn_detector.looks_complete(text)
                    else min_silence_ms
                )

                vad_ok = turn_state is not None and turn_state.vad_active
                if vad_ok and turn_state.silence_ms >= threshold:
                    record_turn_end("silence", threshold, text, silence_ms=turn_state.silence_ms)
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
                    record_turn_end("no_new_words" if vad_ok else "no_vad_idle", threshold, text, idle_ms=idle_ms)
                    turn_ready.set()

        async def process_turn(transcript: str) -> AsyncIterator[str]:
            nonlocal first_token, response_buffer, last_bot_text
            turn = None
            if trace is not None:
                if "turn_end_detected" not in trace.listening_turn().marks:
                    record_turn_end("audio_ended", 0, transcript)
                turn = trace.start_responding()
                trace.event("stt.final", turn=turn, stage="stt", **telemetry.content(transcript))
            if emit:
                # The one authoritative "is_final" transcript event for
                # this turn — every event out of _stt_stage was a partial.
                emit({"type": "transcript", "text": transcript, "is_final": True})
            if stt_reset is not None:
                # Drop the STT's rolling window now that this turn is
                # done, so the next turn's transcript doesn't re-include
                # (and duplicate) speech we've already acted on.
                stt_reset.set()

            messages = [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=transcript),
            ]
            llm_budget = budget.allocate("llm", 150)

            if barge_in is not None:
                barge_in.mark_speaking()

            llm_config = getattr(self.config, "llm", None)
            llm_labels = {
                "runtime": getattr(getattr(llm_config, "provider", None), "value", None),
                "model": getattr(llm_config, "model", None),
            }
            if turn is not None:
                turn.mark("llm_request")
                trace.event("llm.request", turn=turn, stage="llm", messages=len(messages),
                            **llm_labels, **telemetry.content(transcript, "prompt"))
            finish_reason = None

            stream = self.llm.generate_stream(messages, budget_ms=llm_budget.remaining_ms)
            while True:
                waited_from = time.monotonic()
                try:
                    llm_result = await stream.__anext__()
                except StopAsyncIteration:
                    break
                except Exception as e:
                    raise tag_stage(e, "llm")
                if turn is not None and not first_token:
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
                    break

                if first_token:
                    metrics.llm_first_token_ms = (time.perf_counter() - metrics.pipeline_start) * 1000
                    first_token = False
                    if turn is not None:
                        turn.mark("llm_first_token")
                        trace.event("llm.first_token", turn=turn, stage="llm",
                                    duration_ms=turn.between_ms("llm_request", "llm_first_token"), **llm_labels)

                if llm_result.is_final:
                    metrics.llm_total_ms = (time.perf_counter() - llm_start) * 1000
                    finish_reason = llm_result.finish_reason or "stop"

                if llm_result.text:
                    response_buffer += llm_result.text
                    if turn is not None:
                        turn.add("llm_tokens")

                # The terminal chunk from most providers carries no text
                # (text="", is_final=True) — it must still be emitted, or
                # the client's "is_final" completion event never arrives
                # and nothing ever gets printed even though TTS already
                # spoke the reply.
                if emit and (llm_result.text or llm_result.is_final):
                    emit({
                        "type": "response",
                        "text": response_buffer,
                        "is_final": llm_result.is_final,
                    })

                if llm_result.text:
                    yield llm_result.text

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
        budget: LatencyBudget,
        metrics: PipelineMetrics,
        trace: Optional[SessionTrace] = None,
    ) -> AsyncIterator[bytes]:
        """TTS streaming from LLM tokens."""
        tts_start = time.perf_counter()
        first_chunk = True
        
        tts_budget = budget.allocate("tts", 100)
        
        stream = self.tts.synthesize_stream(llm_stream, budget_ms=tts_budget.remaining_ms)
        while True:
            try:
                tts_result = await stream.__anext__()
            except StopAsyncIteration:
                break
            except Exception as e:
                raise tag_stage(e, "tts")
            if trace is not None and trace.responding is not None and tts_result.audio:
                turn = trace.responding
                audio_s = len(tts_result.audio) / 2 / (tts_result.sample_rate or 24000)
                turn.add("tts_chunks")
                turn.add("tts_audio_s", audio_s)
                turn.add("tts_synth_ms", tts_result.latency_ms or 0.0)
                if "tts_first_chunk" not in turn.marks:
                    turn.mark("tts_first_chunk")
                    trace.event("tts.first_chunk", turn=turn, stage="tts", duration_ms=tts_result.latency_ms,
                                audio_ms=round(audio_s * 1000),
                                since_first_token_ms=turn.between_ms("llm_first_token", "tts_first_chunk"))
                else:
                    trace.event("tts.chunk", turn=turn, level="debug", stage="tts", duration_ms=tts_result.latency_ms,
                                audio_ms=round(audio_s * 1000))
            if first_chunk:
                metrics.tts_first_chunk_ms = (time.perf_counter() - metrics.pipeline_start) * 1000
                first_chunk = False
            
            if tts_result.is_final:
                metrics.tts_total_ms = (time.perf_counter() - tts_start) * 1000
            
            yield tts_result.audio
    
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
    
    def get_metrics_summary(self) -> dict:
        """Get aggregated metrics."""
        if not self.metrics_history:
            return {}
        
        m = self.metrics_history
        return {
            "count": len(m),
            "stt_p50": sorted([x.stt_latency_ms for x in m])[len(m)//2],
            "llm_first_p50": sorted([x.llm_first_token_ms for x in m])[len(m)//2],
            "tts_first_p50": sorted([x.tts_first_chunk_ms for x in m])[len(m)//2],
            "e2e_p50": sorted([x.e2e_latency_ms for x in m])[len(m)//2],
            "e2e_p99": sorted([x.e2e_latency_ms for x in m])[int(len(m)*0.99)],
        }


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
