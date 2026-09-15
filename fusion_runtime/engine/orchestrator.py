"""The voice conversation loop: audio → VAD → STT → turn detection → LLM → TTS → audio."""
from typing import AsyncIterator, Optional, List, Callable, Awaitable
import asyncio
import contextlib
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
        """Warm up all models."""
        print("🔥 Warming up models...")
        await asyncio.gather(
            self.stt.warmup(),
            self.llm.warmup(),
            self.tts.warmup(),
        )
        print("✅ Models ready")
        
        if self.config.enable_batching:
            self._batch_task = asyncio.create_task(self._batch_worker())
    
    async def shutdown(self):
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
    ) -> AsyncIterator[bytes]:
        """
        Main pipeline: Audio → STT → LLM → TTS → Audio
        Yields audio chunks for playback.
        
        on_event(dict) — optional callback receiving live events:
          {"type": "transcript", "text": ..., "is_final": bool}
          {"type": "response", "text": <chunk so far>}
          {"type": "metrics", "stt_ms": ..., "llm_first_ms": ..., ...}
        """
        pipeline_start = time.perf_counter()
        budget = LatencyBudget(total_ms=self.config.target_latency_ms)
        metrics = PipelineMetrics()
        metrics.pipeline_start = pipeline_start

        # Per-turn latency logging, separate from `metrics`/`metrics_history`
        # above. Those exist for the single-shot callers (benchmark scripts,
        # the SDK example) that call run_pipeline() once per turn, so
        # `pipeline_start` really is "this turn's start" for them. Over a
        # live websocket session run_pipeline() is called ONCE for the whole
        # connection — `_llm_stage` loops over every turn internally — so
        # that end-of-call emit() a few lines down only ever fires at
        # disconnect, and its numbers are measured from session start, not
        # turn start. That's why no 📊 line has ever shown up in a live
        # session. Track turn-scoped timestamps here instead, off the
        # existing turn-boundary events ("transcript"/is_final starts a
        # turn, "response"/is_final ends it) — additive only, so it can't
        # change STT/LLM/TTS/barge-in behavior, only report on it.
        turn_t0: Optional[float] = None
        turn_first_token_ms: Optional[float] = None
        turn_first_audio_ms: Optional[float] = None

        def _fmt_ms(ms: Optional[float]) -> str:
            return f"{ms:.0f}ms" if ms is not None else "n/a"

        def emit(event: dict):
            nonlocal turn_t0, turn_first_token_ms, turn_first_audio_ms
            etype = event.get("type")
            now = time.perf_counter()
            turn_metrics_event = None
            if etype == "transcript" and event.get("is_final"):
                turn_t0 = now
                turn_first_token_ms = None
                turn_first_audio_ms = None
            elif etype == "response" and turn_t0 is not None:
                if turn_first_token_ms is None and event.get("text"):
                    turn_first_token_ms = (now - turn_t0) * 1000
                if event.get("is_final"):
                    llm_total_ms = (now - turn_t0) * 1000
                    tag = "interrupted" if event.get("interrupted") else "complete"
                    print(f"📊 turn: llm_first={_fmt_ms(turn_first_token_ms)} "
                          f"first_audio={_fmt_ms(turn_first_audio_ms)} "
                          f"llm_total={llm_total_ms:.0f}ms ({tag})")
                    turn_metrics_event = {
                        "type": "metrics",
                        "llm_first_ms": turn_first_token_ms or 0.0,
                        "first_audio_ms": turn_first_audio_ms or 0.0,
                        "llm_total_ms": llm_total_ms,
                    }
                    turn_t0 = None
            if on_event is not None:
                try:
                    on_event(event)
                except Exception:
                    pass
            if turn_metrics_event is not None:
                # Sent after the response's own is_final event (not before),
                # so a live client's transcript reads bot-reply-then-metrics
                # rather than metrics appearing to precede the reply it's
                # measuring.
                emit(turn_metrics_event)
        
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
        tee_task = asyncio.create_task(self._tee_audio(audio_stream, [main_q, watch_q]))
        watcher_task = asyncio.create_task(
            self._barge_in_watcher(self._drain(watch_q), barge_in, emit)
        )

        try:
            # Stage 1: VAD + STT Streaming
            stt_stream = self._stt_stage(
                self._drain(main_q), budget, metrics, emit, turn_state, stt_reset
            )

            # Stage 2: LLM Streaming (consumes STT partials)
            llm_stream = self._llm_stage(
                stt_stream, system_prompt, budget, metrics, emit, turn_state, stt_reset, barge_in
            )

            # Stage 3: TTS Streaming (consumes LLM tokens)
            tts_stream = self._tts_stage(llm_stream, budget, metrics)

            # Yield audio chunks
            first_audio = True
            async for audio_chunk in tts_stream:
                now = time.perf_counter()
                if first_audio:
                    metrics.tts_first_chunk_ms = (now - pipeline_start) * 1000
                    first_audio = False
                if turn_t0 is not None and turn_first_audio_ms is None:
                    turn_first_audio_ms = (now - turn_t0) * 1000
                yield audio_chunk
        finally:
            tee_task.cancel()
            watcher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await tee_task
            with contextlib.suppress(asyncio.CancelledError):
                await watcher_task

        metrics.e2e_latency_ms = (time.perf_counter() - pipeline_start) * 1000
        self.metrics_history.append(metrics)
        
        if emit:
            emit({
                "type": "metrics",
                "stt_ms": metrics.stt_latency_ms,
                "llm_first_ms": metrics.llm_first_token_ms,
                "tts_first_ms": metrics.tts_first_chunk_ms,
                "e2e_ms": metrics.e2e_latency_ms,
            })
        
        print(f"📊 Pipeline: STT={metrics.stt_latency_ms:.0f}ms "
              f"LLM_first={metrics.llm_first_token_ms:.0f}ms "
              f"TTS_first={metrics.tts_first_chunk_ms:.0f}ms "
              f"E2E={metrics.e2e_latency_ms:.0f}ms")
    
    @staticmethod
    async def _tee_audio(source: AsyncIterator[bytes], queues: List[asyncio.Queue]):
        """Fan one audio stream out to several queues so it can have more
        than one independent consumer (see `run_pipeline`)."""
        try:
            async for chunk in source:
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

        def load():
            import torch
            model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True,
                verbose=False,
            )
            return model

        try:
            return await asyncio.get_running_loop().run_in_executor(None, load)
        except Exception:
            return None

    async def _barge_in_watcher(
        self,
        audio_stream: AsyncIterator[bytes],
        barge_in: BargeInState,
        emit=None,
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
        async for chunk in audio_stream:
            buffer.extend(chunk)
            while len(buffer) >= bytes_per_frame:
                frame = bytes(buffer[:bytes_per_frame])
                buffer = buffer[bytes_per_frame:]

                if not barge_in.speaking:
                    speech_run_ms = 0.0
                    continue  # nothing to interrupt right now

                import torch
                audio_np = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
                with torch.no_grad():
                    prob = model(torch.from_numpy(audio_np).unsqueeze(0), sample_rate).item()

                if prob >= threshold:
                    speech_run_ms += frame_ms
                    if speech_run_ms >= min_speech_ms:
                        barge_in.fire()  # also stops re-firing until the next reply starts
                        print(f"⏹  barge-in: {speech_run_ms:.0f} ms of speech over the bot")
                        speech_run_ms = 0.0
                        if emit:
                            emit({"type": "interrupted"})
                else:
                    if speech_run_ms >= min_speech_ms / 2:
                        # Logged so a missed interruption leaves evidence: speech
                        # was heard over the bot, but broke off before the bar.
                        print(f"👂 heard {speech_run_ms:.0f} ms of speech over the bot "
                              f"(interrupting needs {min_speech_ms} ms unbroken)")
                    speech_run_ms = 0.0

    async def _stt_stage(
        self,
        audio_stream: AsyncIterator[bytes],
        budget: LatencyBudget,
        metrics: PipelineMetrics,
        emit=None,
        turn_state: Optional[TurnState] = None,
        stt_reset: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[STTResult]:
        """VAD + STT with streaming partial results."""
        stt_start = time.perf_counter()
        stt_budget = budget.allocate("stt", 100)

        # Apply VAD filter
        vad_filtered = self._apply_vad(audio_stream, turn_state)

        # Stream STT
        async for result in self.stt.transcribe_stream(
            vad_filtered, budget_ms=stt_budget.remaining_ms, reset_signal=stt_reset
        ):
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

        buffer = bytearray()
        async for chunk in audio_stream:
            buffer.extend(chunk)
            while len(buffer) >= bytes_per_frame:
                frame = bytes(buffer[:bytes_per_frame])
                buffer = buffer[bytes_per_frame:]

                audio_np = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
                import torch
                with torch.no_grad():
                    prob = model(torch.from_numpy(audio_np).unsqueeze(0), sample_rate).item()

                if turn_state is not None:
                    turn_state.vad_active = True
                    if prob >= threshold:
                        turn_state.silence_ms = 0.0
                    else:
                        turn_state.silence_ms += frame_ms

                if prob >= threshold:
                    yield frame
    
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
                    turn_ready.set()

        async def process_turn(transcript: str) -> AsyncIterator[str]:
            nonlocal first_token, response_buffer, last_bot_text
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

            async for llm_result in self.llm.generate_stream(
                messages, budget_ms=llm_budget.remaining_ms
            ):
                if barge_in is not None and barge_in.interrupted.is_set():
                    # The watcher caught real user speech starting while we
                    # were mid-reply — stop forwarding further tokens for
                    # this turn right away. The "interrupted" emit already
                    # went out from the watcher itself, not from here, so
                    # the client hears about it as early as possible.
                    barge_in.interrupted.clear()
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

                if llm_result.is_final:
                    metrics.llm_total_ms = (time.perf_counter() - llm_start) * 1000

                if llm_result.text:
                    response_buffer += llm_result.text

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
                print(f'🪞 discarded likely self-echo: "{candidate}"')
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
        metrics: PipelineMetrics
    ) -> AsyncIterator[bytes]:
        """TTS streaming from LLM tokens."""
        tts_start = time.perf_counter()
        first_chunk = True
        
        tts_budget = budget.allocate("tts", 100)
        
        async for tts_result in self.tts.synthesize_stream(
            llm_stream,
            budget_ms=tts_budget.remaining_ms
        ):
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
                print(f"Batch worker error: {e}")
    
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
