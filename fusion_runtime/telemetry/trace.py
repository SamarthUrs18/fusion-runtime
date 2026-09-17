"""Per-conversation and per-turn timelines.

A turn goes through two phases that can overlap with the next turn:
  listening   user speech arrives: VAD, partial transcripts, turn end detected
  responding  LLM → TTS → audio out, possibly interrupted
While the bot is still responding, the user may already be speaking the next
turn, so a session keeps a `listening` turn and a `responding` turn separately.

Two clocks:
  processing clock  when the code handled something (model latencies)
  arrival clock     when the server received a given audio sample. Audio can
                    queue behind speech-to-text or VAD, so "speech ended" is the
                    arrival time of that sample, not when it was processed, or
                    latencies look wrong exactly when things are slow. (Sample
                    position x sample rate would drift from real time over a long
                    call; arrival times don't.) Only meaningful when audio streams
                    in real time (a live client); for audio sent faster than real
                    time (a file), speech-relative numbers such as TTFA are left
                    out rather than reported wrong. Speech *durations* come from
                    sample counts, which are right either way.
"""
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from fusion_runtime.telemetry.events import ErrorInfo, Event
from fusion_runtime.telemetry.hub import Telemetry, current_session_id, telemetry as default_hub


@dataclass
class TurnTrace:
    turn_id: str
    marks: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # name -> (monotonic, wall)
    counts: Dict[str, float] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)
    finished: bool = False

    def mark(self, name: str, at_mono: Optional[float] = None, overwrite: bool = False) -> float:
        now_mono = time.monotonic()
        at_mono = now_mono if at_mono is None else at_mono
        if overwrite or name not in self.marks:
            self.marks[name] = (at_mono, time.time() - (now_mono - at_mono))
        return self.marks[name][0]

    def add(self, name: str, amount: float = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + amount

    def at(self, name: str) -> Optional[float]:
        return self.marks[name][0] if name in self.marks else None

    def between_ms(self, start: str, end: str) -> Optional[float]:
        a, b = self.at(start), self.at(end)
        return None if a is None or b is None else (b - a) * 1000

    @property
    def started_mono(self) -> Optional[float]:
        return min((m for m, _ in self.marks.values()), default=None)


class SessionTrace:
    SAMPLE_RATE = 16000

    def __init__(
        self,
        session_id: Optional[str] = None,
        hub: Optional[Telemetry] = None,
        on_turn_trace: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.hub = hub or default_hub
        self.session_id = session_id or current_session_id() or f"local-{uuid.uuid4().hex[:8]}"
        self.on_turn_trace = on_turn_trace
        self.started_mono = time.monotonic()
        self.turn_count = 0
        self.listening: Optional[TurnTrace] = None
        self.responding: Optional[TurnTrace] = None
        # arrival clock: (first sample index of chunk, monotonic arrival time), recent chunks only
        self._arrivals: deque = deque(maxlen=4000)  # ~80 s of 20 ms chunks
        self._audio_t0: Optional[float] = None
        self._samples_received = 0
        self._last_arrival: Optional[float] = None

    # ---- audio clock --------------------------------------------------------------

    def audio_received(self, num_bytes: int, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        if self._audio_t0 is None:
            self._audio_t0 = now
        self._arrivals.append((self._samples_received, now))
        self._samples_received += num_bytes // 2
        self._last_arrival = now

    def arrival_time(self, sample_index: int) -> Optional[float]:
        """When the server received the chunk containing `sample_index` (None if unknown)."""
        for first_sample, arrived in reversed(self._arrivals):
            if first_sample <= sample_index:
                return arrived
        return self._arrivals[0][1] if self._arrivals else None

    @property
    def realtime_audio(self) -> bool:
        """True when audio arrives at roughly speaking speed (a live stream), so the audio clock is meaningful."""
        if self._audio_t0 is None or self._last_arrival is None or self._samples_received == 0:
            return False
        audio_s = self._samples_received / self.SAMPLE_RATE
        if audio_s < 0.5:
            return True  # too little audio to tell; assume live
        return (self._last_arrival - self._audio_t0) >= audio_s * 0.5

    # ---- turns --------------------------------------------------------------------

    def listening_turn(self) -> TurnTrace:
        if self.listening is None:
            self.turn_count += 1
            self.listening = TurnTrace(turn_id=f"t{self.turn_count}")
        return self.listening

    def start_responding(self) -> TurnTrace:
        """The turn that was being listened to now gets a reply."""
        turn = self.listening_turn()
        self.listening = None
        self.responding = turn
        return turn

    def event(
        self,
        name: str,
        turn: Optional[TurnTrace] = None,
        level: str = "info",
        stage: Optional[str] = None,
        duration_ms: Optional[float] = None,
        error: Optional[ErrorInfo] = None,
        request_id: Optional[str] = None,
        **attrs: Any,
    ) -> Event:
        return self.hub.emit(
            name, level=level, stage=stage, session_id=self.session_id,
            turn_id=turn.turn_id if turn is not None else None, request_id=request_id,
            duration_ms=duration_ms, error=error, **attrs,
        )

    def end_turn(self, turn: TurnTrace, outcome: str) -> Dict[str, Any]:
        """Close a turn: emit turn.summary and hand the full trace to on_turn_trace (e.g. the WebSocket)."""
        if turn.finished:
            return {}
        turn.finished = True
        turn.mark("turn_end")
        if self.listening is turn:
            self.listening = None
        if self.responding is turn:
            self.responding = None
        summary = self.summarize(turn, outcome)
        self.event("turn.summary", turn=turn, stage="turn", duration_ms=summary.get("total_ms"), **summary)
        trace = {
            "session_id": self.session_id,
            "turn_id": turn.turn_id,
            "summary": summary,
            "timeline": self.timeline(turn),
        }
        if self.on_turn_trace is not None:
            try:
                self.on_turn_trace(trace)
            except Exception:
                pass
        return trace

    def finish(self) -> None:
        """Session over: close anything still open."""
        for turn, outcome in ((self.responding, "unfinished"), (self.listening, "no_transcript")):
            if turn is not None and turn.marks:
                self.end_turn(turn, outcome)

    # ---- derived numbers --------------------------------------------------------------

    def summarize(self, turn: TurnTrace, outcome: str) -> Dict[str, Any]:
        realtime = self.realtime_audio
        c, i = turn.counts, turn.info

        def rounded(value: Optional[float]) -> Optional[float]:
            return None if value is None else round(value, 1)

        llm_duration = turn.between_ms("llm_request", "llm_done")
        tokens = int(c.get("llm_tokens", 0))
        decode_ms = c.get("llm_decode_ms")  # time waiting on the model after the first token, not wall clock
        tokens_per_s = (tokens - 1) / (decode_ms / 1000) if decode_ms and tokens > 1 else None
        tts_audio_s = c.get("tts_audio_s", 0.0)
        tts_synth_ms = c.get("tts_synth_ms", 0.0)

        summary: Dict[str, Any] = {
            "outcome": outcome,
            "realtime_audio": realtime,
            # listening
            "speech_ms": rounded(c.get("speech_audio_ms")),
            "speech_segments": int(c.get("speech_segments", 0)) or None,
            "stt_windows": int(c.get("stt_windows", 0)) or None,
            "stt_transcribe_ms": rounded(c.get("stt_transcribe_ms")),
            "stt_first_partial_ms": rounded(turn.between_ms("speech_start", "stt_first_partial")) if realtime else None,
            # 0 when the transcript was already up to date by the time speech ended
            "transcription_delay_ms": rounded(max(0.0, turn.between_ms("speech_end", "stt_last_partial")))
            if realtime and turn.between_ms("speech_end", "stt_last_partial") is not None else None,
            "end_of_turn_wait_ms": rounded(turn.between_ms("speech_end", "turn_end_detected")) if realtime else None,
            "turn_end_reason": i.get("turn_end_reason"),
            # responding
            "llm_queue_ms": rounded(c.get("llm_queue_ms")) if c.get("llm_queue_ms", 0) >= 1 else None,
            "llm_first_token_ms": rounded(turn.between_ms("llm_request", "llm_first_token")),
            "llm_duration_ms": rounded(llm_duration),
            "llm_tokens": tokens or None,
            "llm_tokens_per_second": rounded(tokens_per_s),
            "llm_finish": i.get("llm_finish"),
            "tts_first_chunk_ms": rounded(turn.between_ms("llm_first_token", "tts_first_chunk")),
            "tts_chunks": int(c.get("tts_chunks", 0)) or None,
            "tts_audio_s": round(tts_audio_s, 2) if tts_audio_s else None,
            "tts_synth_ms": rounded(tts_synth_ms) if tts_synth_ms else None,
            "tts_rtf": round(tts_synth_ms / 1000 / tts_audio_s, 3) if tts_audio_s else None,
            "response_ms": rounded(turn.between_ms("turn_end_detected", "audio_first_sent")),
            "ttfa_ms": rounded(turn.between_ms("speech_end", "audio_first_sent")) if realtime else None,
            "playback_delay_ms": rounded(turn.between_ms("audio_first_sent", "playback_started")),
            # interruption
            "interrupted": "barge_in_fired" in turn.marks,
            "barge_in_speech_ms": rounded(i.get("barge_in_speech_ms")),
            "interruption_stop_ms": rounded(turn.between_ms("barge_in_fired", "llm_stopped")),
            "errors": int(c.get("errors", 0)) or None,
        }
        start = turn.started_mono
        if start is not None:
            end = turn.at("turn_end") or time.monotonic()  # still open: measure up to now
            summary["total_ms"] = round((end - start) * 1000, 1)
        return {k: v for k, v in summary.items() if v is not None}

    @staticmethod
    def timeline(turn: TurnTrace) -> List[Dict[str, Any]]:
        start = turn.started_mono or 0.0
        from datetime import datetime, timezone

        ordered = sorted(turn.marks.items(), key=lambda item: item[1][0])
        return [
            {
                "event": name,
                "t_ms": round((mono - start) * 1000, 1),
                "ts": datetime.fromtimestamp(wall, tz=timezone.utc).isoformat(timespec="milliseconds"),
            }
            for name, (mono, wall) in ordered
        ]
