"""The `frun talk` client: microphone → server → speaker, full duplex.

You can talk over the bot and it stops.

Echo
----
On laptop speakers the bot's own voice reaches the microphone about as loud
as you do. This client plays and records through
fusion_runtime.audio.duplex_audio.DuplexAudio, which removes the bot's voice from
the microphone on this device — using the exact audio it just played — before
anything is sent. For the first moment of the first reply, while the
canceller is still learning the room, microphone audio that could contain
echo is held back instead of sent (shown as 🔇).

Interruptions are decided on the server. It receives clean audio, so when you
talk over the bot its voice activity detector hears you and stops the reply.
This client tells the server while the bot is audible, and stops playback the
moment the server reports an interruption.

On a new machine, `python3 scripts/measure_echo.py` first shows how much echo
the canceller removes on your speakers.

Imported only when `frun talk` runs: it loads numpy, websockets and the audio stack.
"""
import asyncio
import json
import sys
import time
from typing import Optional

import numpy as np
import websockets

from fusion_runtime.audio.duplex_audio import DuplexAudio, MicChunk

TTS_SAMPLE_RATE = 24000
DEFAULT_URL = "ws://127.0.0.1:8000/v1/voice/ws"  # see fusion_runtime.cli.talk for why not "localhost"


class ReplyGate:
    """Decides whether bot audio arriving from the server should be played.

    After an interruption the server can still send the tail of the reply it
    was producing (a sentence already being synthesized). That audio is
    dropped until the next turn starts, so the bot doesn't carry on talking
    over the user.
    """

    def __init__(self):
        self._discarding = False

    def on_interrupted(self):
        self._discarding = True

    def on_new_turn(self):
        self._discarding = False

    def should_play(self) -> bool:
        return not self._discarding


class PlaybackReporter:
    """Produces a control message whenever bot audio starts or stops being
    audible, once per change — so the server keeps listening for
    interruptions until the speaker actually goes quiet, not just until it
    has finished generating the reply."""

    def __init__(self):
        self._playing = False

    def update(self, playing: bool) -> Optional[str]:
        if playing == self._playing:
            return None
        self._playing = playing
        return json.dumps({"type": "playback", "playing": playing})


def format_turn_summary(summary: dict) -> str:
    """One line per turn: the numbers that explain how it felt."""
    def ms(key: str) -> str:
        value = summary.get(key)
        return f"{value:.0f}ms" if value is not None else "n/a"

    parts = []
    if summary.get("ttfa_ms") is not None:
        parts.append(f"TTFA {ms('ttfa_ms')}")
    parts.append(f"response {ms('response_ms')}")
    if summary.get("end_of_turn_wait_ms") is not None:
        parts.append(f"end-of-turn {ms('end_of_turn_wait_ms')}")
    if summary.get("stt_transcribe_ms") is not None:
        parts.append(f"stt {ms('stt_transcribe_ms')}")
    if summary.get("llm_queue_ms") is not None:
        parts.append(f"queued {ms('llm_queue_ms')}")
    if summary.get("llm_first_token_ms") is not None:
        rate = f" {summary['llm_tokens_per_second']:.0f} tok/s" if summary.get("llm_tokens_per_second") else ""
        parts.append(f"llm {ms('llm_first_token_ms')}{rate}")
    if summary.get("tts_first_chunk_ms") is not None:
        parts.append(f"tts {ms('tts_first_chunk_ms')}")
    if summary.get("playback_delay_ms") is not None:
        parts.append(f"playback +{ms('playback_delay_ms')}")
    if summary.get("interrupted"):
        parts.append(f"interrupted (stopped in {ms('interruption_stop_ms')})")
    outcome = summary.get("outcome", "?")
    return f"📊 {outcome}: " + " · ".join(parts)


def format_timeline(timeline: list) -> str:
    return "\n".join(f"     {step['t_ms']:>8.0f} ms  {step['event']}" for step in timeline)


def format_error(msg: dict) -> str:
    stage = f"[{msg['stage']}] " if msg.get("stage") else ""
    line = f"❌ {stage}{msg.get('code', 'error')}: {msg.get('message', '')}"
    if msg.get("retryable"):
        line += " (retryable)"
    if msg.get("fix"):
        line += f"\n   → {msg['fix']}"
    return line


class VoiceChatClient:
    def __init__(self, uri: str = DEFAULT_URL, echo_cancellation: bool = True, verbose: bool = False,
                 key: Optional[str] = None):
        self.uri = uri
        self.verbose = verbose
        self.key = key
        self.audio = DuplexAudio(echo_cancellation=echo_cancellation)
        self.replies = ReplyGate()
        self.reporter = PlaybackReporter()
        self._last_meter = 0.0
        self._last_state = None
        # A live bar needs a terminal; anywhere else it would print one line per redraw.
        self.redraws = sys.stdout.isatty()

    async def run(self):
        # Unlike a browser, this client can set a header and can hold a key, so it
        # sends the key itself — no session token round trip.
        options = {"max_size": 2**22}
        if self.key:
            # websockets renamed this in 14.0; we support both.
            import inspect

            parameters = inspect.signature(websockets.connect).parameters
            name = "additional_headers" if "additional_headers" in parameters else "extra_headers"
            options[name] = {"Authorization": f"Bearer {self.key}"}
        async with websockets.connect(self.uri, **options) as ws:
            config = json.loads(await ws.recv())
            print(f"Connected: {config}")
            if self.audio.echo_cancellation:
                print("🎤 Speak whenever you like — you can talk over the bot. Ctrl+C to quit.\n")
            else:
                print("🎤 Echo cancellation is off: the mic is held back while the bot talks. Ctrl+C to quit.\n")

            self.audio.start()
            tasks = [asyncio.create_task(self._send(ws)), asyncio.create_task(self._receive(ws))]
            try:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
                for task in pending:
                    task.cancel()
                for task in done:
                    task.result()  # surface whatever ended the session
            finally:
                self.audio.stop()

    async def _send(self, ws):
        loop = asyncio.get_running_loop()
        while True:
            chunk = await loop.run_in_executor(None, self.audio.read, 0.1)
            change = self.reporter.update(self.audio.bot_audible)
            if change is not None:
                await ws.send(change)
            if chunk is None:
                continue
            self._show_meter(chunk)
            if chunk.safe_to_send:
                await ws.send(chunk.pcm16)

    async def _receive(self, ws):
        while True:
            data = await ws.recv()
            if isinstance(data, bytes):
                if self.replies.should_play():
                    self.audio.play(data, TTS_SAMPLE_RATE)
            else:
                self._handle_server_message(json.loads(data))

    def _show_meter(self, chunk: MicChunk):
        """A mic level bar ~4x/sec, so you can see the mic is alive.

        The bar redraws itself with a carriage return, which needs a terminal.
        Somewhere that only captures lines — a pipe, a log file, an editor's
        output pane — every redraw would land as another line, and one turn fills
        the screen with meters. There, only a change of state is worth saying.
        """
        now = time.monotonic()
        if now - self._last_meter < 0.25:
            return
        self._last_meter = now
        samples = np.frombuffer(chunk.pcm16, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples**2)))
        bar = "#" * min(int(rms / 32768 * 80), 40)
        if not chunk.safe_to_send:
            tag, state = "🔇", "mic held back while the canceller learns the room"
        elif self.audio.bot_audible:
            tag, state = "🔊", "bot talking, mic open with its echo removed"
        else:
            tag, state = "  ", "listening"
        stats = self.audio.echo_stats
        echo = f"echo -{stats.erle_db:.0f}dB" if stats and stats.far_end_active else ""
        if self.redraws:
            print(f"\r  mic {tag}|{bar:<40}| {echo:<12}", end="", flush=True)
        elif state != self._last_state:
            self._last_state = state
            print(f"  mic: {state}" + (f" ({echo})" if echo else ""), flush=True)

    def _handle_server_message(self, msg: dict):
        mtype = msg.get("type")
        if mtype == "transcript":
            if msg.get("is_final"):
                self.replies.on_new_turn()
                print(f"\n🗣  You: {msg['text']}")
        elif mtype == "response":
            # One "response" event arrives per LLM token; print only the
            # finished reply, so it doesn't fight the mic bar for the line.
            if msg.get("is_final"):
                cut = " (cut off — you interrupted)" if msg.get("interrupted") else ""
                print(f"\n🤖 Bot: {msg['text']}{cut}")
        elif mtype == "interrupted":
            started = time.perf_counter()
            self.audio.flush()
            flush_ms = (time.perf_counter() - started) * 1000
            self.replies.on_interrupted()
            print(f"\n⏹  (you interrupted — bot stopped, speaker cleared in {flush_ms:.1f} ms)")
        elif mtype == "echo_discarded":
            # The server decided a "user" turn was really the bot's own voice
            # and dropped it. Shown so a wrong call is visible too.
            print(f'\n🪞 (server discarded likely self-echo: "{msg.get("text", "")}")')
        elif mtype == "turn_resumed":
            # The caller kept talking right after a pause: the cut-off reply is dropped and
            # what they said before and after the pause is answered as one turn.
            print("\n↪  (you kept talking, so both parts are answered together)")
        elif mtype == "turn.trace":
            print(f"\n{format_turn_summary(msg.get('summary', {}))}")
            if self.verbose and msg.get("timeline"):
                print(format_timeline(msg["timeline"]))
        elif mtype == "error":
            print(f"\n{format_error(msg)}")
        else:
            print(f"\n[server] {msg}")
