#!/usr/bin/env python3
"""
Example: WebSocket client for real-time voice chat.

Microphone → server (VAD → STT → LLM → TTS) → speaker, full duplex: you can
talk over the bot and it stops.

Requirements: pip install -e ".[examples]"
NOTE: On macOS, allow Microphone access for your terminal app
      (System Settings → Privacy & Security → Microphone).

Echo
----
On laptop speakers the bot's own voice reaches the microphone about as loud
as you do. This client plays and records through
fusion_runtime.duplex_audio.DuplexAudio, which removes the bot's voice from
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

FUSION_AEC=0 turns echo cancellation off: the microphone is then held back
whenever the bot is audible (no interrupting, but no echo either).
"""
import asyncio
import json
import os
import sys
import time
from typing import Optional

import numpy as np
import websockets

try:
    from fusion_runtime.duplex_audio import DuplexAudio, MicChunk
except ImportError:  # running from a source checkout that isn't installed
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from fusion_runtime.duplex_audio import DuplexAudio, MicChunk

TTS_SAMPLE_RATE = 24000
AEC_ENABLED = os.environ.get("FUSION_AEC", "1") not in ("0", "false", "False")


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


class VoiceChatClient:
    def __init__(self, uri: str = "ws://localhost:8000/v1/voice/ws", echo_cancellation: bool = AEC_ENABLED):
        self.uri = uri
        self.audio = DuplexAudio(echo_cancellation=echo_cancellation)
        self.replies = ReplyGate()
        self.reporter = PlaybackReporter()
        self._last_meter = 0.0

    async def run(self):
        async with websockets.connect(self.uri, max_size=2**22) as ws:
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
        """A mic level bar ~4x/sec, so you can see the mic is alive."""
        now = time.monotonic()
        if now - self._last_meter < 0.25:
            return
        self._last_meter = now
        samples = np.frombuffer(chunk.pcm16, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples**2)))
        bar = "#" * min(int(rms / 32768 * 80), 40)
        if not chunk.safe_to_send:
            tag = "🔇"  # held back: echo possible and not cancelled yet
        elif self.audio.bot_audible:
            tag = "🔊"  # bot talking, mic open with its echo removed
        else:
            tag = "  "
        stats = self.audio.echo_stats
        echo = f"echo -{stats.erle_db:.0f}dB" if stats and stats.far_end_active else ""
        print(f"\r  mic {tag}|{bar:<40}| {echo:<12}", end="", flush=True)

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
            self.audio.flush()
            self.replies.on_interrupted()
            print("\n⏹  (you interrupted — bot stopped)")
        elif mtype == "echo_discarded":
            # The server decided a "user" turn was really the bot's own voice
            # and dropped it. Shown so a wrong call is visible too.
            print(f'\n🪞 (server discarded likely self-echo: "{msg.get("text", "")}")')
        elif mtype == "metrics":
            parts = [f"{k}={v:.0f}ms" for k, v in msg.items() if k != "type"]
            print(f"\n📊 {' '.join(parts)}")
        else:
            print(f"\n[server] {msg}")


async def main():
    client = VoiceChatClient()
    try:
        await client.run()
    except ConnectionRefusedError:
        print("❌ Server not running. Start it first:  make run-native")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Stopping...")
