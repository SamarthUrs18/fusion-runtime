"""How one caller's audio and events get in and out: a browser's WebSocket, or a phone line.

A voice session is the same whoever carries it: the same limits, audio checks, interruption,
telemetry and pipeline (server._voice_session). What differs is the wire. A browser sends raw
16 kHz PCM and JSON control messages and hears our 24 kHz audio; a phone provider sends 8 kHz
μ-law inside its own JSON and plays what we send back to it. A Link turns either into:

    receive()       ("audio", pcm16 at the pipeline's rate, bytes on the wire)
                    ("control", {...}, bytes)    interrupt / playback, in the browser's terms
                    ("bad", None, bytes)         a message that couldn't be read
    send_audio()    the agent's audio, at the text-to-speech rate
    event()         what happened (transcripts, interruptions, problems): sent on, or acted on
    fail()          an error the caller's side should see, then close

A link that learns about playback another way than control messages (a phone provider's marks)
reports it through `controls`, the session's handler for them.
"""
import asyncio
import contextlib
import json
from typing import Any, Callable, Dict, Optional, Tuple

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

Received = Tuple[str, Any, int]


def client_gone(websocket: WebSocket, exc: Optional[BaseException] = None) -> bool:
    """True when a failure just means the other side disconnected, not that something broke."""
    if isinstance(exc, WebSocketDisconnect):
        return True
    if websocket.client_state != WebSocketState.CONNECTED or websocket.application_state != WebSocketState.CONNECTED:
        return True
    name = type(exc).__name__ if exc is not None else ""
    return name in ("ConnectionClosed", "ConnectionClosedOK", "ConnectionClosedError", "ClientDisconnected")


class Link:
    kind = "link"
    phone_line = False  # a dead phone line is hiss, not zeros: the session checks its level too

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.trace = None
        self.controls: Callable[[Dict], None] = lambda message: None  # set by the session

    def describe(self) -> Dict[str, Any]:
        """For session.start: what kind of caller this is. No content."""
        return {"transport": self.kind}

    async def open(self, session_id: str, trace, config: Dict[str, Any]) -> None:
        self.trace = trace

    async def receive(self) -> Received:
        raise NotImplementedError

    async def send_audio(self, pcm: bytes) -> None:
        raise NotImplementedError

    def event(self, event: Dict[str, Any]) -> None:
        pass

    async def fail(self, payload: Dict[str, Any], close_code: int) -> None:
        with contextlib.suppress(Exception):
            await self.websocket.close(code=close_code)

    def gone(self, exc: Optional[BaseException] = None) -> bool:
        return client_gone(self.websocket, exc)

    async def _message(self) -> Dict[str, Any]:
        message = await self.websocket.receive()
        if message["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(message.get("code", 1000))
        return message

    def _send_soon(self, payload: Dict[str, Any]) -> None:
        """Fire and forget. The socket may already be closing by the time this runs (the caller left
        mid-reply): that's swallowed, not surfaced as "Task exception was never retrieved"."""
        async def send():
            try:
                await self.websocket.send_json(payload)
            except (RuntimeError, WebSocketDisconnect):
                pass
            except Exception as e:
                if not self.gone(e) and self.trace is not None:
                    from fusion_runtime.telemetry import describe_error

                    self.trace.event("client.send_failed", level="warning", stage="server",
                                     error=describe_error(e, "server"), event_type=payload.get("type")
                                     or payload.get("event"))
        try:
            asyncio.get_running_loop().create_task(send())
        except RuntimeError:
            pass


class BrowserLink(Link):
    """fusion's own WebSocket protocol: what the browser client and `frun talk` speak."""

    kind = "websocket"

    def __init__(self, websocket: WebSocket, sample_rate: int):
        super().__init__(websocket)
        from fusion_runtime.engine.audio_format import Intake

        self.intake = Intake(sample_rate)

    async def open(self, session_id: str, trace, config: Dict[str, Any]) -> None:
        await super().open(session_id, trace, config)
        await self.websocket.send_json({"type": "config", **config, "session_id": session_id})

    async def receive(self) -> Received:
        message = await self._message()
        data = message.get("bytes")
        if data is not None:
            pcm = self.intake.accept(data)  # whole samples; a WAV header gone; BadAudioFormat otherwise
            if self.intake.note:
                self.trace.event("audio.format_fixed", level="warning", stage="audio", hint=self.intake.note)
                self.event({"type": "warning", "code": "wav_header", "message": self.intake.note})
                self.intake.note = None
            return "audio", pcm, len(data)
        text = message.get("text") or ""
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            return "bad", None, len(text)
        return ("control", parsed, len(text)) if isinstance(parsed, dict) else ("bad", None, len(text))

    async def send_audio(self, pcm: bytes) -> None:
        await self.websocket.send_bytes(pcm)

    def event(self, event: Dict[str, Any]) -> None:
        self._send_soon(event)

    async def fail(self, payload: Dict[str, Any], close_code: int) -> None:
        with contextlib.suppress(Exception):
            await self.websocket.send_json(payload)
        await super().fail(payload, close_code)
