"""
FastAPI Server - HTTP/WebSocket API for fusion-runtime
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from starlette.websockets import WebSocketState
from typing import Optional
import asyncio
import base64
import contextlib
import json
import os
import platform
import time
import uuid

from fusion_runtime import __version__
from fusion_runtime.config import DEVELOPMENT_CONFIG, PRODUCTION_CONFIG
from fusion_runtime.engine import BargeInState, PipelineOrchestrator
from fusion_runtime.telemetry import LoopMonitor, SessionTrace, describe_error, session_scope, telemetry


app = FastAPI(title="fusion-runtime", version=__version__)

# Global orchestrator (single worker)
orchestrator: Optional[PipelineOrchestrator] = None
_started_at = time.monotonic()
_active_sessions: set = set()
_loop_monitor: Optional[LoopMonitor] = None


@app.on_event("startup")
async def startup():
    global orchestrator, _loop_monitor, _started_at
    _started_at = time.monotonic()
    telemetry.configure_from_env()
    # Use development config by default (CPU-friendly small models),
    # production requires explicit FUSION_CONFIG=production
    config_name = os.getenv("FUSION_CONFIG", "development")
    config = {
        "development": DEVELOPMENT_CONFIG,
        "production": PRODUCTION_CONFIG,
    }.get(config_name, DEVELOPMENT_CONFIG)
    telemetry.emit(
        "server.start", stage="server", version=__version__, profile=config_name, pid=os.getpid(),
        python=platform.python_version(), platform=f"{platform.system()} {platform.machine()}",
        log_format=os.getenv("FUSION_LOG_FORMAT", "pretty"), log_content=telemetry.log_content,
    )
    _loop_monitor = LoopMonitor()
    _loop_monitor.start()

    orchestrator = PipelineOrchestrator(config)
    await orchestrator.initialize()
    telemetry.emit("server.ready", stage="server", duration_ms=(time.monotonic() - _started_at) * 1000)


@app.on_event("shutdown")
async def shutdown():
    global orchestrator
    telemetry.emit("server.stop", stage="server", uptime_s=round(time.monotonic() - _started_at, 1),
                   active_sessions=len(_active_sessions))
    if _loop_monitor is not None:
        await _loop_monitor.stop()
    if orchestrator:
        await orchestrator.shutdown()


# ============ Models ============

class VoiceChatRequest(BaseModel):
    audio_base64: str
    system_prompt: Optional[str] = None
    stt_provider: Optional[str] = None
    llm_provider: Optional[str] = None
    tts_provider: Optional[str] = None
    voice: Optional[str] = None


class VoiceChatResponse(BaseModel):
    audio_base64: str
    transcript: str
    response_text: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool
    uptime_s: float
    active_sessions: int
    config: dict


def _error_response(e: Exception, request_id: str) -> JSONResponse:
    info = describe_error(e)
    return JSONResponse(status_code=500, content={"error": {**info.for_client(), "request_id": request_id}})


# ============ REST Endpoints ============

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if orchestrator is not None else "starting",
        version=__version__,
        models_loaded=orchestrator is not None and getattr(orchestrator, "ready", False),
        uptime_s=round(time.monotonic() - _started_at, 1),
        active_sessions=len(_active_sessions),
        config={
            "stt": orchestrator.config.stt.provider.value if orchestrator else None,
            "llm": orchestrator.config.llm.provider.value if orchestrator else None,
            "tts": orchestrator.config.tts.provider.value if orchestrator else None,
        }
    )


@app.post("/v1/voice/chat", response_model=VoiceChatResponse)
async def voice_chat(request: VoiceChatRequest):
    """Single-turn voice chat (complete audio in/out)."""
    if not orchestrator:
        raise HTTPException(503, "Orchestrator not initialized")

    start = time.perf_counter()
    request_id = f"req-{uuid.uuid4().hex[:12]}"

    # Decode audio
    audio = base64.b64decode(request.audio_base64)

    # Override providers if specified (requires allow_cloud_fallback)
    if request.stt_provider or request.llm_provider or request.tts_provider:
        if not orchestrator.config.allow_cloud_fallback:
            raise HTTPException(400, "Cloud provider override requires allow_cloud_fallback=true")

    # Run pipeline
    from fusion_runtime.engine import run_single_turn
    with session_scope(request_id):
        try:
            output_audio = await run_single_turn(
                orchestrator,
                audio,
                request.system_prompt or "You are a helpful voice assistant."
            )
        except Exception as e:
            return _error_response(e, request_id)

    latency = (time.perf_counter() - start) * 1000

    return VoiceChatResponse(
        audio_base64=base64.b64encode(output_audio).decode(),
        transcript="",  # Would need to capture from pipeline
        response_text="",  # Would need to capture from pipeline
        latency_ms=latency,
    )


@app.post("/v1/voice/stream")
async def voice_stream(request: VoiceChatRequest):
    """Streaming voice chat - returns audio chunks as they're generated."""
    if not orchestrator:
        raise HTTPException(503, "Orchestrator not initialized")

    audio = base64.b64decode(request.audio_base64)
    request_id = f"req-{uuid.uuid4().hex[:12]}"

    async def audio_iterator():
        yield audio

    async def generate():
        with session_scope(request_id):
            async for chunk in orchestrator.run_pipeline(
                audio_iterator(),
                request.system_prompt or "You are a helpful voice assistant."
            ):
                yield chunk

    return StreamingResponse(
        generate(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(orchestrator.config.sample_rate),
            "X-Channels": str(orchestrator.config.channels),
            "X-Request-Id": request_id,
        }
    )


# ============ WebSocket Endpoint ============

def _client_gone(websocket: WebSocket, exc: Optional[BaseException] = None) -> bool:
    """True when a failure just means the client disconnected, not that something broke."""
    if isinstance(exc, WebSocketDisconnect):
        return True
    if websocket.client_state != WebSocketState.CONNECTED or websocket.application_state != WebSocketState.CONNECTED:
        return True
    name = type(exc).__name__ if exc is not None else ""
    return name in ("ConnectionClosed", "ConnectionClosedOK", "ConnectionClosedError", "ClientDisconnected")


@app.websocket("/v1/voice/ws")
async def voice_websocket(websocket: WebSocket):
    """WebSocket for real-time bidirectional voice chat."""
    await websocket.accept()

    session_id = str(uuid.uuid4())
    started = time.monotonic()
    end_reason = "client_disconnected"
    error_code = None
    trace = SessionTrace(session_id=session_id)
    _active_sessions.add(session_id)

    with session_scope(session_id):
        client = websocket.client
        telemetry.emit("session.start", stage="server",
                       client=f"{client.host}:{client.port}" if client else None,
                       profile=os.getenv("FUSION_CONFIG", "development"))
        receive_task = send_task = None
        try:
            # Send config
            await websocket.send_json({
                "type": "config",
                "sample_rate": orchestrator.config.sample_rate,
                "channels": orchestrator.config.channels,
                "session_id": session_id,
            })

            # Audio buffer for incoming stream
            audio_queue: asyncio.Queue = asyncio.Queue()
            barge_in = BargeInState()

            def _handle_control(raw: str):
                """Control messages the client sends alongside the audio stream."""
                try:
                    msg = json.loads(raw)
                except (ValueError, TypeError):
                    trace.event("client.bad_message", level="warning", stage="server", bytes=len(raw))
                    return
                if msg.get("type") == "interrupt":
                    # A client that detects interruptions itself, and has already
                    # stopped its own playback. Cancel generation so we stop
                    # producing a reply nobody is listening to any more.
                    barge_in.fire()
                    turn = trace.responding
                    if turn is not None:
                        turn.mark("barge_in_fired")
                    trace.event("barge_in.fired", turn=turn, stage="barge_in", source="client")
                elif msg.get("type") == "playback":
                    # The client reports whether bot audio is still coming out of
                    # its speaker. Playback usually outlasts generation, and the
                    # user has to be able to interrupt until the speaker actually
                    # goes quiet.
                    playing = bool(msg.get("playing"))
                    barge_in.set_playing(playing)
                    turn = trace.responding
                    if playing and turn is not None and "playback_started" not in turn.marks:
                        turn.mark("playback_started")
                    trace.event("client.playback", turn=turn, level="debug", stage="audio", playing=playing)
                else:
                    trace.event("client.unknown_message", level="debug", stage="server", message_type=msg.get("type"))

            async def receive_audio():
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        raise WebSocketDisconnect(message.get("code", 1000))
                    data = message.get("bytes")
                    if data is not None:
                        await audio_queue.put(data)
                        continue
                    text = message.get("text")
                    if text is not None:
                        _handle_control(text)

            async def audio_stream():
                # asyncio.Queue has no __aiter__ — wrap it
                while True:
                    chunk = await audio_queue.get()
                    yield chunk

            def _dispatch_event(ws, event):
                # Fire-and-forget event send (transcripts, responses, traces).
                # The socket may already be closing by the time this task runs
                # (client disconnected mid-pipeline) — swallow that quietly
                # instead of letting it surface as "Task exception was never
                # retrieved".
                async def _send():
                    try:
                        await ws.send_json(event)
                    except (RuntimeError, WebSocketDisconnect):
                        pass
                    except Exception as e:
                        if not _client_gone(ws, e):
                            trace.event("client.send_failed", level="warning", stage="server",
                                        error=describe_error(e, "server"), event_type=event.get("type"))
                try:
                    asyncio.get_running_loop().create_task(_send())
                except RuntimeError:
                    pass

            async def send_audio():
                pipeline = orchestrator.run_pipeline(
                    audio_stream(),
                    "You are a helpful voice assistant. Answer briefly.",
                    on_event=lambda event: _dispatch_event(websocket, event),
                    barge_in=barge_in,
                    trace=trace,
                )
                try:
                    async for chunk in pipeline:
                        await websocket.send_bytes(chunk)
                finally:
                    # Close the pipeline now (not whenever it gets garbage collected), so its
                    # tasks, per-session VAD models and decode threads are released.
                    await pipeline.aclose()

            # Whichever side ends first ends the session. Waiting on the sender alone
            # left sessions running forever after a disconnect: the pipeline kept
            # waiting for audio that would never come.
            receive_task = asyncio.create_task(receive_audio())
            send_task = asyncio.create_task(send_audio())
            done, _ = await asyncio.wait({receive_task, send_task}, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                exc = task.exception()
                if exc is not None and not _client_gone(websocket, exc):
                    raise exc

        except Exception as e:
            if _client_gone(websocket, e):
                end_reason = "client_disconnected"
            else:
                end_reason = "error"
                info = describe_error(e)
                error_code = info.code
                if not getattr(e, "fusion_reported", False):  # pipeline errors are already logged and counted
                    telemetry.emit("session.error", level="error", stage="server", error=info)
                with contextlib.suppress(Exception):
                    await websocket.send_json({"type": "error", "session_id": session_id,
                                               "turn_id": (trace.responding or trace.listening).turn_id
                                               if (trace.responding or trace.listening) else None,
                                               **info.for_client()})
                with contextlib.suppress(Exception):
                    await websocket.close(code=1011)
        finally:
            for task in (send_task, receive_task):
                if task is not None and not task.done():
                    task.cancel()
                    with contextlib.suppress(BaseException):
                        await task
            trace.finish()
            _active_sessions.discard(session_id)
            telemetry.emit("session.end", stage="server", reason=end_reason, error_code=error_code,
                           duration_s=round(time.monotonic() - started, 2), turns=trace.turn_count)


# ============ Metrics Endpoints ============

@app.get("/metrics")
async def metrics():
    """Prometheus metrics: latency histograms, turns, errors, sessions, event-loop lag, memory, CPU."""
    return Response(telemetry.metrics.render(), media_type=telemetry.metrics.content_type)


@app.get("/v1/metrics/summary")
async def metrics_summary():
    """P50/P99 of recent single-shot pipeline runs (REST and library use)."""
    if not orchestrator:
        raise HTTPException(503, "Orchestrator not initialized")
    return orchestrator.get_metrics_summary()


# ============ Main ============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fusion_runtime.server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker - scaling via container replication
        loop="uvloop",
    )
