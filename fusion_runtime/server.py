"""
FastAPI Server - HTTP/WebSocket API for fusion-runtime
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, AsyncIterator
import asyncio
import base64
import json
import uuid

from fusion_runtime import __version__
from fusion_runtime.config import PipelineConfig, DEVELOPMENT_CONFIG, PRODUCTION_CONFIG
from fusion_runtime.engine import BargeInState, PipelineOrchestrator, PipelineMetrics


app = FastAPI(title="fusion-runtime", version=__version__)

# Global orchestrator (single worker)
orchestrator: Optional[PipelineOrchestrator] = None


@app.on_event("startup")
async def startup():
    global orchestrator
    # Use development config by default (CPU-friendly small models),
    # production requires explicit FUSION_CONFIG=production
    import os
    config_name = os.getenv("FUSION_CONFIG", "development")
    config = {
        "development": DEVELOPMENT_CONFIG,
        "production": PRODUCTION_CONFIG,
    }.get(config_name, DEVELOPMENT_CONFIG)
    
    orchestrator = PipelineOrchestrator(config)
    await orchestrator.initialize()


@app.on_event("shutdown")
async def shutdown():
    global orchestrator
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
    models_loaded: bool
    config: dict


# ============ REST Endpoints ============

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        models_loaded=orchestrator is not None and orchestrator.stt._warm,
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
    
    import time
    start = time.perf_counter()
    
    # Decode audio
    audio = base64.b64decode(request.audio_base64)
    
    # Override providers if specified (requires allow_cloud_fallback)
    if request.stt_provider or request.llm_provider or request.tts_provider:
        if not orchestrator.config.allow_cloud_fallback:
            raise HTTPException(400, "Cloud provider override requires allow_cloud_fallback=true")
    
    # Run pipeline
    from fusion_runtime.engine import run_single_turn
    output_audio = await run_single_turn(
        orchestrator,
        audio,
        request.system_prompt or "You are a helpful voice assistant."
    )
    
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
    
    async def audio_iterator():
        yield audio
    
    async def generate():
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
        }
    )


# ============ WebSocket Endpoint ============

@app.websocket("/v1/voice/ws")
async def voice_websocket(websocket: WebSocket):
    """WebSocket for real-time bidirectional voice chat."""
    await websocket.accept()
    
    session_id = str(uuid.uuid4())
    print(f"🔌 WS connected: {session_id}")
    
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
                return
            if msg.get("type") == "interrupt":
                # A client that detects interruptions itself, and has already
                # stopped its own playback. Cancel generation so we stop
                # producing a reply nobody is listening to any more.
                barge_in.fire()
                print(f"⏹  barge-in from client: {session_id}")
            elif msg.get("type") == "playback":
                # The client reports whether bot audio is still coming out of
                # its speaker. Playback usually outlasts generation, and the
                # user has to be able to interrupt until the speaker actually
                # goes quiet.
                barge_in.set_playing(bool(msg.get("playing")))

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

        async def send_audio():
            async for chunk in orchestrator.run_pipeline(
                audio_stream(),
                "You are a helpful voice assistant. Answer briefly.",
                on_event=lambda event: _dispatch_event(websocket, event),
                barge_in=barge_in,
            ):
                await websocket.send_bytes(chunk)
        
        def _dispatch_event(ws, event):
            # Fire-and-forget event send (transcripts, responses, metrics).
            # The socket may already be closing by the time this task runs
            # (client disconnected mid-pipeline) — swallow that quietly
            # instead of letting it surface as "Task exception was never
            # retrieved".
            async def _send():
                try:
                    await ws.send_json(event)
                except (RuntimeError, WebSocketDisconnect):
                    pass
            try:
                asyncio.get_running_loop().create_task(_send())
            except RuntimeError:
                pass

        # Run both concurrently; receive task keeps the connection alive
        receive_task = asyncio.create_task(receive_audio())
        send_task = asyncio.create_task(send_audio())
        try:
            await send_task
        except (RuntimeError, WebSocketDisconnect):
            # Client went away mid-stream (e.g. Ctrl+C) — nothing to do.
            pass
        finally:
            receive_task.cancel()
            try:
                await receive_task
            except (asyncio.CancelledError, RuntimeError, WebSocketDisconnect):
                # Swallow so its exception doesn't surface later as
                # "Task exception was never retrieved".
                pass

    except WebSocketDisconnect:
        print(f"🔌 WS disconnected: {session_id}")
    except Exception as e:
        print(f"WS error: {e}")
        try:
            await websocket.close(code=1011)
        except RuntimeError:
            # Already closed (e.g. the disconnect that caused `e` also
            # tore down the socket) — closing again just re-raises.
            pass


# ============ Metrics Endpoint ============

@app.get("/metrics")
async def metrics():
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