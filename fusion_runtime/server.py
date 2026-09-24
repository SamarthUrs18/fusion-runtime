"""
FastAPI Server - HTTP/WebSocket API for fusion-runtime
"""
import asyncio
import base64
import contextlib
import json
import os
import platform
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
from starlette.websockets import WebSocketState

from fusion_runtime import __version__, web
from fusion_runtime.agent import DEFAULT_PROMPT, Agent, load_agent
from fusion_runtime.config import load_profile
from fusion_runtime.engine import BargeInState, PipelineOrchestrator
from fusion_runtime.env import load_env_file
from fusion_runtime.security import (
    ALLOWED_ORIGINS_ENV,
    Authenticator,
    ConfigurationError,
    KeySet,
    OriginRule,
    Principal,
    ProxyTrust,
    TokenStore,
    Unauthorized,
    scrub_access_logs,
)
from fusion_runtime.security.limits import (
    AudioBudget,
    ConnectionRate,
    Limits,
    OverLimit,
    SessionSlots,
)
from fusion_runtime.telemetry import (
    LoopMonitor,
    SessionTrace,
    describe_error,
    session_scope,
    telemetry,
)

app = FastAPI(title="fusion-runtime", version=__version__)

# Global orchestrator (single worker)
orchestrator: Optional[PipelineOrchestrator] = None
auth: Authenticator = Authenticator(KeySet(), TokenStore())  # replaced at startup
limits: Limits = Limits()
rate: ConnectionRate = ConnectionRate(limits.connections_per_minute)  # new sockets, per address
auth_failures: ConnectionRate = ConnectionRate(limits.connections_per_minute)  # guessing, per address
mints: ConnectionRate = ConnectionRate(limits.tokens_per_minute)  # tokens minted, per key
slots: SessionSlots = SessionSlots(limits)
proxies: ProxyTrust = ProxyTrust()
origins: OriginRule = OriginRule()
_live_sessions: dict = {}  # session id -> (key fingerprint, socket), so a revoked key can be cut off
agent: Optional["Agent"] = None  # set when the server was started with an agent file
_started_at = time.monotonic()
_active_sessions: set = set()
_loop_monitor: Optional[LoopMonitor] = None


@app.on_event("startup")
async def startup():
    global orchestrator, agent, _loop_monitor, _started_at
    _started_at = time.monotonic()
    agent_path = os.getenv("FUSION_AGENT")
    directories = [Path.cwd()] + ([Path(agent_path).expanduser().parent] if agent_path else [])
    loaded_env = load_env_file(directories)  # names only are logged; values are secrets
    telemetry.configure_from_env()
    _configure_auth()
    _listen_for_reload()
    # An agent file describes everything; without one, a profile of defaults.
    # production requires explicit FUSION_CONFIG=production
    config_name = os.getenv("FUSION_CONFIG", "development")
    if loaded_env:
        telemetry.emit("env.loaded", stage="server", variables=sorted(loaded_env),
                       hint="from a .env file; values are never logged")
    agent_path = os.getenv("FUSION_AGENT")
    if agent_path:
        agent = load_agent(agent_path)
        config = agent.config()  # plus FUSION_LLM_URL / _MODEL / _TURN_* overrides
        telemetry.emit("agent.loaded", stage="server", **agent.describe())
    else:
        agent = None
        config = load_profile(config_name)
    telemetry.emit(
        "server.start", stage="server", version=__version__, profile=config_name, pid=os.getpid(),
        python=platform.python_version(), platform=f"{platform.system()} {platform.machine()}",
        log_format=os.getenv("FUSION_LOG_FORMAT", "pretty"), log_content=telemetry.log_content,
    )
    _loop_monitor = LoopMonitor()
    _loop_monitor.start()

    orchestrator = PipelineOrchestrator(config)
    await orchestrator.initialize()
    if agent is not None and agent.tools:
        orchestrator.check_tools(agent.tools)  # an LLM that can't call them fails now, not mid-call
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


class Turn(BaseModel):
    """One exchange: what the caller said, what the agent answered, and how long each part took."""

    user: str
    agent: str
    outcome: str  # completed | interrupted | echo_discarded
    metrics: dict  # the same numbers as the per-turn telemetry summary (TTFA, stt, llm, tts, ...)


class VoiceChatResponse(BaseModel):
    audio_base64: str
    transcript: str  # everything the caller said, turns joined
    response_text: str  # everything the agent answered
    latency_ms: float
    turns: list[Turn] = []


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


def _configure_auth(environ=None) -> None:
    """Read the keys and the limits once, at startup. Bad keys stop the server
    here rather than leaving it running with authentication that doesn't work."""
    global auth, limits, rate, auth_failures, mints, slots, proxies, origins

    from fusion_runtime.config import TOKEN_TTL_ENV
    from fusion_runtime.security.tokens import DEFAULT_TTL_S

    environ = os.environ if environ is None else environ
    keys = KeySet.from_environment(environ)
    ttl = int(environ.get(TOKEN_TTL_ENV) or DEFAULT_TTL_S)
    auth = Authenticator(keys, TokenStore(ttl_s=ttl))
    limits = Limits.from_environment(environ)
    rate = ConnectionRate(limits.connections_per_minute)
    auth_failures = ConnectionRate(limits.connections_per_minute)
    mints = ConnectionRate(limits.tokens_per_minute)
    slots = SessionSlots(limits, keys=len(keys))
    proxies = ProxyTrust.from_environment(environ)
    origins = OriginRule.from_environment(environ)
    scrub_access_logs()  # a browser can only carry a token in the query string
    telemetry.emit("auth.configured", stage="server", enabled=auth.enabled, keys=len(keys),
                   session_token_ttl_s=ttl,
                   hint=None if auth.enabled else "no keys set, so only localhost is answered")
    telemetry.emit("limits.configured", stage="server", max_sessions=limits.max_sessions,
                   per_key=limits.per_key(len(keys)), idle_timeout_s=limits.idle_timeout_s,
                   max_session_s=limits.max_session_s, max_turn_audio_s=limits.max_turn_audio_s,
                   connections_per_minute=limits.connections_per_minute,
                   tokens_per_minute=limits.tokens_per_minute, trusted_proxy=proxies.enabled,
                   allowed_origins=origins.allowed or "same origin only")


async def reload_keys() -> dict:
    """Re-read the keys without restarting.

    Restarting reloads the models, which on a GPU is a minute of downtime — far
    too much to pay for revoking one leaked key, and the kind of cost that makes
    people put revocation off. Environment variables can't change under a running
    process, so this is for FUSION_ACCEPTED_KEYS_FILE, which can.

    Removing a key takes effect completely: requests with it fail, tokens it
    minted are dropped, and conversations it started are closed. A key is removed
    because it leaked, and letting the current caller finish is letting the
    attacker finish.
    """
    global auth, slots

    was = {entry["fingerprint"] for entry in auth.keys.describe()}
    try:
        keys = KeySet.from_environment()
    except ConfigurationError as e:
        telemetry.emit("keys.reload_failed", level="error", stage="server", error=str(e),
                       hint="the keys in use are unchanged")
        return {"error": str(e)}
    now = {entry["fingerprint"] for entry in keys.describe()}
    removed, added = was - now, now - was
    auth = Authenticator(keys, auth.tokens)  # the same token store: valid tokens keep working
    slots.keys = len(keys)
    tokens_dropped = sum(auth.tokens.drop_for_key(gone) for gone in removed)
    doomed = [(sid, socket) for sid, (fp, socket) in _live_sessions.items() if fp in removed]
    for session_id, socket in doomed:
        with contextlib.suppress(Exception):
            await socket.send_json({"type": "error", "code": "key_revoked",
                                    "message": "the key this conversation started with was removed",
                                    "retryable": False})
        with contextlib.suppress(Exception):
            await socket.close(code=1008)
        _live_sessions.pop(session_id, None)
    telemetry.emit("keys.reloaded", stage="server", keys=len(keys), added=len(added),
                   removed=len(removed), tokens_dropped=tokens_dropped, sessions_closed=len(doomed))
    return {"keys": len(keys), "added": len(added), "removed": len(removed),
            "sessions_closed": len(doomed)}


def _listen_for_reload() -> None:
    """`kill -HUP <pid>` re-reads the keys file. Not available on Windows, where
    a restart is the only option."""
    import signal

    try:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGHUP, lambda: asyncio.create_task(reload_keys()))
    except (AttributeError, NotImplementedError, RuntimeError, ValueError):
        pass


def _client_host(scope) -> Optional[str]:
    """The address the socket actually came from. Authentication uses only this:
    a forwarded header can claim to be 127.0.0.1, and the localhost rule must
    never be fooled by something a client can write."""
    client = getattr(scope, "client", None)
    return client.host if client else None


def _rate_address(scope) -> Optional[str]:
    """Who to count against. Behind a proxy every caller shares one socket
    address, so a per-address limit would throttle a whole site — but the
    forwarded header is only believed when we were told the proxy is ours."""
    if proxies.believes(_client_host(scope)):
        forwarded = scope.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return _client_host(scope)


def require_key(request: Request) -> Principal:
    """For everything a server-side caller reaches. A session token is not
    accepted here: a token is for a browser holding a WebSocket, and it must not
    be able to mint another or read metrics."""
    return _authenticate(request, allow_token=False)


def _authenticate(request: Request, *, allow_token: bool) -> Principal:
    try:
        principal = auth.authenticate(
            client_host=_client_host(request),
            authorization=request.headers.get("authorization"),
            token=request.query_params.get("token") if allow_token else None,
            allow_token=allow_token,
        )
    except Unauthorized as e:
        telemetry.emit("auth.failed", level="warning", stage="server", path=request.url.path,
                       client=_client_host(request), reason=str(e))
        # Guessing keys should cost something. Only failures are counted, so a
        # busy backend minting a token per visitor is never throttled.
        auth_failures.check(_rate_address(request))
        raise
    if principal.via != "loopback":
        telemetry.emit("auth.ok", level="debug", stage="server", path=request.url.path,
                       key=principal.label, via=principal.via)
    return principal


@app.exception_handler(Unauthorized)
async def _unauthorized(request: Request, exc: Unauthorized) -> JSONResponse:
    return JSONResponse(status_code=401, content={"error": {
        "code": exc.code, "message": str(exc), "fix": exc.fix, "retryable": False}})


@app.exception_handler(OverLimit)
async def _over_limit(request: Request, exc: OverLimit) -> JSONResponse:
    telemetry.emit("request.rejected", level="warning", stage="server", reason=exc.reason,
                   path=request.url.path, client=_client_host(request))
    return JSONResponse(status_code=429, content={"error": {
        "code": exc.reason, "message": str(exc), "retryable": True}})


@app.exception_handler(ConfigurationError)
async def _misconfigured(request: Request, exc: ConfigurationError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": {
        "code": "configuration_error", "message": str(exc), "retryable": False}})


class SessionResponse(BaseModel):
    """What a backend hands to its own page. The key never goes near a browser."""

    token: str
    expires_in: int
    ws_url: str


@app.post("/v1/sessions", response_model=SessionResponse)
async def create_session(request: Request, principal: Principal = Depends(require_key)):
    """Mint a short-lived, single-use token for one browser session."""
    from fusion_runtime.config import TRUSTED_PROXY_ENV

    # A generous ceiling per key, so a leaked key can't mint without end, while a
    # busy site minting one token per visitor never notices it.
    mints.check(principal.fingerprint)

    if not auth.secure_enough_to_mint(
        scheme=request.url.scheme, client_host=_client_host(request),
        forwarded_proto=request.headers.get("x-forwarded-proto"),
        trust_proxy=proxies.believes(_client_host(request)),
    ):
        behind_a_proxy = request.headers.get("x-forwarded-proto")
        raise Unauthorized(
            "tokens are only issued over an encrypted connection",
            # The common case in a deployment: a proxy terminated TLS and speaks
            # plain http to us, so the connection we see looks insecure when it
            # isn't. We believe its header only when told the proxy is ours.
            fix=(f"Your proxy says the caller used {behind_a_proxy}. If that proxy is yours, name it: "
                 f"{TRUSTED_PROXY_ENV}=10.0.0.0/8 (or =1 for any peer, when nothing else can reach "
                 f"this port). Its headers are ignored until then."
                 if behind_a_proxy else
                 "Serve this behind TLS and use https:// (and wss:// for the WebSocket). A token in a URL "
                 "over plain http can be read by every hop in between."),
        )
    minted = auth.tokens.mint(principal)
    telemetry.emit("token.minted", stage="server", key=principal.label, expires_in_s=minted.expires_in)
    base = str(request.base_url).rstrip("/").replace("https://", "wss://").replace("http://", "ws://")
    return SessionResponse(token=minted.token, expires_in=minted.expires_in,
                           ws_url=f"{base}/v1/voice/ws?token={minted.token}")


# ============ Browser Client ============

@app.get("/", include_in_schema=False)
async def console():
    """The console: open the server in a browser and talk to the agent."""
    return HTMLResponse(web.console_html())


@app.get(web.CLIENT_ROUTE, include_in_schema=False)
async def client_script():
    """The client the console runs on, for any page that wants to embed it.

    Served to every origin on purpose: a customer's site loads this from the
    server it talks to. It contains no secrets — a page authenticates with a
    short-lived token, never an API key.
    """
    return Response(
        web.client_js(),
        media_type="application/javascript",
        headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "no-cache"},
    )


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
async def voice_chat(request: VoiceChatRequest, principal: Principal = Depends(require_key)):
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

    # Run the pipeline, keeping each turn's text and numbers (see _collect_turns)
    turns: list = []
    trace = SessionTrace(session_id=request_id)  # run_pipeline sends its turn traces to on_event
    collect = _collect_turns(turns)

    async def audio_chunks():
        yield audio

    output = bytearray()
    with session_scope(request_id):
        try:
            pipeline = orchestrator.run_pipeline(
                audio_chunks(),
                request.system_prompt or (agent.prompt if agent is not None else DEFAULT_PROMPT),
                on_event=collect,
                trace=trace,
                tools=agent.tools if agent is not None else (),
            )
            try:
                async for chunk in pipeline:
                    output.extend(chunk)
            finally:
                await pipeline.aclose()
        except Exception as e:
            return _error_response(e, request_id)

    latency = (time.perf_counter() - start) * 1000
    return VoiceChatResponse(
        audio_base64=base64.b64encode(bytes(output)).decode(),
        transcript=" ".join(turn.user for turn in turns if turn.user),
        response_text=" ".join(turn.agent for turn in turns if turn.agent),
        latency_ms=latency,
        turns=turns,
    )


def _collect_turns(turns: list):
    """Gather each turn's text and metrics from the pipeline's events.

    The pipeline reports a turn's words as they are recognized and its numbers
    when the turn ends, so they're stitched together here.
    """
    said: dict = {"user": "", "agent": ""}

    def on_event(event: dict) -> None:
        kind = event.get("type")
        if kind == "transcript" and event.get("is_final"):
            said["user"] = event.get("text", "")
        elif kind == "response" and event.get("is_final"):
            said["agent"] = event.get("text", "")
        elif kind == "echo_discarded":
            said["user"] = said["agent"] = ""
        elif kind == "turn.trace":
            summary = event.get("summary", {})
            outcome = summary.get("outcome", "completed")
            if outcome != "echo_discarded":
                turns.append(Turn(user=said["user"], agent=said["agent"], outcome=outcome, metrics=summary))
            said["user"] = said["agent"] = ""

    return on_event


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
    async def refuse(code: str, message: str, fix: Optional[str], close_code: int, retryable: bool) -> None:
        # Accepted, told why, then closed: a failed handshake alone leaves a
        # browser unable to tell "wrong token" from "server down".
        with contextlib.suppress(Exception):
            await websocket.accept()
            await websocket.send_json({"type": "error", "code": code, "message": message,
                                       "fix": fix, "retryable": retryable})
            await websocket.close(code=close_code)

    try:
        rate.check(_rate_address(websocket))  # before authentication: guessing shouldn't be free
        page = websocket.headers.get("origin")
        if not origins.permits(page, websocket.headers.get("host")):
            raise Unauthorized(
                f"pages on {page} may not use this server",
                fix=f"Add it to {ALLOWED_ORIGINS_ENV} (comma separated) where the server runs. "
                    "Browsers don't stop one site opening a socket to another, so the server does.",
            )
        # A browser can't set a header here, so it presents a session token in the
        # query string instead; everything else sends the key the usual way.
        principal = auth.authenticate(
            client_host=_client_host(websocket),
            authorization=websocket.headers.get("authorization"),
            token=websocket.query_params.get("token"),
        )
    except Unauthorized as e:
        telemetry.emit("auth.failed", level="warning", stage="server", path="/v1/voice/ws",
                       client=_client_host(websocket), reason=str(e))
        with contextlib.suppress(OverLimit):
            auth_failures.check(_rate_address(websocket))
        await refuse(e.code, str(e), e.fix, 1008, retryable=False)  # policy violation
        return
    except OverLimit as e:
        telemetry.emit("session.rejected", level="warning", stage="server", reason=e.reason,
                       client=_client_host(websocket))
        await refuse(e.reason, str(e), None, e.close_code, retryable=True)
        return

    try:
        slots.take(principal.label)
    except OverLimit as e:
        telemetry.emit("session.rejected", level="warning", stage="server", reason=e.reason,
                       key=principal.label, in_use=slots.in_use)
        await refuse(e.reason, str(e), None, e.close_code, retryable=True)
        return
    await websocket.accept()

    session_id = str(uuid.uuid4())
    started = time.monotonic()
    end_reason = "client_disconnected"
    error_code = None
    trace = SessionTrace(session_id=session_id)
    _active_sessions.add(session_id)
    _live_sessions[session_id] = (principal.fingerprint, websocket)

    with session_scope(session_id):
        client = websocket.client
        telemetry.emit("session.start", stage="server",
                       client=f"{client.host}:{client.port}" if client else None,
                       key=principal.label, authenticated=principal.via,
                       profile=os.getenv("FUSION_CONFIG", "development"))
        receive_task = send_task = clock_task = None
        try:
            # Send config
            # A token is spent by the connection that used it, so without this a
            # page could never reconnect — clicking "talk" a second time would
            # fail. Handing over the next one on the socket we already trust
            # costs nothing and keeps every token single use. A page that sits
            # idle past the expiry still has to ask its backend, as it does on
            # first load.
            next_token = auth.tokens.mint(principal).token if auth.enabled else None
            await websocket.send_json({
                "type": "config",
                "sample_rate": orchestrator.config.sample_rate,  # what we want from the client
                "output_sample_rate": orchestrator.config.tts.sample_rate,  # what replies arrive in
                "channels": orchestrator.config.channels,
                "session_id": session_id,
                **({"next_token": next_token} if next_token else {}),
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

            budget = AudioBudget(limits, orchestrator.config.sample_rate)

            async def receive_audio():
                turns_seen = trace.turn_count
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        raise WebSocketDisconnect(message.get("code", 1000))
                    data = message.get("bytes")
                    if data is not None:
                        if trace.turn_count != turns_seen:  # a turn ended: the budget starts again
                            turns_seen = trace.turn_count
                            budget.turn_ended()
                        budget.audio(len(data))
                        await audio_queue.put(data)
                        continue
                    text = message.get("text")
                    if text is not None:
                        budget.message(len(text))
                        _handle_control(text)

            async def watch_the_clock():
                """A socket that opened and went quiet, or a conversation that never
                ends, holds a slot nobody else can use."""
                while True:
                    await asyncio.sleep(5)
                    expired = budget.expired()
                    if expired is not None:
                        raise expired

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
                    agent.prompt if agent is not None else DEFAULT_PROMPT,
                    on_event=lambda event: _dispatch_event(websocket, event),
                    barge_in=barge_in,
                    trace=trace,
                    tools=agent.tools if agent is not None else (),
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
            clock_task = asyncio.create_task(watch_the_clock())
            done, _ = await asyncio.wait({receive_task, send_task, clock_task},
                                         return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                exc = task.exception()
                if exc is not None and not _client_gone(websocket, exc):
                    raise exc

        except OverLimit as e:
            end_reason, error_code = e.reason, e.reason
            telemetry.emit("session.limited", level="warning", stage="server", reason=e.reason,
                           key=principal.label, detail=str(e))
            with contextlib.suppress(Exception):
                await websocket.send_json({"type": "error", "code": e.reason, "message": str(e),
                                           "retryable": True})
            with contextlib.suppress(Exception):
                await websocket.close(code=e.close_code)
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
            for task in (send_task, receive_task, clock_task):
                if task is not None and not task.done():
                    task.cancel()
                    with contextlib.suppress(BaseException):
                        await task
            trace.finish()
            slots.give_back(principal.label)
            _live_sessions.pop(session_id, None)
            _active_sessions.discard(session_id)
            telemetry.emit("session.end", stage="server", reason=end_reason, error_code=error_code,
                           duration_s=round(time.monotonic() - started, 2), turns=trace.turn_count)


# ============ Metrics Endpoints ============

@app.get("/metrics")
async def metrics(principal: Principal = Depends(require_key)):
    """Prometheus metrics: latency histograms, turns, errors, sessions, event-loop lag, memory, CPU."""
    return Response(telemetry.metrics.render(), media_type=telemetry.metrics.content_type)


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
