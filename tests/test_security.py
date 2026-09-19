"""Who may use the server: keys, session tokens, and the localhost-only rule.

The rule that matters most is the last one — a runtime reachable from elsewhere
with no keys is somebody else's GPU — so it's tested from both ends: the command
refuses to start, and the server refuses to answer.
"""
import time

import pytest
from fastapi.testclient import TestClient

import fusion_runtime.server as server
from fusion_runtime.security import (
    Authenticator,
    ConfigurationError,
    KeySet,
    Principal,
    TokenStore,
    Unauthorized,
    fingerprint,
    is_loopback,
    new_key,
)
from fusion_runtime.security.auth import RedactQueryStrings

KEY = new_key()
OTHER = new_key()


class FakeOrchestrator:
    def __init__(self, config):
        self.config = config
        self.ready = True

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    def get_metrics_summary(self):
        return {"count": 0}

    async def run_pipeline(self, audio_stream, system_prompt, on_event=None, barge_in=None, trace=None):
        # Waits on audio that these tests never send, which is the point: without
        # it the session ended the moment it began, handed its slot straight back,
        # and "the server is full" tests passed or failed on timing.
        async for _chunk in audio_stream:
            yield b"\x00\x00" * 160


@pytest.fixture
def client(monkeypatch):
    """A server with one key, seen from somewhere that isn't localhost, over TLS —
    which is what a deployment looks like."""
    monkeypatch.setenv("FUSION_LOG_FORMAT", "off")
    monkeypatch.setenv("FUSION_ACCEPTED_KEYS", f"web:{KEY}")
    monkeypatch.setattr(server, "PipelineOrchestrator", FakeOrchestrator)
    with TestClient(server.app, client=("203.0.113.9", 4000), base_url="https://testserver") as client:
        yield client


@pytest.fixture
def open_client(monkeypatch):
    """A server with no keys, seen from localhost — the development case."""
    monkeypatch.setenv("FUSION_LOG_FORMAT", "off")
    monkeypatch.setenv("FUSION_ACCEPTED_KEYS", "")  # empty, not absent: a .env would fill it in
    monkeypatch.setattr(server, "PipelineOrchestrator", FakeOrchestrator)
    with TestClient(server.app, client=("127.0.0.1", 4000)) as client:
        yield client


# ---- keys -------------------------------------------------------------------------

def test_a_generated_key_is_long_and_recognisable():
    key = new_key()
    assert key.startswith("frun_") and len(key) > 40
    assert new_key() != key
    assert new_key("web").startswith("web:frun_")


def test_keys_are_parsed_with_optional_names():
    keys = KeySet.from_environment({"FUSION_ACCEPTED_KEYS": f"web:{KEY}, {OTHER}"})
    assert len(keys) == 2
    assert keys.verify(KEY) == Principal(fingerprint=fingerprint(KEY), name="web", via="key")
    assert keys.verify(OTHER).name is None
    assert keys.verify("frun_not_a_real_key_but_long") is None
    assert keys.verify(None) is None


def test_a_short_key_stops_the_server_rather_than_pretending_to_protect_it():
    with pytest.raises(ConfigurationError, match="at least 16"):
        KeySet.from_environment({"FUSION_ACCEPTED_KEYS": "secret123"})


def test_keys_can_come_from_a_file_so_they_can_be_reloaded(tmp_path):
    path = tmp_path / "keys"
    path.write_text(f"# production\nweb:{KEY}\n\n{OTHER}\n")
    keys = KeySet.from_environment({"FUSION_ACCEPTED_KEYS_FILE": str(path)})
    assert len(keys) == 2 and keys.verify(KEY).name == "web"


def test_describe_never_includes_the_key():
    described = KeySet.from_environment({"FUSION_ACCEPTED_KEYS": f"web:{KEY}"}).describe()
    assert described == [{"name": "web", "fingerprint": fingerprint(KEY)}]
    assert KEY not in str(described)


# ---- tokens -----------------------------------------------------------------------

def test_a_token_works_once():
    keys = KeySet.from_environment({"FUSION_ACCEPTED_KEYS": KEY})
    store = TokenStore()
    minted = store.mint(keys.verify(KEY))
    assert store.redeem(minted.token).fingerprint == fingerprint(KEY)
    assert store.redeem(minted.token) is None  # a reusable token is just a key with extra steps


def test_a_token_expires():
    now = [1000.0]
    store = TokenStore(ttl_s=60, clock=lambda: now[0])
    minted = store.mint(Principal(fingerprint="abc"))
    now[0] += 61
    assert store.redeem(minted.token) is None


def test_removing_a_key_drops_the_tokens_it_minted():
    """Otherwise a revoked key keeps working for another minute through tokens
    it already issued."""
    store = TokenStore()
    store.mint(Principal(fingerprint="aaa"))
    keep = store.mint(Principal(fingerprint="bbb"))
    assert store.drop_for_key("aaa") == 1
    assert store.redeem(keep.token) is not None


# ---- the localhost rule -----------------------------------------------------------

def test_without_keys_only_localhost_is_answered():
    auth = Authenticator(KeySet(), TokenStore())
    assert auth.authenticate(client_host="127.0.0.1").via == "loopback"
    assert auth.authenticate(client_host="::1").via == "loopback"
    with pytest.raises(Unauthorized, match="localhost only"):
        auth.authenticate(client_host="203.0.113.9")
    with pytest.raises(Unauthorized):
        auth.authenticate(client_host=None)


def test_a_hostname_is_never_treated_as_localhost():
    assert not is_loopback("localhost")  # only what the socket reports, which is an address
    assert not is_loopback("testclient")


def test_once_keys_are_set_localhost_needs_one_too():
    auth = Authenticator(KeySet.from_environment({"FUSION_ACCEPTED_KEYS": KEY}), TokenStore())
    with pytest.raises(Unauthorized):
        auth.authenticate(client_host="127.0.0.1")
    assert auth.authenticate(client_host="127.0.0.1", authorization=f"Bearer {KEY}").via == "key"


def test_tokens_are_only_minted_over_an_encrypted_connection():
    auth = Authenticator(KeySet.from_environment({"FUSION_ACCEPTED_KEYS": KEY}), TokenStore())
    assert auth.secure_enough_to_mint(scheme="https", client_host="203.0.113.9")
    assert auth.secure_enough_to_mint(scheme="http", client_host="127.0.0.1")
    assert not auth.secure_enough_to_mint(scheme="http", client_host="203.0.113.9")
    # a forwarded header is believed only when we were told there is a proxy
    assert not auth.secure_enough_to_mint(scheme="http", client_host="203.0.113.9",
                                          forwarded_proto="https")
    assert auth.secure_enough_to_mint(scheme="http", client_host="203.0.113.9",
                                      forwarded_proto="https", trust_proxy=True)


# ---- the server ------------------------------------------------------------------

def test_health_stays_open_but_metrics_do_not(client):
    assert client.get("/health").status_code == 200  # load balancers need it
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"Authorization": f"Bearer {KEY}"}).status_code == 200


def test_the_console_and_its_script_stay_open(client):
    """They contain no secrets, and a page has to load before it can authenticate."""
    assert client.get("/").status_code == 200
    assert client.get("/fusion-runtime.js").status_code == 200


def test_a_wrong_key_says_nothing_useful(client):
    response = client.get("/metrics", headers={"Authorization": f"Bearer {OTHER}"})
    assert response.status_code == 401
    error = response.json()["error"]
    assert error["code"] == "auth_failed"
    assert "expired" not in error["message"] and "unknown" not in error["message"]


def test_a_backend_mints_a_token_for_its_page(client):
    minted = client.post("/v1/sessions", headers={"Authorization": f"Bearer {KEY}"})
    assert minted.status_code == 200
    body = minted.json()
    assert body["expires_in"] > 0 and body["ws_url"].startswith("ws")
    # the token opens a socket, once
    with client.websocket_connect(f"/v1/voice/ws?token={body['token']}") as ws:
        assert ws.receive_json()["type"] == "config"
    # ...and only once: the second attempt is told why and closed
    with client.websocket_connect(f"/v1/voice/ws?token={body['token']}") as ws:
        assert ws.receive_json()["code"] == "auth_failed"


def test_a_remote_caller_gets_no_token_over_plain_http(monkeypatch):
    """A token in a URL over http is readable by every hop in between."""
    monkeypatch.setenv("FUSION_LOG_FORMAT", "off")
    monkeypatch.setenv("FUSION_ACCEPTED_KEYS", f"web:{KEY}")
    monkeypatch.setattr(server, "PipelineOrchestrator", FakeOrchestrator)
    with TestClient(server.app, client=("203.0.113.9", 4000), base_url="http://testserver") as insecure:
        response = insecure.post("/v1/sessions", headers={"Authorization": f"Bearer {KEY}"})
    assert response.status_code == 401
    assert "encrypted" in response.json()["error"]["message"]


def test_a_refusal_behind_a_proxy_names_the_setting_that_fixes_it(monkeypatch):
    """The common deployment: a proxy terminates TLS and speaks http to us, so the
    connection looks insecure when it isn't."""
    monkeypatch.setenv("FUSION_LOG_FORMAT", "off")
    monkeypatch.setenv("FUSION_ACCEPTED_KEYS", f"web:{KEY}")
    monkeypatch.setattr(server, "PipelineOrchestrator", FakeOrchestrator)
    with TestClient(server.app, client=("203.0.113.9", 4000), base_url="http://testserver") as proxied:
        response = proxied.post("/v1/sessions", headers={
            "Authorization": f"Bearer {KEY}", "X-Forwarded-Proto": "https"})
    assert "FUSION_TRUSTED_PROXY=1" in response.json()["error"]["fix"]


def test_a_token_cannot_mint_another_token(client):
    token = client.post("/v1/sessions", headers={"Authorization": f"Bearer {KEY}"}).json()["token"]
    assert client.post("/v1/sessions", params={"token": token}).status_code == 401


def test_a_socket_without_a_credential_is_told_why_then_closed(client):
    with client.websocket_connect("/v1/voice/ws") as ws:
        message = ws.receive_json()
    assert message["type"] == "error" and message["code"] == "auth_failed"


def test_a_socket_with_the_key_is_accepted(client):
    with client.websocket_connect("/v1/voice/ws", headers={"Authorization": f"Bearer {KEY}"}) as ws:
        assert ws.receive_json()["type"] == "config"


def test_development_on_localhost_needs_no_ceremony(open_client):
    assert open_client.get("/metrics").status_code == 200
    with open_client.websocket_connect("/v1/voice/ws") as ws:
        assert ws.receive_json()["type"] == "config"


# ---- keeping secrets out of the logs ----------------------------------------------

def test_tokens_are_scrubbed_from_the_access_log():
    """uvicorn writes the full path of every request, and a browser can only carry
    a token in the query string."""
    import logging

    record = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1,
                               '%s - "%s %s HTTP/%s" %d', ("1.2.3.4", "GET", "/v1/voice/ws?token=abc123", "1.1", 200),
                               None)
    RedactQueryStrings().filter(record)
    assert "abc123" not in str(record.args) and "[redacted]" in str(record.args)


# ---- limits ----------------------------------------------------------------------

from fusion_runtime.security.limits import AudioBudget, ConnectionRate, Limits, OverLimit, SessionSlots


def test_a_per_key_limit_defaults_to_the_server_limit():
    """Most deployments have one key; a tighter default would cap the whole server."""
    assert Limits(max_sessions=4).per_key() == 4
    assert Limits(max_sessions=4, max_sessions_per_key=1).per_key() == 1


def test_limits_come_from_the_environment_and_survive_nonsense():
    limits = Limits.from_environment({"FUSION_MAX_SESSIONS": "2", "FUSION_IDLE_TIMEOUT_S": "banana"})
    assert limits.max_sessions == 2
    assert limits.idle_timeout_s == Limits.idle_timeout_s  # a typo shouldn't remove the limit


def test_connections_are_counted_per_address_in_a_sliding_window():
    now = [100.0]
    rate = ConnectionRate(per_minute=2, clock=lambda: now[0])
    rate.check("1.2.3.4")
    rate.check("1.2.3.4")
    with pytest.raises(OverLimit, match="too many connections"):
        rate.check("1.2.3.4")
    rate.check("5.6.7.8")  # a different caller is unaffected
    now[0] += 61
    rate.check("1.2.3.4")  # the window has moved on


def test_one_key_cannot_take_every_slot():
    slots = SessionSlots(Limits(max_sessions=3, max_sessions_per_key=2))
    slots.take("web")
    slots.take("web")
    with pytest.raises(OverLimit, match="as many conversations"):
        slots.take("web")
    slots.take("mobile")  # another key still fits inside the server limit
    slots.give_back("web")
    slots.take("web")
    assert slots.in_use == 3


def test_an_oversized_message_is_refused():
    budget = AudioBudget(Limits(max_message_bytes=1024), sample_rate=16000)
    with pytest.raises(OverLimit) as refused:
        budget.message(2048)
    assert refused.value.close_code == 1009


def test_speech_that_never_pauses_ends_the_session():
    budget = AudioBudget(Limits(max_turn_audio_s=1.0), sample_rate=16000)
    budget.audio(16000)  # half a second of 16 kHz, 16-bit mono
    budget.audio(16000)
    with pytest.raises(OverLimit, match="without a pause"):
        budget.audio(16000)


def test_a_turn_ending_starts_the_budget_again():
    budget = AudioBudget(Limits(max_turn_audio_s=1.0), sample_rate=16000)
    budget.audio(32000)
    budget.turn_ended()
    budget.audio(16000)  # the next turn has its own allowance


def test_a_quiet_socket_and_an_endless_one_both_expire():
    now = [0.0]
    budget = AudioBudget(Limits(idle_timeout_s=30, max_session_s=900), 16000, clock=lambda: now[0])
    assert budget.expired() is None
    now[0] = 31
    assert budget.expired().reason == "idle"
    budget.message(10)  # activity resets it
    assert budget.expired() is None
    now[0] = 1000
    assert budget.expired().reason == "session_too_long"


def test_a_full_server_turns_a_caller_away_rather_than_queueing_forever(monkeypatch):
    monkeypatch.setenv("FUSION_LOG_FORMAT", "off")
    monkeypatch.setenv("FUSION_ACCEPTED_KEYS", f"web:{KEY}")
    monkeypatch.setenv("FUSION_MAX_SESSIONS", "1")
    monkeypatch.setattr(server, "PipelineOrchestrator", FakeOrchestrator)
    headers = {"Authorization": f"Bearer {KEY}"}
    with TestClient(server.app, client=("203.0.113.9", 4000), base_url="https://testserver") as client:
        with client.websocket_connect("/v1/voice/ws", headers=headers) as first:
            assert first.receive_json()["type"] == "config"
            with client.websocket_connect("/v1/voice/ws", headers=headers) as second:
                refused = second.receive_json()
        assert refused["code"] == "server_full" and refused["retryable"] is True
        # the slot comes back when the first caller leaves
        with client.websocket_connect("/v1/voice/ws", headers=headers) as third:
            assert third.receive_json()["type"] == "config"


def test_guessing_keys_is_rate_limited(monkeypatch):
    monkeypatch.setenv("FUSION_LOG_FORMAT", "off")
    monkeypatch.setenv("FUSION_ACCEPTED_KEYS", f"web:{KEY}")
    monkeypatch.setenv("FUSION_CONNECTIONS_PER_MINUTE", "3")
    monkeypatch.setattr(server, "PipelineOrchestrator", FakeOrchestrator)
    with TestClient(server.app, client=("203.0.113.9", 4000), base_url="https://testserver") as client:
        codes = [client.get("/metrics", headers={"Authorization": f"Bearer {OTHER}"}).status_code
                 for _ in range(5)]
    assert codes[:3] == [401, 401, 401]
    assert codes[-1] == 429  # guessing costs something


def test_a_real_caller_minting_tokens_is_never_throttled(monkeypatch):
    """Only failures are counted: a busy site mints one token per visitor."""
    monkeypatch.setenv("FUSION_LOG_FORMAT", "off")
    monkeypatch.setenv("FUSION_ACCEPTED_KEYS", f"web:{KEY}")
    monkeypatch.setenv("FUSION_CONNECTIONS_PER_MINUTE", "2")
    monkeypatch.setattr(server, "PipelineOrchestrator", FakeOrchestrator)
    with TestClient(server.app, client=("203.0.113.9", 4000), base_url="https://testserver") as client:
        codes = [client.post("/v1/sessions", headers={"Authorization": f"Bearer {KEY}"}).status_code
                 for _ in range(6)]
    assert codes == [200] * 6


def test_a_forwarded_header_is_believed_only_from_a_named_proxy():
    """The header is free to write, so what matters is who the connection came from."""
    from fusion_runtime.security import ProxyTrust

    assert not ProxyTrust(None).enabled  # ignored until configured
    assert not ProxyTrust("0").believes("10.0.0.7")
    assert ProxyTrust("10.0.0.0/8").believes("10.0.0.7")
    assert not ProxyTrust("10.0.0.0/8").believes("203.0.113.9")
    assert ProxyTrust("192.168.1.5").believes("192.168.1.5")
    assert ProxyTrust("1").believes("203.0.113.9")  # any peer, for a port nothing else can reach
    assert not ProxyTrust("not-an-address").enabled  # a typo must not become "trust everyone"


def test_when_keys_are_shared_one_of_them_keeps_a_slot_free():
    """With a single key there is nothing to share with, so it gets the server."""
    limits = Limits(max_sessions=3)
    alone = SessionSlots(limits, keys=1)
    for _ in range(3):
        alone.take("web")
    shared = SessionSlots(limits, keys=2)
    shared.take("web")
    shared.take("web")
    with pytest.raises(OverLimit, match="as many conversations"):
        shared.take("web")
    shared.take("mobile")


def test_minting_is_capped_per_key(monkeypatch):
    """A leaked key shouldn't be able to mint without end."""
    monkeypatch.setenv("FUSION_LOG_FORMAT", "off")
    monkeypatch.setenv("FUSION_ACCEPTED_KEYS", f"web:{KEY}")
    monkeypatch.setenv("FUSION_TOKENS_PER_MINUTE", "3")
    monkeypatch.setattr(server, "PipelineOrchestrator", FakeOrchestrator)
    with TestClient(server.app, client=("203.0.113.9", 4000), base_url="https://testserver") as client:
        codes = [client.post("/v1/sessions", headers={"Authorization": f"Bearer {KEY}"}).status_code
                 for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]


# ---- which pages may connect ------------------------------------------------------

def test_a_browser_on_another_site_is_refused_by_default(client):
    """Browsers don't stop one site opening a socket to another, so the server does."""
    with client.websocket_connect("/v1/voice/ws", headers={
            "Authorization": f"Bearer {KEY}", "Origin": "https://evil.example"}) as ws:
        refused = ws.receive_json()
    assert refused["code"] == "auth_failed"
    assert "FUSION_ALLOWED_ORIGINS" in refused["fix"]


def test_the_console_this_server_serves_still_works(client):
    with client.websocket_connect("/v1/voice/ws", headers={
            "Authorization": f"Bearer {KEY}", "Origin": "https://testserver"}) as ws:
        assert ws.receive_json()["type"] == "config"


def test_a_named_site_is_allowed(monkeypatch):
    monkeypatch.setenv("FUSION_LOG_FORMAT", "off")
    monkeypatch.setenv("FUSION_ACCEPTED_KEYS", f"web:{KEY}")
    monkeypatch.setenv("FUSION_ALLOWED_ORIGINS", "https://shopkart.example")
    monkeypatch.setattr(server, "PipelineOrchestrator", FakeOrchestrator)
    with TestClient(server.app, client=("203.0.113.9", 4000), base_url="https://testserver") as client:
        headers = {"Authorization": f"Bearer {KEY}", "Origin": "https://shopkart.example"}
        with client.websocket_connect("/v1/voice/ws", headers=headers) as ws:
            assert ws.receive_json()["type"] == "config"
        headers["Origin"] = "https://someone-else.example"
        with client.websocket_connect("/v1/voice/ws", headers=headers) as ws:
            assert ws.receive_json()["code"] == "auth_failed"


def test_clients_that_are_not_browsers_send_no_origin_and_are_unaffected(client):
    with client.websocket_connect("/v1/voice/ws", headers={"Authorization": f"Bearer {KEY}"}) as ws:
        assert ws.receive_json()["type"] == "config"


# ---- revoking a key without restarting -------------------------------------------

class _FakeSocket:
    def __init__(self):
        self.sent, self.closed = [], None

    async def send_json(self, message):
        self.sent.append(message)

    async def close(self, code=1000):
        self.closed = code


async def test_reloading_keys_revokes_completely(monkeypatch, tmp_path):
    """A restart would reload the models — far too much to pay for revoking one
    key, and the kind of cost that makes people put revocation off."""
    path = tmp_path / "keys"
    path.write_text(f"web:{KEY}\nmobile:{OTHER}\n")
    monkeypatch.setenv("FUSION_LOG_FORMAT", "off")
    monkeypatch.setenv("FUSION_ACCEPTED_KEYS", "")  # empty, not absent: a .env would fill it in
    monkeypatch.setenv("FUSION_ACCEPTED_KEYS_FILE", str(path))
    server._configure_auth()
    assert server.auth.keys.verify(KEY) is not None

    minted = server.auth.tokens.mint(server.auth.keys.verify(KEY))
    kept = server.auth.tokens.mint(server.auth.keys.verify(OTHER))
    socket = _FakeSocket()
    server._live_sessions["s1"] = (fingerprint(KEY), socket)

    path.write_text(f"mobile:{OTHER}\n")  # the leaked key is taken out
    result = await server.reload_keys()

    assert result == {"keys": 1, "added": 0, "removed": 1, "sessions_closed": 1}
    assert server.auth.keys.verify(KEY) is None
    assert server.auth.tokens.redeem(minted.token) is None  # its tokens went with it
    assert server.auth.tokens.redeem(kept.token) is not None  # the other key is untouched
    assert socket.closed == 1008 and socket.sent[0]["code"] == "key_revoked"
    server._live_sessions.clear()


async def test_a_broken_keys_file_leaves_the_running_keys_alone(monkeypatch, tmp_path):
    path = tmp_path / "keys"
    path.write_text(f"web:{KEY}\n")
    monkeypatch.setenv("FUSION_LOG_FORMAT", "off")
    monkeypatch.setenv("FUSION_ACCEPTED_KEYS", "")  # empty, not absent: a .env would fill it in
    monkeypatch.setenv("FUSION_ACCEPTED_KEYS_FILE", str(path))
    server._configure_auth()

    path.write_text("oops\n")  # too short to be a key
    result = await server.reload_keys()

    assert "error" in result
    assert server.auth.keys.verify(KEY) is not None  # still serving with what it had


# ---- one key on one machine -------------------------------------------------------

def test_a_local_client_can_use_the_key_its_own_server_accepts():
    """On a development machine the two are the same string, and typing it twice
    teaches nothing."""
    from fusion_runtime.security import client_key

    environ = {"FUSION_ACCEPTED_KEYS": f"web:{KEY}"}
    assert client_key("ws://127.0.0.1:8000/v1/voice/ws", environ) == KEY
    assert client_key("ws://localhost:8000/v1/voice/ws", environ) == KEY


def test_it_never_sends_your_server_s_key_to_someone_else_s():
    """The loopback condition is the whole safety of that shortcut."""
    from fusion_runtime.security import client_key

    environ = {"FUSION_ACCEPTED_KEYS": f"web:{KEY}"}
    assert client_key("wss://pod.example/v1/voice/ws", environ) is None
    # ...unless a key was set for that server, which is a different key
    assert client_key("wss://pod.example", {**environ, "FUSION_API_KEY": OTHER}) == OTHER
