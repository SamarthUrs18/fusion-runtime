"""frun up's welcome screen: for a person at a terminal, never in logs."""
import io

from fusion_runtime.cli import banner

ROWS = [("agent", "agent.py · greets callers"), ("llm", "Qwen2.5-7B-Instruct-AWQ on vLLM · starting it now"),
        ("talk", "http://localhost:8000 in a browser · or: frun talk")]


class Terminal(io.StringIO):
    def isatty(self):
        return True


def test_shown_only_to_a_person_at_a_terminal():
    assert banner.should_show(Terminal(), json_logs=False, environ={})
    assert not banner.should_show(Terminal(), json_logs=True, environ={})  # a log collector is reading
    assert not banner.should_show(io.StringIO(), json_logs=False, environ={})  # a file, a pipe, docker logs
    assert not banner.should_show(Terminal(), json_logs=False, environ={"TERM": "dumb"})


def test_wide_terminal_gets_the_letters_and_every_row():
    text = banner.render("0.1.2", ROWS, notes=["loading models"], width=100, color=False)
    lines = text.splitlines()
    assert sum(row in text for row in banner.WORD) == 5  # five rows of the word
    assert banner.WAVES[0] in text
    assert "r u n t i m e   0.1.2" in text and banner.TAGLINE in text
    for label, value in ROWS:
        assert any(line.strip().startswith(label) and line.endswith(value) for line in lines)
    assert "loading models" in text
    assert max(len(line) for line in lines) <= banner.WIDE_ENOUGH + 30  # nothing absurdly wide besides the rows


def test_a_narrow_terminal_gets_one_line_instead_of_wrapped_letters():
    text = banner.render("0.1.2", ROWS, width=banner.WIDE_ENOUGH - 1, color=False)
    assert banner.WORD[-1] not in text and "fusion-runtime 0.1.2" in text


def test_no_color_means_no_escape_codes():
    out = Terminal()
    banner.show("0.1.2", ROWS, stream=out, environ={"NO_COLOR": "1"})
    assert "\033[" not in out.getvalue() and "agent" in out.getvalue()
    coloured = banner.render("0.1.2", ROWS, width=100, color=True)
    assert banner.ORANGE in coloured


def test_a_terminal_that_cant_draw_the_dots_gets_bars_or_nothing():
    assert banner.wave_for("utf-8", {}) == banner.WAVES[0]
    assert banner.wave_for("cp1252", {}) == ""  # neither dots nor dotless-i bars encode: no wave, no boxes
    assert banner.wave_for(None, {}) == ""
    text = banner.render("0.1.2", ROWS, width=100, color=False, wave="")
    assert "r u n t i m e" in text and banner.WAVES[0] not in text


def test_a_pod_shows_its_public_link_not_localhost():
    from fusion_runtime.cli.up import public_url

    pod = {"RUNPOD_POD_ID": "abc123"}
    assert public_url("0.0.0.0", 8000, pod) == "https://abc123-8000.proxy.runpod.net"
    # Set by hand (a domain, another provider's proxy), it wins
    assert public_url("0.0.0.0", 8000, {**pod, "FUSION_PUBLIC_URL": "https://voice.shopkart.example/"}) == \
        "https://voice.shopkart.example"
    # Bound to this machine only, Runpod's proxy can't reach it: no public link to guess
    assert public_url("127.0.0.1", 8000, pod) is None
    # ...but behind a proxy on this machine (Caddy), the address you set is the one people open
    assert public_url("127.0.0.1", 8000, {"FUSION_PUBLIC_URL": "https://voice.example.com"}) == \
        "https://voice.example.com"
    assert public_url("0.0.0.0", 8000, {}) is None
