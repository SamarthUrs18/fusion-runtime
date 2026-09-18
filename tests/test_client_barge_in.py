"""
Tests for the `frun talk` client's interruption handling
(fusion_runtime/cli/_talk_client.py).

The audio side — echo cancellation, instant flush, the send gate — is tested
in test_duplex_audio.py. What's left in the client is small but easy to get
wrong: dropping the stale tail of an interrupted reply, and telling the server
when the bot is audible, so interruptions keep working until the speaker
actually goes quiet rather than only until the reply finishes generating.
"""
import json

import pytest

from fusion_runtime.cli._talk_client import PlaybackReporter, ReplyGate


class TestReplyGate:
    def test_plays_bot_audio_normally(self):
        assert ReplyGate().should_play()

    def test_drops_the_rest_of_an_interrupted_reply(self):
        gate = ReplyGate()
        gate.on_interrupted()
        assert not gate.should_play(), "the bot would resume talking over the user"

    def test_plays_again_once_the_next_turn_starts(self):
        gate = ReplyGate()
        gate.on_interrupted()
        gate.on_new_turn()
        assert gate.should_play()


class TestPlaybackReporter:
    def test_reports_each_change_exactly_once(self):
        reporter = PlaybackReporter()
        assert reporter.update(False) is None, "nothing to report before anything plays"
        assert json.loads(reporter.update(True)) == {"type": "playback", "playing": True}
        assert reporter.update(True) is None, "must not flood the socket while playing"
        assert json.loads(reporter.update(False)) == {"type": "playback", "playing": False}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestMicMeter:
    """The level bar redraws itself with a carriage return, which needs a terminal.

    Where output is captured line by line — a pipe, a log, an editor's output pane —
    each redraw arrives as another line, and a single turn fills the screen with
    meters. There we say only what changed.
    """

    def _client(self, redraws: bool, bot_audible: bool = False):
        from fusion_runtime.cli._talk_client import VoiceChatClient

        class FakeAudio:
            echo_stats = None

        client = VoiceChatClient.__new__(VoiceChatClient)
        client._last_meter, client._last_state, client.redraws = 0.0, None, redraws
        client.audio = FakeAudio()
        client.audio.bot_audible = bot_audible
        return client

    def _chunk(self):
        from fusion_runtime.audio.duplex_audio import MicChunk

        return MicChunk(pcm16=b"\x00\x00" * 800, echo_possible=False, echo_cancelled=True)

    def test_a_terminal_gets_the_live_bar(self, capsys):
        client = self._client(redraws=True)
        for _ in range(3):
            client._last_meter = 0
            client._show_meter(self._chunk())
        printed = capsys.readouterr().out
        assert printed.count("\r") == 3 and "mic" in printed

    def test_captured_output_gets_one_line_per_change_instead(self, capsys):
        client = self._client(redraws=False)
        for _ in range(6):
            client._last_meter = 0
            client._show_meter(self._chunk())
        client.audio.bot_audible = True
        client._last_meter = 0
        client._show_meter(self._chunk())
        lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
        assert len(lines) == 2  # "listening", then "bot talking"
        assert "\r" not in "".join(lines)
