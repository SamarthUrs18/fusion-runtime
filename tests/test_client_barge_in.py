"""
Tests for the reference client's interruption handling
(examples/websocket_client.py).

The audio side — echo cancellation, instant flush, the send gate — is tested
in test_duplex_audio.py. What's left in the client is small but easy to get
wrong: dropping the stale tail of an interrupted reply, and telling the server
when the bot is audible, so interruptions keep working until the speaker
actually goes quiet rather than only until the reply finishes generating.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from websocket_client import PlaybackReporter, ReplyGate  # noqa: E402


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
