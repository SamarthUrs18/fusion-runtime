"""Noticing when the agent can't hear the caller.

Two ways a call goes deaf without anything failing:

    no audio       frames stop arriving (a stalled connection, a client that stopped sending)
    silent audio   frames arrive, but every sample is 0 (a muted mic, a dead audio path)

A caller who is simply quiet still sends room noise, which is never exactly zero, so neither
fires for a pause to think. Raised by Asif Ali, who runs a phone agent: with the greeting going
out first, a muted mic is caught while the greeting still plays, before anyone sits in silence.

The watch only reports, once per episode, and again when the audio comes back. It doesn't end
the call: the caller may unmute. A call that stays dead is ended by the silence limit
(FUSION_MAX_SILENCE_S). Phone lines will need a level-plus-duration rule instead (a dead line
there is faint hiss, not zeros).
"""
import time
from typing import Callable, Dict, Optional

MESSAGES = {
    "no_audio": "The agent isn't receiving any audio from your microphone. Check your connection.",
    "silent_audio": "Your microphone is sending only silence. Is it muted?",
}


class AudioWatch:
    def __init__(self, after_s: float = 3.0, clock: Callable[[], float] = time.monotonic):
        self.after_s = after_s
        self._clock = clock
        self._last_frame = clock()  # from the start: no frame at all for a while is a problem too
        self._zeros_since: Optional[float] = None
        self.problem: Optional[str] = None  # "no_audio" | "silent_audio" while it lasts
        self._problem_since: Optional[float] = None

    @property
    def enabled(self) -> bool:
        return self.after_s > 0

    def frame(self, data: bytes) -> Optional[Dict]:
        """A chunk of caller audio arrived. Returns an event when the state changes."""
        if not self.enabled:
            return None
        now = self._clock()
        self._last_frame = now
        if data and data.count(0) == len(data):
            self._zeros_since = self._zeros_since or now
            if now - self._zeros_since >= self.after_s and self.problem != "silent_audio":
                return self._start("silent_audio", now - self._zeros_since, now)
            return None
        self._zeros_since = None
        if self.problem is not None:
            return self._end(now)
        return None

    def check(self) -> Optional[Dict]:
        """Called on a timer: frames that stopped arriving can't report themselves."""
        if not self.enabled:
            return None
        now = self._clock()
        quiet = now - self._last_frame
        if quiet >= self.after_s and self.problem != "no_audio":
            return self._start("no_audio", quiet, now)
        return None

    def _start(self, problem: str, seconds: float, now: float) -> Dict:
        self.problem, self._problem_since = problem, now
        return {"event": "audio.problem", "problem": problem, "seconds": round(seconds, 1),
                "message": MESSAGES[problem]}

    def _end(self, now: float) -> Dict:
        problem, since = self.problem, self._problem_since
        self.problem, self._problem_since = None, None
        return {"event": "audio.ok", "problem": problem, "lasted_s": round(now - (since or now), 1)}
