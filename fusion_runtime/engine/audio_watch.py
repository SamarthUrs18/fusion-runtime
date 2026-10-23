"""Noticing when the agent can't hear the caller.

Two ways a call goes deaf without anything failing:

    no audio       frames stop arriving (a stalled connection, a client that stopped sending)
    silent audio   frames arrive, but every sample is 0 (a muted mic, a dead audio path)

A caller who is simply quiet still sends room noise, which is never exactly zero, so neither
fires for a pause to think. Raised by Asif Ali, who runs a phone agent: with the greeting going
out first, a muted mic is caught while the greeting still plays, before anyone sits in silence.

The watch only reports, once per episode, and again when the audio comes back. It doesn't end
the call: the caller may unmute. A call that stays dead is ended by the silence limit
(FUSION_MAX_SILENCE_S).

Phone lines add a third: a dead path there is faint hiss, not zeros (Asif Ali again: under ~40 RMS,
where speech peaks over 1000). With a floor set, nothing above it for a while is a "quiet_line".
A caller listening in a quiet room can sit under the floor too, so it's reported, never acted on,
and the loudest moment of the call is kept so a line that was never alive shows after the call.
"""
import time
from typing import Callable, Dict, Optional

import numpy as np

MESSAGES = {
    "no_audio": "The agent isn't receiving any audio from your microphone. Check your connection.",
    "silent_audio": "Your microphone is sending only silence. Is it muted?",
    "quiet_line": "Nothing above the line's noise floor: a dead phone line, or a very quiet caller.",
}


class AudioWatch:
    """`floor_rms` and `quiet_after_s` are for phone lines, where a dead path is faint hiss rather
    than zeros: nothing louder than the floor for that long is a "quiet_line". The loudest moment of
    the call (`peak_rms`) is kept either way, so a call whose caller was never heard shows afterwards.
    """

    def __init__(self, after_s: float = 3.0, clock: Callable[[], float] = time.monotonic,
                 floor_rms: float = 0.0, quiet_after_s: float = 0.0):
        self.after_s = after_s
        self.floor_rms = floor_rms
        self.quiet_after_s = quiet_after_s
        self._clock = clock
        self._last_frame = clock()  # from the start: no frame at all for a while is a problem too
        self._zeros_since: Optional[float] = None
        self._quiet_since: Optional[float] = None
        self.peak_rms = 0.0
        self.problem: Optional[str] = None  # "no_audio" | "silent_audio" | "quiet_line" while it lasts
        self._problem_since: Optional[float] = None

    @property
    def heard_caller(self) -> bool:
        """Something above the floor arrived at some point: the line was alive."""
        return self.peak_rms >= self.floor_rms

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
        if data:
            samples = np.frombuffer(data[: len(data) - len(data) % 2], dtype=np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
            self.peak_rms = max(self.peak_rms, rms)
            if self.floor_rms > 0 and self.quiet_after_s > 0:
                if rms < self.floor_rms:
                    self._quiet_since = self._quiet_since or now
                    if self.problem in ("no_audio", "silent_audio"):  # audio is back, if quiet
                        return self._end(now)
                    if now - self._quiet_since >= self.quiet_after_s and self.problem != "quiet_line":
                        return self._start("quiet_line", now - self._quiet_since, now)
                    return None
                self._quiet_since = None
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
