"""What one call has said so far, trimmed to fit the model's context.

Kept per call by the engine, never inside a runtime (runtimes are shared by
every call and hold no conversation state). Each turn sends the system
prompt, as much recent history as fits, and the new user message.

A reply cut off by an interruption is stored as far as it got, since that
is roughly what the caller heard. (Tokens are pulled as speech is
synthesized, so the stored text runs at most about a sentence ahead of the
audio; playback position from the client can tighten this later.) If the
caller interrupted before any reply text existed, their earlier words are
carried into their next message instead of leaving an empty assistant turn.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from fusion_runtime.contract import Message

CHARS_PER_TOKEN = 3  # conservative: English averages ~4, many other languages fewer


@dataclass
class Conversation:
    system_prompt: str
    max_messages: int = 20  # history messages kept, not counting system and the new user message
    max_chars: Optional[int] = None  # total prompt budget in characters; None = no limit
    history: List[Message] = field(default_factory=list)
    _carried_user_text: str = ""

    @classmethod
    def for_context(cls, system_prompt: str, n_ctx: Optional[int], reply_tokens: Optional[int],
                    max_messages: int = 20) -> "Conversation":
        """Size the history budget from the model's context window minus room for the reply."""
        max_chars = None
        if n_ctx:
            max_chars = max(0, (n_ctx - (reply_tokens or 0)) * CHARS_PER_TOKEN)
        return cls(system_prompt, max_messages=max_messages, max_chars=max_chars)

    def messages_for(self, user_text: str) -> List[Message]:
        """The prompt for a new user message: system, the recent history that fits, the message."""
        user = Message(role="user", content=self._with_carried(user_text))
        system = Message(role="system", content=self.system_prompt)
        kept = self.history[-self.max_messages:] if self.max_messages > 0 else []
        if self.max_chars is not None:
            used = len(system.content) + len(user.content)
            fitted: List[Message] = []
            for message in reversed(kept):
                used += len(message.content)
                if used > self.max_chars:
                    break
                fitted.append(message)
            kept = list(reversed(fitted))
        while kept and kept[0].role != "user":  # never start history on an assistant reply
            kept = kept[1:]
        return [system, *kept, user]

    def add_turn(self, user_text: str, reply_text: str, interrupted: bool = False) -> None:
        """Record a finished (or interrupted) exchange."""
        user_text = self._with_carried(user_text)
        self._carried_user_text = ""
        reply_text = reply_text.strip()
        if not reply_text:
            if interrupted:
                self._carried_user_text = user_text  # nothing was said back; fold into the next message
            return
        self.history.append(Message(role="user", content=user_text))
        self.history.append(Message(role="assistant", content=reply_text))
        overflow = len(self.history) - max(self.max_messages, 0) * 2  # keep memory bounded on long calls
        if overflow > 0:
            del self.history[:overflow + overflow % 2]

    def retract_last_turn(self) -> Optional[str]:
        """Undo the last exchange: the user wasn't finished, they paused.

        Their words carry into their next message, and the reply they cut off
        leaves the history (the agent effectively hadn't answered yet).
        Returns the retracted user text, or None if there was nothing to undo.
        """
        if len(self.history) >= 2 and self.history[-1].role == "assistant":
            self.history.pop()
            user_text = self.history.pop().content
            self._carried_user_text = f"{user_text} {self._carried_user_text}".strip()
            return user_text
        return None

    @property
    def has_carried_text(self) -> bool:
        return bool(self._carried_user_text)

    @property
    def turns(self) -> int:
        return len(self.history) // 2

    def _with_carried(self, user_text: str) -> str:
        if not self._carried_user_text:
            return user_text
        return f"{self._carried_user_text} {user_text}".strip()
