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

History is kept as whole turns: the user's message, any tool calls and their
results, and the reply. A turn is trimmed or retracted as one piece, so a tool
result never reaches the model without the call it answers. Tool traffic stays
because the model otherwise sees only its own earlier answers, not that they
came from a lookup, and imitates them instead of looking again.
"""
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence

from fusion_runtime.contract import Message

CHARS_PER_TOKEN = 3  # conservative: English averages ~4, many other languages fewer
FULL_TOOL_RESULT_TURNS = 2  # the most recent turns whose tool results are sent in full
OLD_TOOL_RESULT_CHARS = 500  # older results are shortened to this; the lookup still shows it happened


@dataclass
class Conversation:
    system_prompt: str
    max_messages: int = 20  # history messages kept, not counting system and the new user message
    max_chars: Optional[int] = None  # total prompt budget in characters; None = no limit
    _turns: List[List[Message]] = field(default_factory=list)
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
        """The prompt for a new user message: system, the recent turns that fit, the message."""
        user = Message(role="user", content=self._with_carried(user_text))
        system = Message(role="system", content=self.system_prompt)
        used_messages = 0
        used_chars = len(system.content) + len(user.content)
        kept: List[List[Message]] = []
        for age, turn in enumerate(reversed(self._turns)):
            if age >= FULL_TOOL_RESULT_TURNS:
                turn = [_shortened(m) for m in turn]
            used_messages += len(turn)
            used_chars += sum(_chars(m) for m in turn)
            if used_messages > self.max_messages or (self.max_chars is not None and used_chars > self.max_chars):
                break
            kept.append(turn)
        return [system, *(m for turn in reversed(kept) for m in turn), user]

    def add_turn(self, user_text: str, reply_text: str, interrupted: bool = False,
                 steps: Sequence[Message] = ()) -> None:
        """Record a finished (or interrupted) exchange.

        `steps` are the tool calls and results made during the turn, in order (each assistant
        message that asked for tools, then its tool messages); `reply_text` is what was said after
        the last of them.
        """
        user_text = self._with_carried(user_text)
        self._carried_user_text = ""
        reply_text = reply_text.strip()
        if not reply_text and not steps:
            if interrupted:
                self._carried_user_text = user_text  # nothing was said back; fold into the next message
            return
        turn = [Message(role="user", content=user_text), *steps]
        if reply_text:
            turn.append(Message(role="assistant", content=reply_text))
        self._turns.append(turn)
        if len(self._turns) > max(self.max_messages, 0):  # keep memory bounded on long calls
            del self._turns[:len(self._turns) - max(self.max_messages, 0)]

    def retract_last_turn(self) -> Optional[str]:
        """Undo the last exchange: the user wasn't finished, they paused.

        Their words carry into their next message, and the reply they cut off
        leaves the history (the agent effectively hadn't answered yet).
        Returns the retracted user text, or None if there was nothing to undo.
        """
        if not self._turns:
            return None
        user_text = self._turns.pop()[0].content
        self._carried_user_text = f"{user_text} {self._carried_user_text}".strip()
        return user_text

    @property
    def has_carried_text(self) -> bool:
        return bool(self._carried_user_text)

    @property
    def turns(self) -> int:
        return len(self._turns)

    @property
    def history(self) -> List[Message]:
        """Every stored message, oldest first (not trimmed to a budget; see messages_for)."""
        return [m for turn in self._turns for m in turn]

    def _with_carried(self, user_text: str) -> str:
        if not self._carried_user_text:
            return user_text
        return f"{self._carried_user_text} {user_text}".strip()


def _chars(message: Message) -> int:
    return len(message.content) + sum(len(c.name) + len(c.arguments) for c in message.tool_calls)


def _shortened(message: Message) -> Message:
    if message.role != "tool" or len(message.content) <= OLD_TOOL_RESULT_CHARS:
        return message
    return replace(message, content=message.content[:OLD_TOOL_RESULT_CHARS] + " …(shortened)")
