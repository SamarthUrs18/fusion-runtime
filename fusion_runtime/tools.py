"""Tools an agent can call mid-conversation: plain Python functions.

    from fusion_runtime import Agent, tool

    @tool
    async def order_status(order_id: str) -> str:
        \"\"\"Look up where an order is.

        Args:
            order_id: The order number, as the caller reads it out.
        \"\"\"
        return await shop.status(order_id)

    agent = Agent(prompt="...", tools=[order_status])

The model sees the function's name, the first paragraph of its docstring and
a JSON Schema built from its type hints (parameter descriptions come from an
`Args:` section). When it asks for a call, the arguments are checked against
the signature, the function runs, and what it returns goes back to the model
as text: a string as it is, anything else as JSON.

Tools are async or ordinary functions. An ordinary function runs on a worker
thread, because anything blocking on the event loop stalls audio and
interruption for every call on the server. Each call has a timeout
(`@tool(timeout_s=...)`, default 10 s): the caller is waiting on the line.
Python can't stop a thread, so an ordinary function that times out keeps
running in the background; its result is dropped.

A tool that fails doesn't end the call. The model is told what went wrong
(`error: ...`) and can apologise, ask again or try something else. Arguments
and results are content: telemetry only logs them when content logging is on.
"""
import asyncio
import collections.abc
import enum
import functools
import inspect
import json
import re
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from fusion_runtime.contract import ToolSpec

DEFAULT_TIMEOUT_S = 10.0
MAX_RESULT_CHARS = 8000  # a result longer than this is cut: it all goes into the prompt
_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # what OpenAI-compatible servers accept


class ToolError(ValueError):
    """A tool can't be defined as written."""


@dataclass(frozen=True)
class ToolResult:
    """What one call returned, as the text the model will read."""

    content: str
    ok: bool = True
    duration_ms: float = 0.0
    error: Optional[str] = None  # "timeout", "bad_arguments", "unknown_tool" or "failed"


@dataclass(frozen=True)
class Tool:
    """A function the model can call. Made with @tool, or from any function in Agent(tools=[...])."""

    fn: Callable[..., Any]
    name: str
    description: str
    parameters: Dict[str, Any]
    timeout_s: float = DEFAULT_TIMEOUT_S
    _signature: inspect.Signature = field(default=None, repr=False, compare=False)  # type: ignore[assignment]
    _enums: Mapping[str, type] = field(default_factory=dict, repr=False, compare=False)  # parameter -> Enum class

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Still the plain function, so it can be called and tested directly."""
        return self.fn(*args, **kwargs)

    async def run(self, arguments: str) -> ToolResult:
        """Call the tool with the model's JSON arguments. Never raises, except for cancellation."""
        started = asyncio.get_running_loop().time()

        def result(content: str, error: Optional[str] = None) -> ToolResult:
            elapsed = (asyncio.get_running_loop().time() - started) * 1000
            return ToolResult(content=content, ok=error is None, duration_ms=elapsed, error=error)

        try:
            kwargs = self._arguments(arguments)
        except ToolError as e:
            return result(f"error: {e}", "bad_arguments")
        try:
            if inspect.iscoroutinefunction(self.fn):
                value = await asyncio.wait_for(self.fn(**kwargs), self.timeout_s)
            else:
                loop = asyncio.get_running_loop()
                call = functools.partial(self.fn, **kwargs)
                value = await asyncio.wait_for(loop.run_in_executor(None, call), self.timeout_s)
                if inspect.isawaitable(value):  # a plain function that returned a coroutine
                    value = await asyncio.wait_for(value, self.timeout_s)
        except asyncio.TimeoutError:
            return result(f"error: {self.name} took longer than {self.timeout_s:g} s and was stopped", "timeout")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return result(f"error: {self.name} failed: {type(e).__name__}: {e}", "failed")
        return result(_as_text(value))

    def _arguments(self, arguments: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(arguments) if arguments and arguments.strip() else {}
        except ValueError:
            raise ToolError(f"the arguments for {self.name} weren't valid JSON: {arguments[:200]!r}") from None
        if not isinstance(parsed, dict):
            raise ToolError(f"the arguments for {self.name} must be a JSON object")
        signature = self._signature or inspect.signature(self.fn)
        accepts_any = any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values())
        known = {name for name, p in signature.parameters.items() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
        unknown = sorted(set(parsed) - known)
        if unknown and not accepts_any:
            raise ToolError(f"{self.name} has no parameter {', '.join(unknown)}; it takes {', '.join(sorted(known)) or 'none'}")
        missing = [name for name in self.parameters.get("required", ()) if name not in parsed]
        if missing:
            raise ToolError(f"{self.name} needs {', '.join(missing)}")
        for name, enum_class in self._enums.items():  # the model sends the value; the function gets the member
            if parsed.get(name) is not None:
                try:
                    parsed[name] = enum_class(parsed[name])
                except ValueError:
                    allowed = ", ".join(repr(m.value) for m in enum_class)
                    raise ToolError(f"{self.name}({name}) is one of {allowed}, got {parsed[name]!r}") from None
        return parsed


def tool(fn: Optional[Callable[..., Any]] = None, *, name: Optional[str] = None, description: Optional[str] = None,
         timeout_s: float = DEFAULT_TIMEOUT_S) -> Any:
    """Make a function callable by the model: @tool, or @tool(name=..., description=..., timeout_s=...)."""
    def make(function: Callable[..., Any]) -> Tool:
        return _build(function, name=name, description=description, timeout_s=timeout_s)

    return make(fn) if fn is not None else make


def as_tools(values: Sequence[Any]) -> Tuple[Tool, ...]:
    """Agent(tools=[...]): Tool objects, or plain functions made into tools. Names must be unique."""
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise ToolError(f"tools takes a list of functions, got {type(values).__name__}")
    tools: List[Tool] = []
    for value in values:
        if isinstance(value, Tool):
            tools.append(value)
        elif callable(value):
            tools.append(_build(value))
        else:
            raise ToolError(f"tools takes functions, got {type(value).__name__}: {value!r}")
    seen: Dict[str, Tool] = {}
    for t in tools:
        if t.name in seen:
            raise ToolError(f"two tools are named {t.name!r}; give one a different name with @tool(name=...)")
        seen[t.name] = t
    return tuple(tools)


# ---- building a tool from a function ---------------------------------------------------------

def _build(fn: Callable[..., Any], *, name: Optional[str] = None, description: Optional[str] = None,
           timeout_s: float = DEFAULT_TIMEOUT_S) -> Tool:
    if not callable(fn):
        raise ToolError(f"@tool needs a function, got {type(fn).__name__}")
    tool_name = name or getattr(fn, "__name__", "")
    if not _NAME.match(tool_name or ""):
        raise ToolError(f"a tool name uses letters, digits, _ and - (up to 64), got {tool_name!r}; "
                        "set one with @tool(name=...)")
    if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool) or timeout_s <= 0:
        raise ToolError(f"@tool(timeout_s=...) must be a positive number of seconds, got {timeout_s!r}")
    summary, arg_docs = _parse_docstring(inspect.getdoc(fn) or "")
    tool_description = (description or summary).strip()
    if not tool_description:
        raise ToolError(f"{tool_name} needs a description: it's how the model decides when to call it. "
                        "Add a docstring, or @tool(description=...)")
    try:
        signature = inspect.signature(fn)
        hints = typing.get_type_hints(fn)
    except (TypeError, ValueError, NameError) as e:
        raise ToolError(f"can't read the parameters of {tool_name}: {e}") from e

    properties: Dict[str, Any] = {}
    required: List[str] = []
    enums: Dict[str, type] = {}
    for param in signature.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.kind is param.POSITIONAL_ONLY:
            raise ToolError(f"{tool_name}({param.name}) is positional-only; the model passes arguments by name")
        hint = hints.get(param.name, Any)
        schema = _schema(hint, f"{tool_name}({param.name})")
        enum_class = _enum_of(hint)
        if enum_class is not None:
            enums[param.name] = enum_class
        if param.name in arg_docs:
            schema = {**schema, "description": arg_docs[param.name]}
        if param.default is param.empty:
            required.append(param.name)
        elif _json_safe(param.default):
            schema = {**schema, "default": json.loads(json.dumps(param.default))}  # as the model will see it
        properties[param.name] = schema
    parameters: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required
    return Tool(fn=fn, name=tool_name, description=tool_description, parameters=parameters,
                timeout_s=float(timeout_s), _signature=signature, _enums=enums)


_SIMPLE = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _schema(hint: Any, where: str) -> Dict[str, Any]:
    """JSON Schema for a type hint. Covers what a tool's parameters realistically use."""
    if hint is Any or hint is inspect.Parameter.empty:
        return {}
    if hint in _SIMPLE:
        return {"type": _SIMPLE[hint]}
    if isinstance(hint, type) and issubclass(hint, enum.Enum):
        values = [member.value for member in hint]
        return {"enum": values, **({"type": _SIMPLE[type(values[0])]} if values and type(values[0]) in _SIMPLE else {})}
    if hint in (dict, Dict, Mapping):
        return {"type": "object"}
    if hint in (list, List, tuple, Tuple):
        return {"type": "array"}
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin is typing.Literal:
        values = list(args)
        kinds = {type(v) for v in values}
        return {"enum": values, **({"type": _SIMPLE[kinds.pop()]} if len(kinds) == 1 and next(iter(kinds)) in _SIMPLE else {})}
    if origin in (Union, types.UnionType):
        options = [a for a in args if a is not type(None)]
        if len(options) == 1:  # Optional[X]: the default says it can be left out
            return _schema(options[0], where)
        return {"anyOf": [_schema(a, where) for a in options]}
    if origin in (list, set, frozenset, collections.abc.Sequence) and len(args) == 1:
        return {"type": "array", "items": _schema(args[0], where)}
    if origin is tuple and len(args) == 2 and args[1] is Ellipsis:  # Tuple[X, ...]
        return {"type": "array", "items": _schema(args[0], where)}
    if origin in (list, tuple, set, frozenset, collections.abc.Sequence):
        return {"type": "array"}
    if origin in (dict, collections.abc.Mapping):
        return {"type": "object"}
    raise ToolError(f"{where}: the type {getattr(hint, '__name__', hint)!r} can't be described to the model; "
                    "use str, int, float, bool, Literal, an Enum, or lists and dicts of those")


def _enum_of(hint: Any) -> Optional[type]:
    """The Enum class behind a hint (X or Optional[X]), if there is one."""
    if typing.get_origin(hint) in (Union, types.UnionType):
        options = [a for a in typing.get_args(hint) if a is not type(None)]
        hint = options[0] if len(options) == 1 else None
    return hint if isinstance(hint, type) and issubclass(hint, enum.Enum) else None


def _parse_docstring(doc: str) -> Tuple[str, Dict[str, str]]:
    """(the summary, {parameter: description}) from a docstring with a Google-style `Args:` section."""
    lines = doc.splitlines()
    summary: List[str] = []
    for line in lines:
        if not line.strip():
            break
        summary.append(line.strip())
    args: Dict[str, str] = {}
    in_args, current, arg_indent = False, None, None
    for line in lines[len(summary):]:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if stripped.lower() in ("args:", "arguments:", "parameters:", "params:"):
            in_args, current, arg_indent = True, None, None
            continue
        if not in_args or not stripped:
            continue
        if indent == 0:  # the next section starts
            in_args = False
            continue
        match = re.match(r"^\*{0,2}(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$", stripped)
        if match and (arg_indent is None or indent <= arg_indent):
            arg_indent = indent
            current = match.group(1)
            args[current] = match.group(2).strip()
        elif current is not None:
            args[current] = f"{args[current]} {stripped}".strip()
    return " ".join(summary), args


def _json_safe(value: Any) -> bool:
    if isinstance(value, enum.Enum):
        return False
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def _as_text(value: Any) -> str:
    if value is None:
        text = "done"
    elif isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(value)
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + " …(cut: the result was too long)"
    return text
