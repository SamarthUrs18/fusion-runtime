#!/usr/bin/env python3
"""A voice agent that answers from your own documents. Run it with:

    export FUSION_LLM_ENGINE_ENV=~/vllm-env    # where vLLM is installed (see the README)
    frun up examples/knowledge_agent.py       # starts vLLM on port 8002, then the agent on 8000
    frun talk                                 # in another terminal, then ask "how long do refunds take?"

Retrieval is a tool: the model calls search_help with the caller's question, gets the few
passages that match, and answers from them. The documents are the Markdown files in
examples/knowledge/, split at their `##` headings. Put your own there, or replace search() with
a call to your own search service (a vector database, Elasticsearch, your help centre's API):
the agent only needs a function that takes a question and returns a few short passages.

Two things are different on a call than in a chat window:

- Every lookup costs a second model round before the caller hears the answer. If your knowledge
  fits in a page or two, put it in the prompt instead: that's faster than any search.
- Passages go into the prompt, so keep them short (a few sentences each, a few per search).
  Long ones slow the model's first word, and the caller is waiting in silence for it.

The search here is BM25 over words, in plain Python with nothing to install, which is fine for a
few hundred passages. It matches words, not meaning: "money back" won't find "refund" unless the
document says both.
"""
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

from fusion_runtime import LLM, STT, TTS, Agent, Turns, tool

KNOWLEDGE = Path(__file__).resolve().parent / "knowledge"
RESULTS = 3

_WORD = re.compile(r"[a-z0-9]+")
# Words that say nothing about the topic, including the ones spoken questions are made of ("how long
# does it take"), which would otherwise match whichever passage happens to use them.
_STOP = {"a", "an", "and", "are", "can", "do", "does", "for", "get", "how", "i", "if", "in", "is", "it",
         "long", "much", "my", "need", "of", "on", "or", "take", "the", "to", "want", "what", "when",
         "where", "will", "with", "you", "your"}


def words(text: str) -> List[str]:
    # A crude plural strip, so "refunds" finds "refund"; the same on both sides, so it only has to agree.
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w
            for w in _WORD.findall(text.lower()) if w not in _STOP]


def load_passages(folder: Path) -> List[Dict[str, str]]:
    """Each `##` section of each Markdown file is one passage, named by its article and heading."""
    passages = []
    for path in sorted(folder.glob("*.md")):
        article = path.stem.replace("_", " ")
        for section in re.split(r"^## ", path.read_text(), flags=re.MULTILINE)[1:]:
            heading, _, body = section.partition("\n")
            passages.append({"source": f"{article}: {heading.strip()}", "text": " ".join(body.split())})
    return passages


class Index:
    """BM25 over a list of passages: the standard keyword ranking, small enough to read."""

    def __init__(self, passages: List[Dict[str, str]], k1: float = 1.5, b: float = 0.75):
        self.passages, self.k1, self.b = passages, k1, b
        self.counts = [Counter(words(p["source"] + " " + p["text"])) for p in passages]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average = sum(self.lengths) / max(len(self.lengths), 1)
        seen_in = Counter(word for c in self.counts for word in c)
        total = len(passages)
        self.idf = {w: math.log(1 + (total - n + 0.5) / (n + 0.5)) for w, n in seen_in.items()}

    def search(self, question: str, limit: int = RESULTS) -> List[Dict[str, str]]:
        query = set(words(question))
        scored = []
        for passage, counts, length in zip(self.passages, self.counts, self.lengths, strict=True):
            score = sum(self.idf[w] * counts[w] * (self.k1 + 1)
                        / (counts[w] + self.k1 * (1 - self.b + self.b * length / self.average))
                        for w in query if w in counts)
            if score > 0:
                scored.append((score, passage))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [passage for _, passage in scored[:limit]]


index = Index(load_passages(KNOWLEDGE))  # built once, when the agent loads, not on every call


@tool(timeout_s=3)
def search_help(question: str) -> dict:
    """Search ShopKart's help articles: returns, refunds, delivery, cancelling orders.

    Args:
        question: What the caller wants to know, in their words, e.g. "how long do refunds take".
    """
    found = index.search(question)
    if not found:
        return {"found": False, "hint": "nothing in the help articles matches; say you don't know"}
    return {"found": True, "passages": found}


agent = Agent(
    name="shopkart-help",
    prompt=(
        "You are the help line for ShopKart, an online shop. For any question about returns, refunds, "
        "delivery or cancelling, call search_help right away and answer only from what it returns; "
        "never answer those from memory. If it finds nothing, say you don't have that information. "
        "You're speaking on a call: answer in one or two short sentences, with no lists or links, "
        "and don't mention the articles."
    ),
    stt=STT("whisper-small"),
    llm=LLM("vllm:hf:Qwen/Qwen2.5-7B-Instruct-AWQ", max_tokens=200, max_tool_rounds=2),
    tts=TTS("kokoro-v1.0", voice="af_heart"),
    turns=Turns(wait_ms=500, interrupt_after_ms=300),
    tools=[search_help],
    greeting="Hi, this is ShopKart help. What would you like to know?",
    profile="production",  # speech on the GPU; the development profile keeps Whisper on the CPU
)
