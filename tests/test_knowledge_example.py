"""examples/knowledge_agent.py: the agent loads, and its search finds the passage a caller is asking about."""
import importlib.util
from pathlib import Path

import pytest
from fusion_runtime.agent import load_agent

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "knowledge_agent.py"


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location("knowledge_agent_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_agent_file_loads_with_its_search_tool():
    agent = load_agent(EXAMPLE)
    assert [t.name for t in agent.tools] == ["search_help"]


def test_every_heading_is_a_passage(example):
    sources = [p["source"] for p in example.index.passages]
    assert "refunds: Cash on delivery orders" in sources
    assert all(p["text"] and "##" not in p["text"] for p in example.index.passages)


@pytest.mark.parametrize("question, expected", [
    ("how long do refunds take", "refunds:"),
    ("I paid cash, where does my refund go", "refunds: Cash on delivery orders"),
    ("can I cancel after it shipped", "cancellations: After it ships"),
    ("how much is delivery", "delivery: Delivery charges"),
    ("my refund hasn't come", "refunds: A refund that hasn't arrived"),
])
def test_finds_what_the_caller_asked_about(example, question, expected):
    assert example.index.search(question)[0]["source"].startswith(expected)


def test_question_words_alone_match_nothing(example):
    # "how long does it take" is in many questions; on its own it says nothing about the topic.
    assert example.index.search("how long does it take") == []


def test_nothing_found_tells_the_model_to_say_so(example):
    assert example.search_help("what's the weather today")["found"] is False
    assert example.search_help("how long do refunds take")["found"] is True
