"""frun bench: the numbers it reports, the capacity line, and what it refuses."""
import pytest
import typer
from fusion_runtime.cli import bench
from fusion_runtime.cli.app import app
from typer.testing import CliRunner

runner = CliRunner()


def results(callers, walls, server=None, failed=0):
    rows = [{"caller": i, "turn": 0, "wall_ms": w, "server_ms": server, "llm_first_token_ms": 40,
             "tts_first_chunk_ms": 300, "tool_calls": 1} for i, w in enumerate(walls)]
    return rows + [{"caller": 99, "turn": 0, "error": "no audio came back"}] * failed


def test_a_round_is_summarized_with_where_the_time_went():
    row = bench.summarize(4, results(4, [800, 900, 1000, 1400], server=700), 12.34)
    assert (row["median_ms"], row["worst_ms"], row["server_median_ms"]) == (950, 1400, 700)
    assert row["stages_median_ms"] == {"llm first token": 40, "tts first audio": 300}
    assert row["tool_calls"] == 4 and row["failed"] == 0
    text = bench.report(row, baseline=500)
    assert "median   950 ms" in text and "1.9x of one caller" in text and "4 tool call(s)" in text


def test_capacity_is_the_most_callers_under_the_target_without_failures():
    rows = [bench.summarize(n, results(n, walls), 1.0) for n, walls in
            ((1, [400]), (4, [700] * 4), (8, [1100] * 8), (12, [1900] * 12))]
    assert bench.capacity(rows, 1500).startswith("Up to 8 caller(s) stayed under 1500 ms")
    rows[2]["failed"] = 1  # a failure at 8 stops the count at 4, even though 8 was fast
    assert "Up to 4 caller(s)" in bench.capacity(rows, 1500)
    assert bench.capacity(rows, 100).startswith("No round stayed under 100 ms")


def test_the_stage_that_grew_is_named_not_a_fixed_hint():
    """Pod test, 13 Oct: on vLLM the old closing line still said to move the LLM to vLLM."""
    from fusion_runtime.cli.bench import bottleneck

    speech_bound = [{"callers": 1, "stages_median_ms": {"stt": 139, "llm first token": 43, "tts first audio": 383}},
                    {"callers": 16, "stages_median_ms": {"stt": 329, "llm first token": 48, "tts first audio": 1540}}]
    said = bottleneck(speech_bound)
    assert said.startswith("Speech is where callers wait: tts first audio went from 383 to 1540 ms")
    assert "LLM held (llm first token 48 ms)" in said and "vLLM" not in said
    llm_bound = [{"callers": 1, "stages_median_ms": {"stt": 70, "llm queue": 0, "tts first audio": 300}},
                 {"callers": 4, "stages_median_ms": {"stt": 80, "llm queue": 2200, "tts first audio": 350}}]
    assert bottleneck(llm_bound).startswith("The LLM is where callers wait: llm queue went from 0 to 2200 ms")
    assert "nothing is queueing" in bottleneck([{**speech_bound[0], "callers": 1}, {**speech_bound[0], "callers": 4}])


def test_the_shipped_recordings_are_found_and_readable():
    for name in bench.RECORDINGS:
        assert len(bench.load_audio(bench.audio_path(name))) > 16000  # over half a second
    with pytest.raises(typer.BadParameter, match="use hello, order or a path"):
        bench.audio_path("nope")


def test_bad_caller_counts_are_refused_with_the_format():
    result = runner.invoke(app, ["bench", "--callers", "1,x"])
    assert result.exit_code != 0 and "numbers separated by commas" in result.output


def test_an_unreachable_server_is_named(monkeypatch):
    monkeypatch.setenv("FUSION_API_KEY", "")
    result = runner.invoke(app, ["bench", "--url", "ws://127.0.0.1:9", "--callers", "1", "--turns", "1"])
    assert result.exit_code == 1 and "Is the server running?" in result.output


def test_a_call_the_server_drops_is_a_failure_not_a_crash(monkeypatch):
    """Pod test, 13 Oct: the server closed calls with 1011 and frun bench died with a traceback."""
    import asyncio

    from fusion_runtime.cli import bench as bench_module
    from websockets.exceptions import ConnectionClosedError
    from websockets.frames import Close

    async def dropped(index, url, key, audio, turns, results):
        raise ConnectionClosedError(Close(1011, "internal error"), None)

    monkeypatch.setattr(bench_module, "one_session", dropped)
    results = asyncio.run(bench_module.run_round(2, "ws://x", "", b"", 1))
    assert len(results) == 2 and all("closed the call (internal error)" in r["error"] for r in results)
    row = bench_module.summarize(2, results, 1.0)
    assert row["failed"] == 2 and "median_ms" not in row
