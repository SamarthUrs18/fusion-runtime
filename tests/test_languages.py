"""Languages end to end: sentence splitting, word matching, and language settings per stage."""
import pytest

from fusion_runtime.config import TTSConfig, load_profile
from fusion_runtime.contract import InvalidRequest, ModelSpec, STTRequest
from fusion_runtime.engine import PipelineOrchestrator
from fusion_runtime.config import TurnDetectionConfig
from fusion_runtime.engine.text import speakable_segments, words
from fusion_runtime.runtimes.ctranslate2.stt import CTranslate2STT
from fusion_runtime.runtimes.onnx.kokoro import KokoroFamily


async def split(*parts, max_chars=100):
    async def tokens():
        for part in parts:
            yield part
    return [s async for s in speakable_segments(tokens(), max_chars=max_chars)]


# ---- sentences ------------------------------------------------------------------------------

async def test_hindi_sentences_end_at_the_danda():
    assert await split("हाँ,", " आपकी", " टेबल", " बुक", " हो", " गई", " है।", " शाम", " सात", " बजे", " मिलते", " हैं।") == [
        "हाँ, आपकी टेबल बुक हो गई है।", " शाम सात बजे मिलते हैं।",
    ]


async def test_chinese_and_japanese_sentences_split_without_spaces():
    assert await split("好的", "。", "明天", "见", "！") == ["好的。", "明天见！"]
    assert await split("はい", "。", "また", "ね", "？") == ["はい。", "またね？"]


async def test_arabic_question_mark_ends_a_sentence():
    assert await split("هل", " تريد", " طاولة", "؟", " حسنا") == ["هل تريد طاولة؟", " حسنا"]


async def test_decimals_and_abbreviations_are_not_cut():
    assert await split("It", " costs", " 3", ".", "5", " dollars", ".", " See", " you", ".") == [
        "It costs 3.5 dollars.", "See you.",
    ]
    assert await split("Call", " Dr", ".", "Rao", " today", ".") == ["Call Dr.Rao today."]


async def test_long_runs_break_at_a_clause_mark_in_any_script():
    parts = ["नमस्ते"] + [" शब्द"] * 6 + ["،"] + [" और"] * 20
    segments = await split(*parts, max_chars=60)
    assert segments[0].endswith("،") and "".join(segments) == "".join(parts)


async def test_quotes_after_a_sentence_end_stay_with_it():
    assert await split("He", " said", ' "yes', '."', " Then", " left", ".") == ['He said "yes."', "Then left."]


# ---- words (echo guard) --------------------------------------------------------------------------

def test_words_keep_non_latin_scripts():
    assert words("Hello, World!") == ["hello", "world"]
    assert words("आपकी टेबल बुक हो गई है।") == ["आपकी", "टेबल", "बुक", "हो", "गई", "है"]
    assert words("明天见！") == ["明", "天", "见"]


def test_self_echo_guard_works_for_hindi():
    bot = "आपकी टेबल शाम सात बजे के लिए बुक हो गई है।"
    echo = "टेबल शाम सात बजे के लिए बुक"
    assert PipelineOrchestrator._looks_like_self_echo(echo, bot, TurnDetectionConfig())
    assert not PipelineOrchestrator._looks_like_self_echo("क्या पार्किंग है?", bot, TurnDetectionConfig())


# ---- language settings -------------------------------------------------------------------------

class _EnglishOnlyModel:
    class model:
        is_multilingual = False


async def test_english_only_whisper_refuses_another_language_at_load(tmp_path, monkeypatch):
    (tmp_path / "model.bin").write_bytes(b"x")
    stt = CTranslate2STT(ModelSpec(stage="stt", runtime="ctranslate2", model=str(tmp_path),
                                   options={"language": "hi", "warmup": False}))
    monkeypatch.setattr(stt, "_build", lambda path: _EnglishOnlyModel())
    with pytest.raises(InvalidRequest, match="English-only.*multilingual"):
        await stt.load()


async def test_english_only_whisper_reports_other_languages_per_request(tmp_path):
    stt = CTranslate2STT(ModelSpec(stage="stt", runtime="ctranslate2", model=str(tmp_path)))
    stt.model = _EnglishOnlyModel()
    stt._languages = ("en",)
    results = await stt.transcribe([STTRequest(audio=b"\x00\x00" * 160, language="hi")])
    assert isinstance(results[0], InvalidRequest)


def test_kokoro_language_comes_from_the_request_or_the_voice():
    family = KokoroFamily(model_path=None, options={})
    assert family.language_for("af_heart", None) == "en-us"
    assert family.language_for("bm_lewis", None) == "en-gb"
    assert family.language_for("hf_alpha", None) == "hi"
    assert family.language_for("af_heart", "en") == "en-us"
    assert family.language_for("af_heart", "pt_BR") == "pt-br"


def test_tts_language_is_a_setting():
    assert TTSConfig().language is None
    assert TTSConfig(language="hi").language == "hi"


def test_turn_settings_from_the_environment():
    config = load_profile("development", {"FUSION_TURN_WAIT_MS": "1200", "FUSION_TURN_DETECTOR": "my_pkg.turns:Model"})
    assert (config.turn_detection.min_silence_ms, config.turn_detection.runtime) == (1200, "my_pkg.turns:Model")
    with pytest.raises(ValueError, match="whole number"):
        load_profile("development", {"FUSION_TURN_WAIT_MS": "soon"})
    interrupt = load_profile("development", {"FUSION_INTERRUPT_AFTER_MS": "500"})
    assert interrupt.turn_detection.barge_in_min_speech_ms == 500
    with pytest.raises(ValueError, match="negative"):
        load_profile("development", {"FUSION_INTERRUPT_AFTER_MS": "-1"})


def test_defaults_answer_after_half_a_second_and_interrupt_after_300_ms():
    config = load_profile("development", {})
    assert config.turn_detection.min_silence_ms == 500
    assert config.turn_detection.barge_in_min_speech_ms == 300


async def test_end_of_reply_speaks_the_last_sentence_immediately():
    from fusion_runtime.engine.text import END_OF_REPLY

    seen = []

    async def tokens():
        for part in ("Sure", ".", END_OF_REPLY):
            yield part
        seen.append("generator resumed after the last sentence was taken")

    stream = speakable_segments(tokens())
    assert await stream.__anext__() == "Sure."
    assert seen == [], "the last sentence must come out before the reply stream finishes"
