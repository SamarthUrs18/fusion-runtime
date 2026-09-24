"""Resolver: model references become runtime + family + model facts, from data only."""
import json
import struct

import pytest
from fusion_runtime.catalog import gguf
from fusion_runtime.catalog.entries import ModelEntry, load_catalog
from fusion_runtime.config import DEVELOPMENT_CONFIG, HYBRID_CONFIG
from fusion_runtime.contract.common import InvalidRequest, ModelNotFound, UnsupportedModel
from fusion_runtime.resolver import resolve, resolve_stage_config

# ---- helpers ---------------------------------------------------------------------------

def _gguf_string(text: str) -> bytes:
    data = text.encode()
    return struct.pack("<Q", len(data)) + data


def write_gguf(path, metadata: dict, big_vocab: int = 0) -> None:
    """A minimal GGUF header: enough for the resolver, no tensors."""
    items = []
    for key, value in metadata.items():
        if isinstance(value, str):
            items.append(_gguf_string(key) + struct.pack("<I", 8) + _gguf_string(value))
        else:
            items.append(_gguf_string(key) + struct.pack("<I", 4) + struct.pack("<I", value))
    if big_vocab:
        tokens = b"".join(_gguf_string(f"tok{i}") for i in range(big_vocab))
        items.append(_gguf_string("tokenizer.ggml.tokens") + struct.pack("<I", 9) + struct.pack("<IQ", 8, big_vocab) + tokens)
        items.append(_gguf_string("tokenizer.ggml.scores") + struct.pack("<I", 9) + struct.pack("<IQ", 6, big_vocab)
                     + b"\0" * 4 * big_vocab)
    header = b"GGUF" + struct.pack("<IQQ", 3, 0, len(items))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + b"".join(items))


LLAMA_META = {
    "general.architecture": "llama", "general.name": "tiny-llama", "llama.context_length": 8192,
    "llama.block_count": 16, "llama.embedding_length": 2048, "llama.attention.head_count": 32,
    "llama.attention.head_count_kv": 8, "tokenizer.chat_template": "{{ messages }}",
}


def write_whisper(folder, vocab_size: int) -> None:
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps({"lang_ids": [50259], "suppress_ids_begin": [220]}))
    (folder / "model.bin").write_bytes(b"x")
    (folder / "vocabulary.txt").write_text("\n".join(f"t{i}" for i in range(vocab_size)) + "\n")


def empty_catalog():
    return {}


# ---- GGUF metadata ---------------------------------------------------------------------

def test_gguf_metadata_is_read_and_large_arrays_skipped(tmp_path):
    path = tmp_path / "m.gguf"
    write_gguf(path, LLAMA_META, big_vocab=500)
    metadata = gguf.read_metadata(path)
    assert "tokenizer.ggml.tokens" not in metadata
    facts = gguf.summarize(metadata)
    assert facts["architecture"] == "llama"
    assert facts["context_length"] == 8192
    assert facts["chat_template"] == "{{ messages }}"
    # 2 (K and V) × 16 layers × 8 KV heads × 64 dims × 2 bytes
    assert facts["kv_bytes_per_token"] == 2 * 16 * 8 * 64 * 2


def test_non_gguf_and_truncated_files_are_rejected(tmp_path):
    (tmp_path / "a.gguf").write_bytes(b"NOPE")
    with pytest.raises(gguf.GGUFError, match="not a GGUF"):
        gguf.read_metadata(tmp_path / "a.gguf")
    write_gguf(tmp_path / "b.gguf", LLAMA_META)
    (tmp_path / "b.gguf").write_bytes((tmp_path / "b.gguf").read_bytes()[:40])
    with pytest.raises(gguf.GGUFError, match="unexpected end"):
        gguf.read_metadata(tmp_path / "b.gguf")


# ---- formats -----------------------------------------------------------------------------

def test_any_gguf_file_runs_on_llama_cpp(tmp_path):
    write_gguf(tmp_path / "anything.gguf", LLAMA_META)
    resolved = resolve("llm", str(tmp_path / "anything.gguf"), catalog=empty_catalog(), root=tmp_path)
    assert (resolved.spec.runtime, resolved.format, resolved.source) == ("llama_cpp", "gguf", "path")
    assert resolved.metadata["architecture"] == "llama"
    assert resolved.describe()["chat_template"] is True  # presence only, never the template text


def test_paths_are_also_found_under_the_model_directory(tmp_path):
    write_gguf(tmp_path / "llm" / "x.gguf", LLAMA_META)
    assert resolve("llm", "llm/x.gguf", catalog=empty_catalog(), root=tmp_path).spec.model == str(tmp_path / "llm" / "x.gguf")


def test_folder_with_one_gguf_resolves_to_it(tmp_path):
    write_gguf(tmp_path / "m" / "only.gguf", LLAMA_META)
    assert resolve("llm", str(tmp_path / "m"), catalog=empty_catalog(), root=tmp_path).spec.model.endswith("only.gguf")


def test_split_gguf_needs_first_part_and_all_parts(tmp_path):
    first = tmp_path / "big-00001-of-00002.gguf"
    write_gguf(first, LLAMA_META)
    with pytest.raises(ModelNotFound, match="big-00002-of-00002.gguf"):
        resolve("llm", str(first), catalog=empty_catalog(), root=tmp_path)
    write_gguf(tmp_path / "big-00002-of-00002.gguf", LLAMA_META)
    with pytest.raises(InvalidRequest, match="use the first part: big-00001-of-00002.gguf"):
        resolve("llm", str(tmp_path / "big-00002-of-00002.gguf"), catalog=empty_catalog(), root=tmp_path)
    assert resolve("llm", str(first), catalog=empty_catalog(), root=tmp_path).format == "gguf"


def test_whisper_folders_run_on_ctranslate2_with_languages_from_the_vocabulary(tmp_path):
    write_whisper(tmp_path / "english", 51864)
    write_whisper(tmp_path / "multi", 51866)
    english = resolve("stt", str(tmp_path / "english"), catalog=empty_catalog(), root=tmp_path)
    multi = resolve("stt", str(tmp_path / "multi"), catalog=empty_catalog(), root=tmp_path)
    assert (english.spec.runtime, english.spec.family, english.languages) == ("ctranslate2", "whisper", ("en",))
    assert multi.languages is None


def test_onnx_needs_a_family_unless_files_show_it(tmp_path):
    (tmp_path / "plain").mkdir()
    (tmp_path / "plain" / "model.onnx").write_bytes(b"x")
    with pytest.raises(UnsupportedModel, match="family"):
        resolve("tts", str(tmp_path / "plain"), catalog=empty_catalog(), root=tmp_path)
    assert resolve("tts", str(tmp_path / "plain"), family="piper", catalog=empty_catalog(), root=tmp_path).spec.family == "piper"

    kokoro = tmp_path / "kokoro"
    (kokoro / "onnx").mkdir(parents=True)
    (kokoro / "onnx" / "model.onnx").write_bytes(b"x")
    (kokoro / "voices").mkdir()
    for voice in ("af_heart", "bm_lewis"):
        (kokoro / "voices" / f"{voice}.bin").write_bytes(b"x")
    resolved = resolve("tts", str(kokoro), catalog=empty_catalog(), root=tmp_path)
    assert (resolved.spec.runtime, resolved.spec.family, resolved.voices) == ("onnx", "kokoro", ("af_heart", "bm_lewis"))


def test_safetensors_points_at_a_serving_engine(tmp_path):
    (tmp_path / "hf").mkdir()
    (tmp_path / "hf" / "model.safetensors").write_bytes(b"x")
    with pytest.raises(UnsupportedModel, match="vLLM"):
        resolve("llm", str(tmp_path / "hf"), catalog=empty_catalog(), root=tmp_path)


def test_wrong_stage_for_a_format_is_explained(tmp_path):
    write_gguf(tmp_path / "m.gguf", LLAMA_META)
    with pytest.raises(UnsupportedModel, match="serves llm, not tts"):
        resolve("tts", str(tmp_path / "m.gguf"), catalog=empty_catalog(), root=tmp_path)


# ---- other sources ---------------------------------------------------------------------

def test_urls_use_the_openai_http_runtime(tmp_path):
    resolved = resolve("llm", "http://localhost:8000/v1", options={"model_name": "m"}, catalog=empty_catalog(), root=tmp_path)
    assert (resolved.spec.runtime, resolved.format, resolved.spec.model) == ("openai_http", "http", "http://localhost:8000/v1")


def test_plugin_runtimes_receive_references_unchanged(tmp_path):
    resolved = resolve("tts", "voice-pack://studio", runtime="my_pkg.tts:Engine", catalog=empty_catalog(), root=tmp_path)
    assert (resolved.source, resolved.spec.runtime, resolved.spec.model) == ("plugin", "my_pkg.tts:Engine", "voice-pack://studio")


def test_hf_reference_must_be_owner_slash_repo(tmp_path):
    with pytest.raises(InvalidRequest, match="hf:owner/repo"):
        resolve("llm", "hf:not-a-repo", catalog=empty_catalog(), root=tmp_path)


def test_hf_reference_uses_the_matching_catalog_entry(tmp_path):
    entry = ModelEntry(id="tiny", stage="llm", description="", license="MIT", source="huggingface", repo="org/tiny",
                       revision="a" * 40, local_dir="llm", path="llm/tiny.gguf", files={"tiny.gguf": 1}, runtime="llama_cpp")
    write_gguf(tmp_path / "llm" / "tiny.gguf", LLAMA_META)
    size = (tmp_path / "llm" / "tiny.gguf").stat().st_size
    entry = ModelEntry(**{**entry.__dict__, "files": {"tiny.gguf": size}})
    resolved = resolve("llm", "hf:org/tiny", catalog={"tiny": entry}, root=tmp_path)
    assert (resolved.source, resolved.catalog_id) == ("catalog", "tiny")


def test_catalog_models_not_downloaded_say_how_to_get_them(tmp_path):
    catalog = load_catalog()
    with pytest.raises(ModelNotFound, match="frun models pull qwen2.5-0.5b-q4"):
        resolve("llm", "qwen2.5-0.5b-q4", catalog=catalog, root=tmp_path)


def test_catalog_id_for_another_stage_is_rejected(tmp_path):
    with pytest.raises(InvalidRequest, match="is a tts model"):
        resolve("stt", "kokoro-v1.0", catalog=load_catalog(), root=tmp_path)


def test_unknown_reference_suggests_a_close_catalog_id(tmp_path):
    with pytest.raises(ModelNotFound, match="Did you mean 'qwen2.5-0.5b-q4'"):
        resolve("llm", "qwen2.5-0.5b", catalog=load_catalog(), root=tmp_path)


def test_every_catalog_model_declares_its_runtime():
    for entry in load_catalog().values():
        assert entry.runtime, f"{entry.id} needs a runtime"


# ---- today's config -----------------------------------------------------------------------

def test_hybrid_llm_resolves_to_http_without_copying_the_key(tmp_path):
    resolved = resolve_stage_config("llm", HYBRID_CONFIG.llm, catalog=load_catalog(), root=tmp_path)
    assert resolved.spec.runtime == "openai_http"
    assert resolved.spec.options["model_name"] == "gpt-4o-mini"
    assert "api_key" not in resolved.spec.options


def test_development_profile_resolves_when_models_are_present(tmp_path):
    """Uses whatever models are downloaded in this checkout; skips stages that aren't."""
    catalog = load_catalog()
    for stage in ("stt", "llm", "tts"):
        try:
            resolved = resolve_stage_config(stage, getattr(DEVELOPMENT_CONFIG, stage), catalog=catalog)
        except ModelNotFound:
            continue
        assert resolved.source == "catalog"
        assert resolved.spec.runtime == {"stt": "ctranslate2", "llm": "llama_cpp", "tts": "onnx"}[stage]


# ---- downloading any Hugging Face model -------------------------------------------------------

def test_hf_reference_parsing_and_folder_names(tmp_path):
    from fusion_runtime.catalog import hf_local_dir, hf_reference

    assert hf_reference("hf:Systran/faster-whisper-small") == ("Systran/faster-whisper-small", None, None)
    assert hf_reference("hf:org/repo@abc123") == ("org/repo", "abc123", None)
    assert hf_reference("hf:org/repo/model-q4_k_m.gguf") == ("org/repo", None, "model-q4_k_m.gguf")
    assert hf_reference("hf:org/repo@abc/onnx/model.onnx") == ("org/repo", "abc", "onnx/model.onnx")
    assert hf_reference("qwen2.5-0.5b-q4") is None
    assert hf_local_dir(tmp_path, "org/repo").name == "org--repo"
    assert hf_local_dir(tmp_path, "org/repo", "abc").name == "org--repo@abc"


def test_a_downloaded_hf_model_resolves_from_the_model_directory(tmp_path):
    from fusion_runtime.catalog import hf_local_dir

    folder = hf_local_dir(tmp_path, "Systran/faster-whisper-tiny")
    write_whisper(folder, 51865)
    resolved = resolve("stt", "hf:Systran/faster-whisper-tiny", catalog=empty_catalog(), root=tmp_path)
    assert (resolved.source, resolved.spec.runtime) == ("huggingface", "ctranslate2")
    assert resolved.spec.model == str(folder)


def test_an_undownloaded_hf_model_says_how_to_get_it(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # don't reach the network from a test
    with pytest.raises(ModelNotFound, match=r"frun models pull hf:org/not-downloaded"):
        resolve("llm", "hf:org/not-downloaded", catalog=empty_catalog(), root=tmp_path)


async def test_the_server_downloads_a_model_the_agent_asks_for(tmp_path, monkeypatch):
    """A deployment starts from an empty disk: the server fetches what the agent names."""
    from fusion_runtime.agent import STT, Agent
    from fusion_runtime.catalog import hf_local_dir
    from fusion_runtime.engine import PipelineOrchestrator

    monkeypatch.setenv("FUSION_MODEL_DIR", str(tmp_path))
    pulled = []

    def fake_pull(ref, root, log=print, force=False, allow_patterns=None):
        pulled.append(ref)
        write_whisper(hf_local_dir(root, "Systran/faster-whisper-tiny"), 51865)

    monkeypatch.setattr("fusion_runtime.catalog.pull_hf", fake_pull)
    agent = Agent(prompt="hi", stt=STT("hf:Systran/faster-whisper-tiny"))
    orch = PipelineOrchestrator(agent.config({}))
    await orch._ensure_downloaded("stt", orch.config.stt)
    assert pulled == ["hf:Systran/faster-whisper-tiny"]
    await orch._ensure_downloaded("stt", orch.config.stt)  # already there: no second download
    assert pulled == ["hf:Systran/faster-whisper-tiny"]


async def test_auto_download_can_be_switched_off(tmp_path, monkeypatch):
    from fusion_runtime.agent import STT, Agent
    from fusion_runtime.engine import PipelineOrchestrator

    monkeypatch.setenv("FUSION_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("FUSION_AUTO_DOWNLOAD", "0")
    monkeypatch.setattr("fusion_runtime.catalog.pull_hf",
                        lambda *a, **k: pytest.fail("must not download when switched off"))
    orch = PipelineOrchestrator(Agent(prompt="hi", stt=STT("hf:org/repo")).config({}))
    await orch._ensure_downloaded("stt", orch.config.stt)


def test_a_repo_with_several_copies_of_a_model_asks_which_one(monkeypatch):
    """A GGUF repo holds the same model 9 times over; downloading them all is gigabytes."""
    from fusion_runtime.catalog import DownloadError, download

    monkeypatch.setattr(download, "hf_files", lambda repo, revision=None, token=None: [
        ("config.json", 2_000),
        ("model-q4_k_m.gguf", 491_000_000),
        ("model-q8_0.gguf", 675_000_000),
    ])
    with pytest.raises(DownloadError, match=r"(?s)holds 2 versions.*hf:org/repo/model-q4_k_m.gguf"):
        download._files_to_fetch("org/repo", None, None, None)

    # naming one takes that file, plus the small files every copy needs
    chosen = [name for name, _ in download._files_to_fetch("org/repo", None, "model-q4_k_m.gguf", None)]
    assert chosen == ["model-q4_k_m.gguf", "config.json"]

    with pytest.raises(DownloadError, match="has no file 'nope.gguf'"):
        download._files_to_fetch("org/repo", None, "nope.gguf", None)


def test_a_repo_with_one_model_needs_no_file_name(monkeypatch):
    from fusion_runtime.catalog import download

    monkeypatch.setattr(download, "hf_files", lambda repo, revision=None, token=None: [
        ("config.json", 2_000), ("model.bin", 145_000_000), ("vocabulary.txt", 460_000),
    ])
    chosen = [name for name, _ in download._files_to_fetch("Systran/faster-whisper-base", None, None, None)]
    assert chosen == ["config.json", "model.bin", "vocabulary.txt"]


def test_a_named_file_resolves_from_the_downloaded_folder(tmp_path):
    from fusion_runtime.catalog import hf_local_dir

    folder = hf_local_dir(tmp_path, "org/gguf-repo")
    write_gguf(folder / "model-q4_k_m.gguf", LLAMA_META)
    resolved = resolve("llm", "hf:org/gguf-repo/model-q4_k_m.gguf", catalog=empty_catalog(), root=tmp_path)
    assert resolved.spec.model.endswith("model-q4_k_m.gguf") and resolved.format == "gguf"

    with pytest.raises(ModelNotFound, match="isn't in"):
        resolve("llm", "hf:org/gguf-repo/other.gguf", catalog=empty_catalog(), root=tmp_path)


def test_a_repo_written_without_the_hf_prefix_says_so(tmp_path):
    with pytest.raises(ModelNotFound, match=r"Write it as hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF, and name a file"):
        resolve("llm", "Qwen/Qwen2.5-0.5B-Instruct-GGUF", catalog=empty_catalog(), root=tmp_path)
    # one that already names a file shouldn't be told to add another
    with pytest.raises(ModelNotFound, match=r"Write it as hf:org/repo/onnx/model.onnx$"):
        resolve("tts", "org/repo/onnx/model.onnx", catalog=empty_catalog(), root=tmp_path)
    # a plain typo still gets the catalog suggestion
    with pytest.raises(ModelNotFound, match="Did you mean"):
        resolve("llm", "qwen2.5-0.5b", catalog=load_catalog(), root=tmp_path)


def test_a_served_model_resolves_to_its_server_under_its_repo_id(tmp_path):
    """vllm:/sglang:/llama_server: in front of a model means "I started that server": talk to it
    over the OpenAI API at its usual address, asking for the model by the id it was loaded with."""
    from fusion_runtime.agent import LLM

    runtime, ref = LLM("vllm:hf:Qwen/Qwen2.5-7B-Instruct-AWQ@main").split()
    vllm = resolve("llm", ref, runtime=runtime, catalog=empty_catalog(), root=tmp_path)
    assert (vllm.spec.runtime, vllm.spec.model) == ("openai_http", "http://localhost:8000/v1")
    assert vllm.spec.options["model_name"] == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    assert vllm.metadata["served_by"] == "vLLM"

    sglang = resolve("llm", "hf:org/model", runtime="sglang", options={"url": "http://gpu:30000/v1"},
                     catalog=empty_catalog(), root=tmp_path)
    assert sglang.spec.model == "http://gpu:30000/v1" and "url" not in sglang.spec.options
    assert resolve("llm", "./model.gguf", runtime="llama_server", catalog=empty_catalog(),
                   root=tmp_path).spec.model == "http://localhost:8080/v1"

    by_url = resolve("llm", "http://gpu:8000/v1", runtime="vllm", options={"model_name": "m"},
                     catalog=empty_catalog(), root=tmp_path)
    assert by_url.spec.model == "http://gpu:8000/v1"
    with pytest.raises(InvalidRequest, match="model_name"):
        resolve("llm", "http://gpu:8000/v1", runtime="vllm", catalog=empty_catalog(), root=tmp_path)
    with pytest.raises(UnsupportedModel, match="serves language models"):
        resolve("stt", "hf:org/whisper", runtime="vllm", catalog=empty_catalog(), root=tmp_path)


def test_an_agent_names_a_served_model_and_keeps_its_settings(tmp_path, monkeypatch):
    from fusion_runtime.agent import LLM, Agent

    monkeypatch.delenv("FUSION_LLM_URL", raising=False)
    config = Agent(llm=LLM("sglang:hf:org/model", max_tokens=100, extra_body={"top_k": 20})).config()
    resolved = resolve_stage_config("llm", config.llm, catalog=empty_catalog(), root=tmp_path)
    assert resolved.spec.model == "http://localhost:30000/v1"
    assert resolved.spec.options["max_tokens"] == 100 and resolved.spec.options["extra_body"] == {"top_k": 20}


def test_unknown_settings_are_refused_with_the_likely_one(tmp_path, monkeypatch):
    from fusion_runtime.agent import LLM, STT, Agent

    monkeypatch.delenv("FUSION_LLM_URL", raising=False)
    gguf_file = tmp_path / "m.gguf"
    gguf_file.write_bytes(b"")

    def resolve_llm(llm):
        return resolve_stage_config("llm", Agent(llm=llm).config().llm, catalog=empty_catalog(), root=tmp_path)

    with pytest.raises(InvalidRequest, match="has no setting 'n_gpu_layer'.*Did you mean 'n_gpu_layers'"):
        resolve_llm(LLM("http://x/v1", model_name="m", n_gpu_layer=20))
    with pytest.raises(InvalidRequest, match="has no setting 'extra_bdy'.*'extra_body'"):
        resolve_llm(LLM("http://x/v1", model_name="m", extra_bdy={}))
    # Server launch settings get pointed at the server's own flags
    with pytest.raises(InvalidRequest, match=r"gpu_memory is a setting for starting vLLM.*--gpu-memory-utilization"):
        resolve_llm(LLM("vllm:hf:org/model", gpu_memory=0.6))
    whisper = tmp_path / "whisper"
    whisper.mkdir()
    (whisper / "model.bin").write_bytes(b"")
    (whisper / "config.json").write_text('{"lang_ids": [1]}')
    with pytest.raises(InvalidRequest, match="beam_sise.*Did you mean 'beam_size'"):
        resolve_stage_config("stt", Agent(stt=STT(str(whisper), beam_sise=2)).config().stt,
                             catalog=empty_catalog(), root=tmp_path)
    # Settings the runtime reads pass, and plugins take whatever they take
    resolve_llm(LLM("vllm:hf:org/model", url="http://gpu:8000/v1", timeout_s=5))
    resolve_llm(LLM("my-model", runtime="my_pkg.llm:Custom", anything=1))


def test_pointing_the_llm_elsewhere_by_env_drops_the_old_runtime_s_settings(tmp_path):
    from fusion_runtime.agent import LLM, Agent

    agent = Agent(llm=LLM("qwen2.5-0.5b-q4", flash_attn=True, extra_body={"top_k": 5}))
    config = agent.config({"FUSION_LLM_URL": "http://gpu:8000/v1", "FUSION_LLM_MODEL": "m"})
    resolved = resolve_stage_config("llm", config.llm, catalog=empty_catalog(), root=tmp_path)
    assert resolved.spec.runtime == "openai_http"
    assert "flash_attn" not in resolved.spec.options and resolved.spec.options["extra_body"] == {"top_k": 5}
