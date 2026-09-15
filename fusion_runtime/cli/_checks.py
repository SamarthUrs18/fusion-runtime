"""Environment checks used by `frun doctor` (and partly by `frun up`).

Each check returns one or more CheckResult. Heavy libraries are imported
inside the checks, so importing this module stays cheap. A check that crashes
is reported as a failure; doctor itself must never crash.
"""
import contextlib
import errno
import os
import platform
import shutil
import socket
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Union

OK, INFO, WARN, FAIL = "ok", "info", "warn", "fail"


@dataclass
class CheckResult:
    status: str  # ok | info | warn | fail
    message: str
    fix: Optional[str] = None


Check = Callable[[], Union[CheckResult, List[CheckResult]]]


# ---- helpers shared with `frun up` ---------------------------------------------

def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # same as uvicorn, so TIME_WAIT isn't "in use"
        try:
            s.bind((host, port))
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                return True
            raise
    return False


def missing_models(profile) -> List[str]:
    from fusion_runtime.catalog import entries_for_profile, is_installed
    from fusion_runtime.cli._common import profile_config
    from fusion_runtime.config import model_dir

    root = model_dir()
    return [e.id for e in entries_for_profile(profile_config(profile)) if not is_installed(e, root)]


@contextlib.contextmanager
def quiet_native_output() -> Iterator[None]:
    """Silence C libraries (llama.cpp's Metal setup, ONNX Runtime) and Python warnings."""
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


def versions_match(a: str, b: str) -> bool:
    """torch and torchaudio must be the same release; ignore build tags like +cpu."""
    return a.split("+")[0] == b.split("+")[0]


def _gb(num_bytes: float) -> str:
    return f"{num_bytes / 1e9:.1f} GB"


# ---- System --------------------------------------------------------------------

def check_python() -> CheckResult:
    version = ".".join(map(str, sys.version_info[:3]))
    if sys.version_info < (3, 11):
        return CheckResult(FAIL, f"Python {version}", "fusion-runtime needs Python 3.11 or newer")
    return CheckResult(OK, f"Python {version}")


def _cpu_name() -> str:
    if sys.platform == "darwin":
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5)
        if out.stdout.strip():
            return out.stdout.strip()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def check_machine() -> CheckResult:
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    message = f"{_cpu_name()} · {ram / 2**30:.0f} GB RAM · {platform.system()} {platform.release()}"  # as sold: 8 GB
    if ram < 6 * 2**30:
        return CheckResult(WARN, message, "Under 6 GB of RAM: even the development profile may run out of memory")
    return CheckResult(OK, message)


def check_disk() -> CheckResult:
    from fusion_runtime.config import model_dir

    probe = model_dir()
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    message = f"{_gb(free)} free for models"
    if free < 1e9:
        return CheckResult(FAIL, message, "Free up disk space, or set FUSION_MODEL_DIR to a bigger disk")
    if free < 6e9:
        return CheckResult(WARN, message, "Not enough for the production model (needs ~5.2 GB)")
    return CheckResult(OK, message)


def check_port() -> CheckResult:
    if port_in_use("127.0.0.1", 8000):
        return CheckResult(WARN, "Port 8000 is in use (a `frun up` may already be running)",
                           "Stop it, or start with: frun up --port 8001")
    return CheckResult(OK, "Port 8000 is free")


# ---- Libraries -----------------------------------------------------------------

def check_torch_pair() -> CheckResult:
    try:
        with quiet_native_output():
            import torch
            import torchaudio
    except ImportError as e:
        return CheckResult(FAIL, f"torch/torchaudio not importable: {e}", "pip install -e .")
    except OSError as e:  # torchaudio's compiled extension doesn't match torch
        return CheckResult(FAIL, f"torchaudio won't load against this torch: {e}",
                           "Reinstall both at the same version: pip install torch==2.9.1 torchaudio==2.9.1")
    if not versions_match(torch.__version__, torchaudio.__version__):
        return CheckResult(
            FAIL, f"torch {torch.__version__} and torchaudio {torchaudio.__version__} don't match",
            f"Silero VAD fails silently like this. Fix: pip install torch=={torch.__version__.split('+')[0]} "
            f"torchaudio=={torch.__version__.split('+')[0]}",
        )
    return CheckResult(OK, f"torch {torch.__version__} and torchaudio {torchaudio.__version__} match")


def check_silero_loads() -> CheckResult:
    from fusion_runtime.catalog import is_installed, load_catalog
    from fusion_runtime.config import model_dir

    if not is_installed(load_catalog()["silero-vad"], model_dir()):
        return CheckResult(FAIL, "Silero VAD isn't downloaded", "frun models pull --vad")
    try:
        with quiet_native_output():
            import torch

            torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True, skip_validation=True, verbose=False)
    except Exception as e:
        return CheckResult(FAIL, f"Silero VAD is downloaded but won't load: {type(e).__name__}: {e}",
                           "Without VAD there's no speech detection or interruptions. Run the torch check above first")
    return CheckResult(OK, "Silero VAD loads")


def _import_check(module: str, package: str, why: str) -> CheckResult:
    from importlib.metadata import PackageNotFoundError, version

    try:
        with quiet_native_output():
            __import__(module)
    except Exception as e:
        return CheckResult(FAIL, f"{package} won't import ({why}): {type(e).__name__}: {e}", "pip install -e .")
    try:
        return CheckResult(OK, f"{package} {version(package)}")
    except PackageNotFoundError:
        return CheckResult(OK, package)


def check_engines() -> List[CheckResult]:
    return [
        _import_check("faster_whisper", "faster-whisper", "speech-to-text"),
        _import_check("llama_cpp", "llama-cpp-python", "LLM"),
        _import_check("kokoro_onnx", "kokoro-onnx", "text-to-speech"),
        _import_check("onnxruntime", "onnxruntime", "text-to-speech"),
        _import_check("fusion_runtime.server", "fastapi", "server"),
    ]


# ---- Acceleration --------------------------------------------------------------

def _cuda_device_count() -> int:
    try:
        with quiet_native_output():
            import ctranslate2

            return ctranslate2.get_cuda_device_count()
    except Exception:
        return 0


def check_acceleration() -> List[CheckResult]:
    from fusion_runtime.config import DEVELOPMENT_CONFIG

    results = []
    try:
        with quiet_native_output():
            import llama_cpp

            gpu_offload = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception as e:
        return [CheckResult(FAIL, f"Can't query llama.cpp: {type(e).__name__}: {e}")]
    backend = "Metal" if sys.platform == "darwin" else "CUDA"
    if gpu_offload:
        results.append(CheckResult(OK, f"llama.cpp can use the GPU ({backend})"))
        if DEVELOPMENT_CONFIG.llm.n_gpu_layers == 0:
            results.append(CheckResult(INFO, "The development profile still runs the LLM on CPU (n_gpu_layers=0)"))
    else:
        results.append(CheckResult(INFO, "llama.cpp was built for CPU only",
                                   "For a GPU build, reinstall llama-cpp-python with the CMAKE_ARGS for your GPU"))

    cuda_devices = _cuda_device_count()
    if cuda_devices:
        results.append(CheckResult(OK, f"{cuda_devices} NVIDIA GPU{'s' if cuda_devices > 1 else ''} visible to CUDA"))
    else:
        results.append(CheckResult(INFO, "No NVIDIA GPU: the production profile needs one"))
    return results


# ---- Models --------------------------------------------------------------------

def check_models() -> List[CheckResult]:
    from fusion_runtime.cli._common import Profile, short_path
    from fusion_runtime.config import model_dir, model_dir_source

    results = [CheckResult(INFO, f"Model directory: {short_path(model_dir())} ({model_dir_source()})")]
    dev_missing = missing_models(Profile.development)
    if dev_missing:
        results.append(CheckResult(FAIL, f"development profile is missing {', '.join(dev_missing)}",
                                   "frun models pull"))
    else:
        results.append(CheckResult(OK, "development profile: all models installed"))

    prod_missing = missing_models(Profile.production)
    if not prod_missing:
        results.append(CheckResult(OK, "production profile: all models installed"))
    else:
        results.append(CheckResult(
            WARN if _cuda_device_count() else INFO,
            f"production profile is missing {', '.join(prod_missing)}",
            "frun models pull --config production",
        ))
    return results


# ---- Audio ---------------------------------------------------------------------

def check_audio() -> List[CheckResult]:
    try:
        with quiet_native_output():
            import sounddevice as sd
    except ImportError:
        return [CheckResult(WARN, "Audio extra not installed: `frun talk` won't work",
                            "pip install 'fusion-runtime[talk]'")]
    except OSError as e:
        hint = "sudo apt install libportaudio2" if sys.platform.startswith("linux") else "install PortAudio"
        return [CheckResult(WARN, f"sounddevice can't load PortAudio: {e}", hint)]

    results = [CheckResult(OK, "sounddevice and PortAudio load")]
    try:
        mic = sd.query_devices(kind="input")
        results.append(CheckResult(OK, f"Microphone: {mic['name']}"))
    except Exception:
        results.append(CheckResult(WARN, "No microphone found", "Plug in a mic or headset for `frun talk`"))
    try:
        speaker = sd.query_devices(kind="output")
        results.append(CheckResult(OK, f"Speaker: {speaker['name']}"))
    except Exception:
        results.append(CheckResult(WARN, "No speaker found", "Plug in speakers or headphones for `frun talk`"))
    if sys.platform == "darwin":
        results.append(CheckResult(
            INFO, "macOS microphone permission can't be checked without recording",
            "If the mic bar in `frun talk` stays flat: System Settings → Privacy & Security → Microphone",
        ))
    return results


# ---- running -------------------------------------------------------------------

SECTIONS: List[tuple] = [
    ("System", [check_python, check_machine, check_disk, check_port]),
    ("Libraries", [check_torch_pair, check_silero_loads, check_engines]),
    ("Acceleration", [check_acceleration]),
    ("Models", [check_models]),
    ("Audio (for frun talk)", [check_audio]),
]


def _safe(check: Check) -> List[CheckResult]:
    try:
        result = check()
    except Exception as e:
        name = getattr(check, "__name__", "check").removeprefix("check_").replace("_", " ")
        return [CheckResult(FAIL, f"The {name} check crashed: {type(e).__name__}: {e}")]
    return result if isinstance(result, list) else [result]


def run_checks(sections=None) -> List[tuple]:
    """[(section title, [CheckResult, ...]), ...]"""
    return [(title, [r for check in checks for r in _safe(check)]) for title, checks in (sections or SECTIONS)]
