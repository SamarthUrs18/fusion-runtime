#!/usr/bin/env bash
# Start (or restart) fusion + vLLM or SGLang on a GPU pod, from whatever state the pod is in.
#
#   bash /workspace/fusion-runtime/scripts/pod_start.sh
#
# Safe to run again at any time. It exists because a pod has two disks: the
# volume (/workspace) keeps vLLM, the models and the key, but the container
# disk is wiped on every stop, restart or resize, and fusion's pip install
# lives there. So each run:
#
#   1. installs fusion if the container lost it,
#   2. makes sure Kokoro will run on the GPU (onnxruntime ships a CPU and a GPU
#      build under one import name; with both installed the CPU one can win,
#      silently, and every TTS number measured after that is wrong),
#   3. starts the LLM server (vLLM, or SGLang with ENGINE=sglang) unless it is
#      already answering,
#   4. starts fusion with a key, the pod's public origin, and limits a
#      load test won't trip over.
#
# Change anything below with an environment variable, e.g.
#   AGENT=examples/my_agent.py MAX_SESSIONS=32 bash scripts/pod_start.sh
#   ENGINE=sglang bash scripts/pod_start.sh
#   WORKSPACE=/root/work bash scripts/pod_start.sh   # when /workspace can't hold code (see step 0)
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO="${REPO:-$WORKSPACE/fusion-runtime}"
AGENT="${AGENT:-examples/tools_agent.py}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
ENGINE="${ENGINE:-vllm}"                        # vllm | sglang
PORT="${PORT:-8888}"                            # a port the pod exposes over its HTTPS proxy
VLLM_PACKAGE="${VLLM_PACKAGE:-vllm}"            # pip's vLLM is built for CUDA 13; see step 0
case "$ENGINE" in
  vllm)   LLM_PORT="${LLM_PORT:-8002}"          # not 8001: Runpod's nginx has it
          TOOL_PARSER="${TOOL_PARSER:-hermes}"  # the parser for the model family (hermes = Qwen 2.5)
          ENGINE_ENV="${ENGINE_ENV:-$WORKSPACE/vllm-env}"
          ENGINE_MATCH="vllm serve" ;;
  sglang) LLM_PORT="${LLM_PORT:-30000}"
          TOOL_PARSER="${TOOL_PARSER:-qwen25}"
          ENGINE_ENV="${ENGINE_ENV:-$WORKSPACE/sglang-env}"
          ENGINE_MATCH="sglang.launch_server" ;;
  *)      printf 'ENGINE must be vllm or sglang, not %s\n' "$ENGINE" >&2; exit 1 ;;
esac
MAX_SESSIONS="${MAX_SESSIONS:-20}"              # the server's default is 4, too few for a load test
ORT_CUDA_INDEX="https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"

export FUSION_MODEL_DIR="${FUSION_MODEL_DIR:-$WORKSPACE/models}"
# Always ours: Runpod's templates set HF_HOME to /workspace, which may be a volume that can't
# hold Hugging Face's cache (it uses symlinks), and the model then fails after a 5 GB download.
export HF_HOME="$WORKSPACE/hf"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$WORKSPACE/vllm-cache}"  # compiled kernels survive restarts
export VLLM_USE_FLASHINFER_SAMPLER=0             # its first-use compile needs tools the pod image lacks
export TMPDIR="$WORKSPACE/tmp"                   # big installs unpack on the volume, not the small disk

step() { printf '\n==> %s\n' "$*"; }
fail() { printf '\nFAILED: %s\n' "$*" >&2; exit 1; }

# ---- 0. the machine ---------------------------------------------------------------------------
# Two things that otherwise fail late and cryptically, after minutes of installing.
step "machine"
command -v nvidia-smi >/dev/null 2>&1 || fail "no nvidia-smi: this machine has no NVIDIA GPU driver"
driver_cuda="$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | grep -o '[0-9.]*$' || true)"
[[ -n "$driver_cuda" ]] || fail "nvidia-smi didn't report a CUDA version; is the GPU visible? (nvidia-smi)"
cuda_major="${driver_cuda%%.*}"; cuda_minor="${driver_cuda#*.}"; cuda_minor="${cuda_minor%%.*}"
echo "GPU driver supports CUDA $driver_cuda"
# torch 2.8+ is built for CUDA 12.8; an older driver can't run it ("CUDA unknown error").
if (( cuda_major < 12 || (cuda_major == 12 && cuda_minor < 8) )); then
  fail "this machine's driver only supports CUDA $driver_cuda; torch needs 12.8 or newer.
  Pick another machine (Runpod: Filter -> CUDA version 12.8+)."
fi
if [[ "$ENGINE" == vllm && "$VLLM_PACKAGE" == vllm && "$cuda_major" -lt 13 && ! -x "$ENGINE_ENV/bin/vllm" ]]; then
  fail "pip's current vLLM is built for CUDA 13, and this driver supports $driver_cuda.
  Set VLLM_PACKAGE to a vLLM release built for CUDA 12 (see vLLM's install docs), or use ENGINE=sglang."
fi
# Some volumes (Runpod's global volume) refuse chmod, which git, pip and venvs all need.
mkdir -p "$TMPDIR"
probe="$WORKSPACE/.pod_start_probe"
if ! { touch "$probe" && chmod +x "$probe"; } 2>/dev/null; then
  rm -f "$probe" 2>/dev/null || true
  fail "$WORKSPACE doesn't allow file permissions (chmod), so code and installs can't live there.
  Use the container disk instead: WORKSPACE=/root/work (clone the repo there first)."
fi
rm -f "$probe"
echo "$WORKSPACE takes code and installs"

# The NVIDIA libraries pip installs (cuDNN, cuBLAS) are only found if they are on this path.
cuda_libs() {
  python - <<'EOF'
import glob, os, site
print(":".join(sorted(glob.glob(os.path.join(site.getsitepackages()[0], "nvidia", "*", "lib")))))
EOF
}

# ---- 1. fusion ------------------------------------------------------------------------------
step "fusion"
[[ -d "$REPO" ]] || fail "no fusion-runtime checkout at $REPO. Clone it first:
  git clone https://github.com/SamarthUrs18/fusion-runtime.git $REPO"
cd "$REPO"
if command -v frun >/dev/null 2>&1; then
  echo "installed ($(frun version 2>/dev/null | head -1))"
else
  echo "not installed (the container disk was reset); installing"
  python -m pip install -q --upgrade pip
  python -m pip install -q --no-cache-dir -e .
fi
export LD_LIBRARY_PATH="$(cuda_libs)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ---- 2. Kokoro on the GPU ---------------------------------------------------------------------
step "Kokoro on the GPU"
kokoro_device() {
  # Open Kokoro's own model with the GPU provider and ask onnxruntime what it actually used.
  # Checking that a package is installed isn't enough: the CPU build answers the same import.
  python - <<'EOF'
import glob, os, sys
import onnxruntime as ort
if "CUDAExecutionProvider" not in ort.get_available_providers():
    print(f"cpu-build {ort.__version__} at {os.path.dirname(ort.__file__)}"); sys.exit()
models = glob.glob(os.path.join(os.environ["FUSION_MODEL_DIR"], "tts", "**", "*.onnx"), recursive=True)
if not models:
    print("no-model"); sys.exit()
ort.set_default_logger_severity(3)
session = ort.InferenceSession(models[0], providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print("gpu" if session.get_providers()[0] == "CUDAExecutionProvider" else "cuda-failed")
EOF
}
device="$(kokoro_device)"
if [[ "$device" == cpu-build* ]]; then
  echo "onnxruntime is the CPU build ($device); replacing it with the GPU build"
  python -m pip uninstall -y -q onnxruntime onnxruntime-gpu || true
  python -m pip install -q --no-cache-dir --force-reinstall --no-deps onnxruntime-gpu --index-url "$ORT_CUDA_INDEX"
  device="$(kokoro_device)"
fi
case "$device" in
  gpu)         echo "Kokoro opens on CUDA" ;;
  no-model)    echo "Kokoro isn't downloaded yet; the GPU build is in place and fusion checks again at startup" ;;
  cuda-failed) fail "onnxruntime has the GPU build but couldn't start CUDA (usually a cuDNN/CUDA mismatch).
  Check: python -c \"import onnxruntime as o; print(o.__version__)\"  and  nvidia-smi" ;;
  *)           fail "onnxruntime is still the CPU build after reinstalling ($device)" ;;
esac

# ---- 3. the LLM server ------------------------------------------------------------------------
step "$ENGINE ($LLM_MODEL on port $LLM_PORT)"
# Only one engine fits next to speech on a 24 GB card: stop the other one if it's running.
other_match="vllm serve"; [[ "$ENGINE" == vllm ]] && other_match="sglang.launch_server"
if pkill -f "$other_match" 2>/dev/null; then echo "stopped the other engine to free the GPU"; sleep 5; fi
if curl -fs "localhost:$LLM_PORT/health" >/dev/null 2>&1; then
  echo "already running"
else
  if [[ ! -x "$ENGINE_ENV/bin/python" ]] || ! "$ENGINE_ENV/bin/python" -c "import $ENGINE" 2>/dev/null; then
    echo "not installed; installing into $ENGINE_ENV (several GB, a few minutes)"
    python -m venv "$ENGINE_ENV"
    if [[ "$ENGINE" == vllm ]]; then
      "$ENGINE_ENV/bin/pip" install -q --no-cache-dir "$VLLM_PACKAGE"
    else
      "$ENGINE_ENV/bin/pip" install -q --no-cache-dir "sglang[all]"
    fi
  fi
  # Its own venv, on PATH: both engines compile kernels on first use and call `ninja` from it.
  # Its torch stays out of fusion's.
  if [[ "$ENGINE" == vllm ]]; then
    serve=(vllm serve "$LLM_MODEL" --port "$LLM_PORT" --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER"
           --gpu-memory-utilization 0.6 --max-model-len 4096)
  else
    serve=(python -m sglang.launch_server --model-path "$LLM_MODEL" --host 127.0.0.1 --port "$LLM_PORT"
           --tool-call-parser "$TOOL_PARSER" --mem-fraction-static 0.6 --context-length 4096)
  fi
  PATH="$ENGINE_ENV/bin:$PATH" nohup "${serve[@]}" > "$WORKSPACE/$ENGINE.log" 2>&1 &
  engine_pid=$!
  echo "starting (log: $WORKSPACE/$ENGINE.log); a first start downloads the model and compiles kernels"
  for _ in $(seq 180); do                       # up to 15 minutes: SGLang's first compile is slow
    curl -fs "localhost:$LLM_PORT/health" >/dev/null 2>&1 && break
    kill -0 "$engine_pid" 2>/dev/null || { tail -20 "$WORKSPACE/$ENGINE.log"; fail "$ENGINE exited while starting (above)"; }
    sleep 5
  done
  curl -fs "localhost:$LLM_PORT/health" >/dev/null 2>&1 || fail "$ENGINE didn't answer in 15 minutes; see $WORKSPACE/$ENGINE.log"
  echo "up"
fi

# ---- 4. fusion's server ---------------------------------------------------------------------
step "fusion server ($AGENT on port $PORT)"
if [[ ! -s "$WORKSPACE/key" ]]; then
  frun key new 2>/dev/null | grep -o -m1 'frun_[A-Za-z0-9_-]*' > "$WORKSPACE/key"
  [[ -s "$WORKSPACE/key" ]] || fail "couldn't generate a key (frun key new)"
  echo "new key in $WORKSPACE/key"
fi
frun models pull "$AGENT"

origin="${PUBLIC_URL:-}"
if [[ -z "$origin" && -n "${RUNPOD_POD_ID:-}" ]]; then
  origin="https://$RUNPOD_POD_ID-$PORT.proxy.runpod.net"
fi

pkill -f "frun up" 2>/dev/null && sleep 2 || true
# Runpod's template starts Jupyter on 8888; nothing else of ours uses it.
if (echo > "/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
  if pkill -f jupyter 2>/dev/null; then
    echo "stopped Jupyter, which had port $PORT"; sleep 2
  else
    fail "port $PORT is taken by something other than Jupyter. Stop it, or choose another: PORT=... bash $0"
  fi
fi
FUSION_LLM_URL="http://localhost:$LLM_PORT/v1" \
FUSION_ACCEPTED_KEYS="$(cat "$WORKSPACE/key")" \
FUSION_ALLOWED_ORIGINS="$origin" \
FUSION_MAX_SESSIONS="$MAX_SESSIONS" \
FUSION_CONNECTIONS_PER_MINUTE="${CONNECTIONS_PER_MINUTE:-300}" \
  nohup frun up "$AGENT" --host 0.0.0.0 --port "$PORT" > "$WORKSPACE/frun.log" 2>&1 &
fusion_pid=$!
for _ in $(seq 60); do
  curl -fs "localhost:$PORT/health" >/dev/null 2>&1 && break
  kill -0 "$fusion_pid" 2>/dev/null || { tail -20 "$WORKSPACE/frun.log"; fail "fusion exited while starting (above)"; }
  sleep 3
done
curl -fs "localhost:$PORT/health" >/dev/null 2>&1 || fail "fusion didn't answer in 3 minutes; see $WORKSPACE/frun.log"
# Only a fall back to the CPU alone counts: onnxruntime also "falls back" from TensorRT to CUDA,
# which is harmless and prints a scary-looking error on every start.
if grep -qF -e "Failed to create CUDAExecutionProvider" -e "Falling back to ['CPUExecutionProvider']" "$WORKSPACE/frun.log"; then
  fail "fusion started, but a model fell back to the CPU; see $WORKSPACE/frun.log"
fi

step "ready"
echo "fusion:  http://localhost:$PORT   (log: $WORKSPACE/frun.log)"
echo "$ENGINE:  http://localhost:$LLM_PORT   (log: $WORKSPACE/$ENGINE.log)"
if [[ -n "$origin" ]]; then
  echo "browser: run 'frun token --url http://127.0.0.1:$PORT --key \$(cat $WORKSPACE/key)'"
  echo "         and open $origin/?token=<the token it prints>"
fi
echo "load test:"
echo "  python scripts/concurrency_check.py --url ws://127.0.0.1:$PORT --key \$(cat $WORKSPACE/key) \\"
echo "      --audio tests/fixtures/order_1042.wav --callers 1,4,8,12,16 --turns 3"
