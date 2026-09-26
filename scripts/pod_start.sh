#!/usr/bin/env bash
# Start (or restart) fusion + vLLM on a GPU pod, from whatever state the pod is in.
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
#   3. starts vLLM unless it is already answering,
#   4. starts fusion with a key, the pod's public origin, and limits a
#      load test won't trip over.
#
# Change anything below with an environment variable, e.g.
#   AGENT=examples/my_agent.py MAX_SESSIONS=32 bash scripts/pod_start.sh
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO="${REPO:-$WORKSPACE/fusion-runtime}"
AGENT="${AGENT:-examples/tools_agent.py}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
TOOL_PARSER="${TOOL_PARSER:-hermes}"            # vLLM's parser for the model family (hermes = Qwen 2.5)
VLLM_PORT="${VLLM_PORT:-8002}"                  # not 8001: Runpod's nginx has it
PORT="${PORT:-8888}"                            # a port the pod exposes over its HTTPS proxy
VLLM_ENV="${VLLM_ENV:-$WORKSPACE/vllm-env}"
MAX_SESSIONS="${MAX_SESSIONS:-20}"              # the server's default is 4, too few for a load test
ORT_CUDA_INDEX="https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"

export FUSION_MODEL_DIR="${FUSION_MODEL_DIR:-$WORKSPACE/models}"
export HF_HOME="${HF_HOME:-$WORKSPACE/hf}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$WORKSPACE/vllm-cache}"  # compiled kernels survive restarts
export VLLM_USE_FLASHINFER_SAMPLER=0             # its first-use compile needs tools the pod image lacks
export TMPDIR="$WORKSPACE/tmp"                   # big installs unpack on the volume, not the small disk
mkdir -p "$TMPDIR"

step() { printf '\n==> %s\n' "$*"; }
fail() { printf '\nFAILED: %s\n' "$*" >&2; exit 1; }

# The NVIDIA libraries pip installs (cuDNN, cuBLAS) are only found if they are on this path.
cuda_libs() {
  python - <<'EOF'
import glob, os, site
print(":".join(sorted(glob.glob(os.path.join(site.getsitepackages()[0], "nvidia", "*", "lib")))))
EOF
}

# ---- 1. fusion ------------------------------------------------------------------------------
step "fusion"
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

# ---- 3. vLLM ----------------------------------------------------------------------------------
step "vLLM ($LLM_MODEL on port $VLLM_PORT)"
if curl -fs "localhost:$VLLM_PORT/health" >/dev/null 2>&1; then
  echo "already running"
else
  if [[ ! -x "$VLLM_ENV/bin/vllm" ]]; then
    echo "not installed; installing into $VLLM_ENV (several GB, a few minutes)"
    python -m venv "$VLLM_ENV"
    "$VLLM_ENV/bin/pip" install -q --no-cache-dir vllm
  fi
  # Its own venv, activated so the tools it runs are on PATH; its torch stays out of fusion's.
  nohup bash -c "source '$VLLM_ENV/bin/activate' && exec vllm serve '$LLM_MODEL' --port $VLLM_PORT \
      --enable-auto-tool-choice --tool-call-parser $TOOL_PARSER \
      --gpu-memory-utilization 0.6 --max-model-len 4096" > "$WORKSPACE/vllm.log" 2>&1 &
  vllm_pid=$!
  echo "starting (log: $WORKSPACE/vllm.log)"
  for _ in $(seq 120); do                       # up to 10 minutes: a first start downloads the model
    curl -fs "localhost:$VLLM_PORT/health" >/dev/null 2>&1 && break
    kill -0 "$vllm_pid" 2>/dev/null || { tail -20 "$WORKSPACE/vllm.log"; fail "vLLM exited while starting (above)"; }
    sleep 5
  done
  curl -fs "localhost:$VLLM_PORT/health" >/dev/null 2>&1 || fail "vLLM didn't answer in 10 minutes; see $WORKSPACE/vllm.log"
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
FUSION_LLM_URL="http://localhost:$VLLM_PORT/v1" \
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
echo "vLLM:    http://localhost:$VLLM_PORT   (log: $WORKSPACE/vllm.log)"
if [[ -n "$origin" ]]; then
  echo "browser: run 'frun token --url http://127.0.0.1:$PORT --key \$(cat $WORKSPACE/key)'"
  echo "         and open $origin/?token=<the token it prints>"
fi
echo "load test:"
echo "  python scripts/concurrency_check.py --url ws://127.0.0.1:$PORT --key \$(cat $WORKSPACE/key) \\"
echo "      --audio tests/fixtures/order_1042.wav --callers 1,4,8,12,16 --turns 3"
