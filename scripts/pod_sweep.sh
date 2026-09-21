#!/usr/bin/env bash
# One GPU session, unattended: prove the runtime uses the card, then measure
# every model swap worth knowing about. Fifteen minutes instead of an hour.
#
# Run it after the pod is installed (see docs, "Deploying"):
#
#     bash scripts/pod_sweep.sh                 # everything
#     bash scripts/pod_sweep.sh --skip-pull     # models already downloaded
#
# It does four things, in order: proves the GPU is actually being used and stops if
# it isn't, downloads the models, measures Whisper tiny/small/medium and a 7B from
# another model family, then starts a server and measures one, two and four callers
# at once. Downloads about 9 GB the first time. Results land in sweep-results.txt.
set -euo pipefail

cd "$(dirname "$0")/.."
export FUSION_MODEL_DIR="${FUSION_MODEL_DIR:-/workspace/models}"
# torch ships its own CUDA libraries; the image's toolkit shadows them and the
# symbols stop matching. These must come first or VAD fails three layers away.
export LD_LIBRARY_PATH="$(python -c "import site,glob,os; p=site.getsitepackages()[0]; print(':'.join(sorted(glob.glob(os.path.join(p,'nvidia','*','lib')))))"):${LD_LIBRARY_PATH:-}"

RESULTS="sweep-results.txt"
SECOND_LLM="${SECOND_LLM:-hf:bartowski/Mistral-7B-Instruct-v0.3-GGUF/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf}"

# ---- 1. is this actually a GPU run? -------------------------------------------------
# Every number below is worthless if either check fails, so they fail the script.
echo "== environment =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("torch cannot see the GPU — check LD_LIBRARY_PATH")
print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")

import onnxruntime
if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
    sys.exit("onnxruntime has no CUDA provider — TTS would run on the CPU and the "
             "numbers would be meaningless. Install onnxruntime-gpu<1.30 from the CUDA 12 feed.")
print(f"onnxruntime {onnxruntime.__version__} with CUDA")

import ctranslate2
if ctranslate2.get_cuda_device_count() < 1:
    sys.exit("ctranslate2 has no CUDA device — speech-to-text would run on the CPU")
print(f"ctranslate2 {ctranslate2.__version__} with {ctranslate2.get_cuda_device_count()} device(s)")
PY

# ---- 2. models -----------------------------------------------------------------------
if [ "${1:-}" != "--skip-pull" ]; then
  echo
  echo "== downloading =="
  frun models pull --config production
  frun models pull hf:Systran/faster-whisper-small
  frun models pull hf:Systran/faster-whisper-medium
  frun models pull "$SECOND_LLM"
fi

# ---- 3. the sweep --------------------------------------------------------------------
: > "$RESULTS"
run() {  # run <label> <bench args...>
  local label="$1"; shift
  echo
  echo "== $label =="
  { echo; echo "--- $label"; } >> "$RESULTS"
  python3 scripts/bench_latency.py --profile production --turns 8 "$@" 2>&1 | tee -a "$RESULTS"
}

# Speech-to-text is the accuracy weak link: tiny.en is fast and mishears names.
# This says what small and medium cost in milliseconds for that accuracy.
run "stt: tiny.en (current default)"
run "stt: small"  --stt hf:Systran/faster-whisper-small
run "stt: medium" --stt hf:Systran/faster-whisper-medium

# A model from another family, to check the claim that the chat template comes
# out of the GGUF and no model-specific code exists anywhere.
run "llm: Mistral 7B (different family, same size)" --llm "$SECOND_LLM"

# ---- 4. several callers at once -------------------------------------------------------
# The one thing the banner says isn't measured. This needs a running server, because
# concurrency is a property of sessions, not of the pipeline in isolation.
echo
echo "== concurrency: starting a server =="
# The production profile, like every measurement above. Plain `frun up` takes the
# development one, which wants qwen2.5-0.5b — a model this script never pulls, so
# the server refused to start and the whole phase was lost.
FUSION_CONFIG=production frun up --host 127.0.0.1 > frun-sweep.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

for i in $(seq 1 90); do
  if curl -fs http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "server ready after ${i}s"
    break
  fi
  sleep 1
done
if ! curl -fs http://127.0.0.1:8000/health > /dev/null 2>&1; then
  echo "the server never became healthy. Its last words:" | tee -a "$RESULTS"
  tail -20 frun-sweep.log | tee -a "$RESULTS"
  exit 1
fi

{ echo; echo "--- concurrency: in-process llama.cpp"; } >> "$RESULTS"
python3 scripts/concurrency_check.py --callers 1,2,4 --turns 3 \
  --label "in-process llama.cpp on this GPU" 2>&1 | tee -a "$RESULTS"

# The comparison that makes the claim real. vLLM and llama-server both speak the
# OpenAI API, which is what the openai_http runtime uses, so pointing the agent at
# one changes nothing but the URL. Start it yourself and set LLM_URL:
#
#   python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --port 8080 &
#   LLM_URL=http://127.0.0.1:8080/v1 bash scripts/pod_sweep.sh --skip-pull
if [ -n "${LLM_URL:-}" ]; then
  echo
  echo "== concurrency: against $LLM_URL =="
  kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true
  FUSION_CONFIG=production FUSION_LLM_URL="$LLM_URL" frun up --host 127.0.0.1 > frun-sweep-remote.log 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 90); do
    curl -fs http://127.0.0.1:8000/health > /dev/null 2>&1 && break
    sleep 1
  done
  { echo; echo "--- concurrency: LLM at $LLM_URL"; } >> "$RESULTS"
  python3 scripts/concurrency_check.py --callers 1,2,4 --turns 3 \
    --label "LLM served at $LLM_URL" 2>&1 | tee -a "$RESULTS"
else
  echo
  echo "Skipping the served-LLM comparison: set LLM_URL to an OpenAI-compatible endpoint"
  echo "(vLLM or llama-server) and run again with --skip-pull to get the other half."
fi

kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true

echo
echo "== done =="
echo "Results in $RESULTS. Transcribe the medians into 11-gpu-pod.md before stopping the pod —"
echo "the container disk goes with it."
