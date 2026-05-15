# mm-cost-aware-scheduler

A minimal experiment pipeline for multimodal request scheduling. Turns normalized image/text examples into replayable serving workloads, sends them through an explicit queue/scheduler/batching pipeline, runs a backend, writes JSONL traces, and computes metrics.

```
raw datasets
  -> dataset assembly
  -> workload generation
  -> request queue
  -> scheduler
  -> batch builder
  -> backend
  -> structured logs
  -> analysis
```

---

## Requirements

- Linux with NVIDIA GPU (A30 or A40 recommended, 24GB+ VRAM)
- NVIDIA driver 580+ (CUDA 13.0) — see [Driver Setup](#driver-setup)
- Python 3.10+

---

## Driver Setup

If `nvidia-smi` is not found, install the driver first:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install nvidia-driver-580-server python3.10-dev -y
sudo reboot
```

Verify after reboot:

```bash
nvidia-smi   # should show driver 580, CUDA 13.0
```

---

## Installation

```bash
# Create a virtual environment (use a path with enough disk space)
python3 -m venv ~/venv
source ~/venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio
pip install vllm
pip install pillow datasets transformers matplotlib huggingface_hub
```

Point HuggingFace cache to a large filesystem (the home directory is usually too small):

```bash
export HF_HOME=/proj/<your-project>/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_DISABLE_XET=1
mkdir -p $HF_HOME $HF_DATASETS_CACHE $HF_HUB_CACHE

# Add to ~/.bashrc so it persists across sessions
echo "export HF_HOME=$HF_HOME" >> ~/.bashrc
echo "export HF_DATASETS_CACHE=$HF_HOME/datasets" >> ~/.bashrc
echo "export HF_HUB_CACHE=$HF_HOME/hub" >> ~/.bashrc
echo "export HF_HUB_DISABLE_XET=1" >> ~/.bashrc
```

Authenticate with HuggingFace (required to download Qwen2-VL):

```bash
hf auth login
```

---

## Running the Pipeline

### Step 1 — Smoke test with mock backend (no model needed)

Good first check that the pipeline works before downloading anything:

```bash
python scripts/run_workload.py \
  --backend mock \
  --workload workloads/sample_pipeline_input.jsonl \
  --log logs/mock_smoke.jsonl --reset-log

python scripts/analyze_logs.py --log logs/mock_smoke.jsonl
```

### Step 2 — Assemble real datasets

Downloads COCO and TextVQA from HuggingFace and saves images locally. Uses streaming mode by default so it does not download full parquet files:

```bash
python scripts/assemble_datasets.py \
  --include coco,textvqa \
  --limit-per-dataset 150 \
  --split validation \
  --image-dir /path/to/images \
  --output data/normalized/assembled.jsonl
```

Verify it worked:

```bash
wc -l data/normalized/assembled.jsonl
head -1 data/normalized/assembled.jsonl | python -c "import json,sys; r=json.load(sys.stdin); print(r['dataset'], r['image_path'])"
```

### Step 3 — Create a workload

```bash
python scripts/create_workload.py \
  --input data/normalized/assembled.jsonl \
  --output workloads/stress.jsonl \
  --arrival-process poisson \
  --rate 8.0 \
  --shuffle \
  --limit 300
```

### Step 4 — Run with vLLM backend

Downloads Qwen2-VL-2B-Instruct (~4GB) on first run. Subsequent runs use the cache:

```bash
python scripts/run_workload.py \
  --backend vllm \
  --workload workloads/stress.jsonl \
  --log logs/vllm_run.jsonl \
  --max-batch-size 8 \
  --reset-log
```

### Step 5 — Compare schedulers

```bash
for SCHED in fifo length-only gmax; do
  echo "Running scheduler: $SCHED"
  python scripts/run_workload.py \
    --backend vllm \
    --workload workloads/stress.jsonl \
    --log logs/stress_${SCHED}.jsonl \
    --max-batch-size 8 \
    --scheduler $SCHED \
    --reset-log
done
```

### Step 6 — Analyze and visualize

```bash
# Print metrics for each scheduler
for SCHED in fifo length-only gmax; do
  echo "=== $SCHED ==="
  python scripts/analyze_logs.py --log logs/stress_${SCHED}.jsonl
done

# Generate plots
for SCHED in fifo length-only gmax; do
  python scripts/visualize_logs.py \
    --log logs/stress_${SCHED}.jsonl \
    --output-dir plots/${SCHED}/
done
```

---

## Project Structure

```
src/mmserve_skeleton/
  models.py          Shared dataclasses: MMRequest, Batch, BackendResult, timings, features
  pipeline.py        End-to-end orchestration: submit, schedule, execute, log
  preprocessing.py   Cheap metadata extraction before queueing (image entropy, edge density, text length)
  analyzer.py        Output length prediction and prefill cost estimation
  queue.py           Explicit waiting queue with queue_enter_time tracking
  scheduler.py       Scheduling policies: FIFO, LengthOnly, GMAX
  batching.py        Converts selected requests into a Batch
  backend.py         MockBackend (fast fake) and VLLMBackend (Qwen2-VL via vLLM)
  logging.py         JSONL log writer for completed requests
  __init__.py        Public package entry point

scripts/
  assemble_datasets.py   Normalize COCO / TextVQA / MMMU into one local JSONL schema
  create_workload.py     Add request IDs and arrival_time values for replay
  run_workload.py        Replay a workload through the serving pipeline
  analyze_logs.py        Print latency, TTFT, queue wait, throughput metrics
  visualize_logs.py      Generate plots from request logs
```

---

## Schedulers

| Scheduler | Flag | Description |
|---|---|---|
| FIFO | `--scheduler fifo` | Oldest requests first |
| Length-only | `--scheduler length-only` | Shortest text prompts first |
| GMAX | `--scheduler gmax` | Cost-homogeneous sliding window with aging protection |

---

## Data Formats

Normalized dataset row (output of `assemble_datasets.py`):

```json
{
  "id": "coco-val-0",
  "dataset": "coco",
  "source": "validation",
  "prompt": "<|vision_start|><|image_pad|><|vision_end|>\nDescribe the image in detail.",
  "image_path": "data/images/coco/validation_0.jpg",
  "answer": null,
  "category": "captioning",
  "metadata": {}
}
```

Workload row (output of `create_workload.py`):

```json
{
  "request_id": "coco-val-0",
  "dataset": "coco",
  "source": "validation",
  "prompt": "<|vision_start|><|image_pad|><|vision_end|>\nDescribe the image in detail.",
  "image_path": "data/images/coco/validation_0.jpg",
  "arrival_time": 0.125,
  "metadata": {}
}
```

Only `prompt` is required for replay. `arrival_time` can be relative seconds from replay start or an absolute Unix timestamp.

---

## Component Responsibilities

**models.py** is the source of truth for field names. All scripts pass data using fields defined on `MMRequest`: `request_id`, `arrival_time`, `prompt`, `image_path`, `dataset`, `source`, `metadata`.

**pipeline.py** owns the serving flow. It creates an `MMRequest`, extracts metadata, enqueues it, asks the scheduler what to run, builds a batch, calls the backend, attaches timing fields, and writes logs.

**scheduler.py** provides pluggable scheduling policies. Policies inspect waiting requests and choose which to dispatch without changing queue storage or backend execution.

**backend.py** is the model boundary. The mock backend makes local testing fast. The vLLM backend runs Qwen2-VL-2B-Instruct with streaming so TTFT is captured on the actual first token.

**logging.py** writes one JSON object per completed request. These logs are the basis for latency, TTFT, queue wait, throughput, output length, and per-dataset analysis.
