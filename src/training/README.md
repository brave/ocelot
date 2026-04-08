## Training (modular)

This folder contains:

- **A modular training framework**: a training framework for LLMs with a LoRA, by providing small CLI.

### Install (pick one)

| Platform | File |
|----------|------|
| **macOS** (CPU / MPS) | `pip install -r src/training/requirements-macos.txt` |
| **Linux + NVIDIA GPU** (CUDA 12) | `pip install -r src/training/requirements-linux-cuda.txt` |

`requirements.txt` in this folder includes the Linux+CUDA set by default (for Docker / GPU servers). On Mac, use **`requirements-macos.txt`** so pip does not pull `nvidia-*-cu12` or `triton` wheels. Prefer **Python 3.11 or 3.12** if anything still lags on 3.13.

### Optional: FlashAttention (`flash-attn`)

The PyPI package name is **`flash-attn`**. It is **not** listed in the requirements files: install it only on **Linux with NVIDIA CUDA** when you want FlashAttention2 (e.g. default `OCELOT_ATTN_IMPLEMENTATION=flash_attention_2` in `modeling/factory.py`). It does **not** apply to macOS / MPS the same way.

```bash
python3 -m pip install flash-attn --no-build-isolation
```

### Integration test (one SFT step) on macOS

From the **repository root**, with **Apple Silicon** (MPS). The test downloads a small VL model from Hugging Face; ensure you have enough unified memory.

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pip install -r src/training/requirements-macos.txt
OCELOT_TRAINING_E2E=1 python3 -m pytest test/training/test_one_batch_integration.py -v
```

Optional: `OCELOT_E2E_MODEL_NAME=...` overrides the default checkpoint in the test file (`Qwen/Qwen3-VL-2B-Instruct` at time of writing). On MPS, 4-bit quantization is turned off automatically.

**Intel Macs** without CUDA or MPS cannot run this integration test (it will skip). Use **Linux + NVIDIA GPU** for that, or run the rest of the suite with `python3 -m pytest` (non-integration tests only need `requirements-dev.txt`).

### Quick start

Run from **repository root** (plain python). `run.py` prepends this directory to `sys.path` so `core`, `methods`, and `data` resolve.

Defaults (see `core/config.py`): **`--model-name`** defaults to `Qwen/Qwen3-VL-4B-Instruct` (or env `MODEL_NAME`); **`--output-dir`** defaults to `./runs/qwen-training` (or `OUTPUT_DIR`). **`--data-path`** is required unless **`PREPARED_DATA_DIR`** / **`--prepared-data-dir`** is set (two-stage flow below).

```bash
python src/training/run.py --trainer sft --epochs 1 --data-path /path/to/data.json --output-dir ./runs/qwen-sft
```

IPO uses TRL **`CPOTrainer`** with **`loss_type="ipo"`**, optionally with an SFT warmup (`--sft-warmup-epochs`, env default `SFT_EPOCHS` is `0`):

```bash
python src/training/run.py --trainer ipo --epochs 1 --sft-warmup-epochs 1 --data-path /path/to/data.json --output-dir ./runs/qwen-ipo
```

DPO is **`--trainer dpo`** (same JSON schema; uses **`--dpo-beta`**).

Run with **Accelerate** (repo ships `accelerate_conf.json` with DeepSpeed; `num_processes` is `1` until you edit it):

```bash
accelerate launch --config_file src/training/accelerate_conf.json src/training/run.py --trainer sft --epochs 1 --data-path /path/to/data.json --output-dir ./runs/qwen-sft
```

### Shorter commands (environment variables)

Most flags have env equivalents (`DATA_PATH`, `OUTPUT_DIR`, `MODEL_NAME`, `PREPARED_DATA_DIR`, `SFT_LEARNING_RATE`, `PREF_LEARNING_RATE`, `IPO_BETA`, `DPO_BETA`, `LORA_RANK`, etc.). Example:

```bash
export DATA_PATH=/path/to/data.json
export OUTPUT_DIR=./runs/qwen-sft
python src/training/run.py --trainer sft --epochs 1
```

### Two-stage training (low RAM)

1. Tokenize once: **`prepare_data.py`** writes **`train.parquet`** and **`validation.parquet`** (and optional `train-*.parquet` shards) under `--output-dir`.
2. Train from Parquet: pass **`--prepared-data-dir`** (or set **`PREPARED_DATA_DIR`**); **`--data-path`** is not required.

```bash
python src/training/prepare_data.py --data-path /path/to/data.json --output-dir /path/to/prepared
python src/training/run.py --trainer sft --epochs 1 --prepared-data-dir /path/to/prepared --output-dir ./runs/qwen-sft
```

Learning rate overrides:

```bash
python src/training/run.py --trainer sft --epochs 1 --sft-learning-rate 3e-5 ...
python src/training/run.py --trainer ipo --epochs 1 --pref-learning-rate 5e-6 ...
```

Beta overrides:

```bash
python src/training/run.py --trainer ipo --epochs 1 --ipo-beta 0.1 ...
python src/training/run.py --trainer dpo --epochs 1 --dpo-beta 0.1 ...
```

### Data format (expected)

The loader in **`data/pipeline.py`** expects:

- A JSON file with **two top-level fields**: **`train`** and **`validation`** (exact keys; not `val`).
- Each example should contain:
  - **`prompt`**: list of chat messages (Qwen-style, multimodal parts allowed)
  - **`chosen`**: assistant response (string)
  - **`rejected`**: assistant response (string)

Even for **SFT-only** runs, **`rejected`** must be present: the pipeline tokenizes both branches into a shared column set (IPO/DPO need both; SFT uses the chosen side).

### Extending with a new training method

Add a new method under `src/training/methods/` implementing `TrainingMethod`, then register it in `src/training/methods/registry.py`.

### Layout

- **`run.py`**: CLI entry point (`RunConfig.from_argv` → selected method)
- **`prepare_data.py`**: optional Stage 1 JSON → Parquet
- **`core/config.py`**: dataclass config + argparse + env defaults
- **`core/hf_args.py`**: shared HF `TrainingArguments` builders
- **`data/pipeline.py`**: dataset loading, preprocessing, filtering, one-pass tokenization
- **`data/collators.py`**: stack-only collators for pre-tokenized fixed-length data
- **`modeling/factory.py`**: model + processor loading, LoRA wiring, optional QAT hooks
- **`methods/`**: training methods (`sft`, `ipo`, ...)

