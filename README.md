# Ocelot

Ocelot is a small end-to-end stack for **preference-style LLM data**, **supervised / preference fine-tuning**, and **comparative evaluation**: collect examples from real browser sessions, train adapters on that data, and rank multiple model outputs with a stronger judge model.

The three main areas of the repo are independent enough to run on their own, but they share a natural flow: **data → training → evaluation**. All three live under [`src/`](src/).

---

## [`src/data/`](src/data/)

Hosts the **collection pipeline** (automation against Brave, a local HTTP API, JSON artifacts on disk) and **postprocessing** that turns raw API output into fixed train/validation/test splits.

The FastAPI “data API” that sits in the middle of collection lives under [`src/data/api/`](src/data/api/); it is documented separately from the top-level data README.

**Where to read next:** [`src/data/README.md`](src/data/README.md) for the full orchestration story, prerequisites, and layout; [`src/data/api/README.md`](src/data/api/README.md) for the OpenAI-compatible gateway, `config/vllm_config.yaml`, and how requests become stored examples.

---

## [`src/evaluation/`](src/evaluation/)

Hosts **LLM-as-judge** tooling: compare three candidate answers to the same prompt using LiteLLM-backed judges (e.g. Bedrock or a local OpenAI-compatible server), with a CLI, Docker option, and a small library API.

**Where to read next:** [`src/evaluation/README.md`](src/evaluation/README.md) for install, input format, providers, and programmatic use.

---

## [`src/training/`](src/training/)

Hosts a **modular training entrypoint** for instruction / preference fine-tuning with LoRA (multiple trainer types, shared config and data plumbing, extension points for new methods).

**Where to read next:** [`src/training/README.md`](src/training/README.md) for how to invoke training, expected dataset shape, and how new methods plug into the registry.

---

## Tests

From the repository root:

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest
```

[`pyproject.toml`](pyproject.toml) sets `pythonpath = ["src"]` and `testpaths = ["test"]`. Tests mirror packages under [`src/`](src/): [`test/data/`](test/data/), [`test/evaluation/`](test/evaluation/), [`test/training/`](test/training/).

### Default run (no GPU)

`pytest` runs everything that is not skipped. That includes data helpers, evaluation/judge logic, training registry checks, and fast training tests such as Parquet roundtrips (`test/training/test_prepare_data.py::test_chunk_to_arrow_roundtrip_through_parquet`, needs `pyarrow` + `numpy`).

Training integration tests are marked `integration` and **skip** unless you opt in (below); they do not fail a normal CI run.

### By area

| Path | What it covers |
|------|----------------|
| [`test/data/`](test/data/) | Brave path resolution, postprocessing merges |
| [`test/evaluation/`](test/evaluation/) | Judge config, prompts, LiteLLM judge wiring, CLI smoke |
| [`test/training/`](test/training/) | Method registry, `prepare_data` / Parquet helpers, optional GPU E2E |

Filter examples:

```bash
python3 -m pytest test/data/ -q
python3 -m pytest test/evaluation/ -q
python3 -m pytest test/training/ -q
python3 -m pytest -m integration   # only marked tests (still skip without OCELOT_TRAINING_E2E)
```

### GPU / Hub end-to-end training (optional)

Install training dependencies for your machine: [`src/training/requirements-macos.txt`](src/training/requirements-macos.txt) on macOS, [`src/training/requirements-linux-cuda.txt`](src/training/requirements-linux-cuda.txt) on Linux with NVIDIA CUDA ([`src/training/requirements.txt`](src/training/requirements.txt) documents the split). On **Linux + CUDA**, optional **`flash-attn`** is installed separately with **`pip install flash-attn --no-build-isolation`** ([`src/training/README.md`](src/training/README.md)). You need **CUDA or Apple MPS** for these.

Set **`OCELOT_TRAINING_E2E=1`** to enable the integration tests. Useful environment variables:

- **`OCELOT_E2E_MODEL_NAME`** — Hugging Face model id (default in tests is a small Qwen3-VL instruct checkpoint).
- **`OCELOT_LOAD_IN_4BIT`** — on Apple Silicon the tests force `0` (no bitsandbytes); on CUDA you can keep 4-bit if `bitsandbytes` is installed.

**[`test/training/test_one_batch_integration.py`](test/training/test_one_batch_integration.py)** — full trainer smoke: one **SFT** stage, load the saved LoRA checkpoint, then one **IPO** (CPO) stage on the same tiny JSON dataset.

**[`test/training/test_prepare_data.py`](test/training/test_prepare_data.py)** — `prepare_data.main()` writes Parquet; **`test_prepare_data_parquet_sft_then_ipo_collator_forward_e2e`** loads it via `PREPARED_DATA_DIR`, runs one **SFT** batch through the model, then an **IPO-style** batch (`TokenizedDPOCollator`) with chosen and rejected forwards.

Example:

```bash
export OCELOT_TRAINING_E2E=1
python3 -m pytest test/training/test_one_batch_integration.py -v
python3 -m pytest test/training/test_prepare_data.py::test_prepare_data_parquet_sft_then_ipo_collator_forward_e2e -v
```
