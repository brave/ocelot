# Ocelot

## About

**A suite of tools to collect web page data, train a model and evaluate it**

Ocelot is a small toolkit that enables the automated collection of webpage data for model training. The primary use of the codebase is to train the Ocelot summarisation model, although it can easily be extended to other specific tasks.

The codebase contains a small playwright script and data API to allow for the collection of web page data. It also contains training code for **supervised and preference fine-tuning** with **mixed text and image** datasets.
Results can be tested using the LLM as a judge evaluation module included as well.

### Data

The data module enables the automated collection of a set of prompts to train a model via Leo in the brave browser. The browser will be pointed to a simple API that recieves the page content, makes requests to LLMs and then stores inputs and responses. This works with both web page text and images.

### Evaluation

Provides an LLM as a judge framework for simultaneously comparing, scoring and ranking, three different responses for a given prompt. This can be used to validate the performance of a trained model.

### Training

Provides a CLI to train an LLM on the collected prompts and responses via supervised fine tuning (SFT) and prefence fine tuning.

The packages under [`src/`](src/) are usable on their own and are meant to compose in a simple pipeline: **data → training → evaluation**.

---

## Module READMEs (full detail)

Setup, CLI usage, configuration, and extension points are documented per module:

| Module | README |
| --- | --- |
| Data collection & postprocessing | [`src/data/README.md`](src/data/README.md) |
| Data API (FastAPI / OpenAI-compatible gateway) | [`src/data/api/README.md`](src/data/api/README.md) |
| Training (LoRA, datasets, registry) | [`src/training/README.md`](src/training/README.md) |
| Evaluation (LLM-as-judge) | [`src/evaluation/README.md`](src/evaluation/README.md) |

---

## Repository layout

The summaries below are intentionally short; use the README links for **further information**.

### [`src/data/`](src/data/)

**Further information:** [`src/data/README.md`](src/data/README.md) (pipeline, prerequisites, layout). The FastAPI “data API” used in the collection flow lives under [`src/data/api/`](src/data/api/) — see [`src/data/api/README.md`](src/data/api/README.md) for the gateway, `config/vllm_config.yaml`, and how requests become stored examples.

### [`src/evaluation/`](src/evaluation/)

**Further information:** [`src/evaluation/README.md`](src/evaluation/README.md) (install, input format, providers, programmatic use).

### [`src/training/`](src/training/)

**Further information:** [`src/training/README.md`](src/training/README.md) (invoking training, expected dataset shape, registering new methods).

---

## Getting started

Both the data collection, and evaluation can be run via docker compose. Note that a built version of the Brave Browser is required to run the data collection, and further instructions on how to do this can be seen in [`brave-core`](https://github.com/brave/brave-core).

It is reccomended to run the training module on a suitable GPU device.

For component-specific setup, follow the **Module READMEs** table above (data, API, training, or evaluation).

---

## Contributing

If you wish to contribute, please raise any issues or PRs directly in this repository and we will endeavour to review them as quickly as possible. 

Please ensure that any PR also has a linked issue explaining the rationale.

## License

This code is made available under the Mozilla Public License 2.0. Please see the [`LICENSE`](LICENSE) for more information.

---

## Tests

From the repository root:
(note that the dependencies from each module will also need to be installed)
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

**[`test/training/test_one_batch_integration.py`](test/training/test_one_batch_integration.py)** — full trainer smoke: one **SFT** stage, load the saved LoRA checkpoint, then one preference stage (**IPO** / CPO or **DPO**, parametrized) on the same tiny JSON dataset.

**[`test/training/test_prepare_data.py`](test/training/test_prepare_data.py)** — `prepare_data.main()` writes Parquet; **`test_prepare_data_parquet_sft_then_ipo_collator_forward_e2e`** loads it via `PREPARED_DATA_DIR`, runs one **SFT** batch through the model, then a preference batch (`TokenizedDPOCollator`, layout shared by IPO and DPO) with chosen and rejected forwards.

Example:

```bash
export OCELOT_TRAINING_E2E=1
python3 -m pytest test/training/test_one_batch_integration.py -v
python3 -m pytest test/training/test_prepare_data.py::test_prepare_data_parquet_sft_then_ipo_collator_forward_e2e -v
```
