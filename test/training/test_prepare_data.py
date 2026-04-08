"""Tests for `prepare_data.py`: Parquet staging and compatibility with collators + model.

E2E (opt-in): run `prepare_data`, load Parquet, then one **SFT** batch forward and one **IPO-style**
batch (same `TokenizedDPOCollator` as IPO) with chosen + rejected forwards.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest


def _e2e_enabled() -> bool:
    return os.environ.get("OCELOT_TRAINING_E2E", "").strip().lower() in {"1", "true", "yes"}


def _sft_like_row(
    *,
    seq_len: int,
    prompt_len: int,
    pixel_values_bytes: bytes | None,
    pixel_values_shape: list[int] | None,
    image_grid_thw: list,
) -> dict:
    """Minimal row shaped like `prepare_data` output (fixed lengths, valid under `_tokenized_row_is_valid`)."""
    pad_id = 0
    p_ids = [pad_id + i for i in range(prompt_len)]
    ans = [10, 11]
    sft_ids = p_ids + ans
    sft_ids = sft_ids[:seq_len]
    while len(sft_ids) < seq_len:
        sft_ids.append(pad_id)
    sft_labels = ([-100] * len(p_ids)) + ans
    sft_labels = sft_labels[:seq_len]
    while len(sft_labels) < seq_len:
        sft_labels.append(-100)
    mm = [0] * seq_len
    attn = [1] * seq_len

    pp = [pad_id] * prompt_len
    pp_mask = [1] * prompt_len
    pp_mm = [0] * prompt_len

    def pad_right(xs: list, L: int, fill):
        out = xs[:L]
        return out + [fill] * (L - len(out))

    c_ids = pad_right([20, 21, 22], seq_len, pad_id)
    r_ids = pad_right([30, 31], seq_len, pad_id)
    c_mask = [1 if x != pad_id else 0 for x in c_ids]
    r_mask = [1 if x != pad_id else 0 for x in r_ids]

    return {
        "input_ids": sft_ids,
        "attention_mask": attn,
        "labels": sft_labels,
        "mm_token_type_ids": mm,
        "prompt_input_ids": pp,
        "prompt_attention_mask": pp_mask,
        "prompt_mm_token_type_ids": pp_mm,
        "chosen_input_ids": c_ids,
        "chosen_attention_mask": c_mask,
        "chosen_mm_token_type_ids": [0] * seq_len,
        "rejected_input_ids": r_ids,
        "rejected_attention_mask": r_mask,
        "rejected_mm_token_type_ids": [0] * seq_len,
        "pixel_values_bytes": pixel_values_bytes,
        "pixel_values_shape": pixel_values_shape,
        "image_grid_thw": image_grid_thw,
        "chosen": "ok",
        "rejected": "no",
    }


def test_chunk_to_arrow_roundtrip_through_parquet() -> None:
    pytest.importorskip("pyarrow")
    np = pytest.importorskip("numpy")
    from data.parquet_rows import chunk_tokenized_rows_to_arrow
    # Text-only + vision row in one chunk (mirrors mixed batches from `prepare_data`).
    n_patch, d = 2, 4
    arr = np.zeros((n_patch, d), dtype=np.float16)
    vision_row = _sft_like_row(
        seq_len=32,
        prompt_len=8,
        pixel_values_bytes=arr.tobytes(),
        pixel_values_shape=list(arr.shape),
        image_grid_thw=[[1, 1, n_patch]],
    )
    text_row = _sft_like_row(
        seq_len=32,
        prompt_len=8,
        pixel_values_bytes=None,
        pixel_values_shape=None,
        image_grid_thw=[],
    )
    table = chunk_tokenized_rows_to_arrow([text_row, vision_row])
    assert table.num_rows == 2
    assert "input_ids" in table.column_names
    assert "pixel_values_bytes" in table.column_names

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "mix.parquet"
        import pyarrow.parquet as pq

        pq.write_table(table, path)
        back = pq.read_table(path)
    assert back.num_rows == 2


def _forward_chosen_or_rejected(model, mb: dict, device, *, branch: str) -> None:
    """One causal forward for chosen or rejected; IPO/CPO use the same tensors + shared prompt vision."""
    import torch

    assert branch in {"chosen", "rejected"}
    kwargs = {
        "input_ids": mb[f"{branch}_input_ids"].to(device),
        "attention_mask": mb[f"{branch}_attention_mask"].to(device),
        "labels": mb[f"{branch}_labels"].to(device),
        "mm_token_type_ids": mb[f"{branch}_mm_token_type_ids"].to(device),
    }
    pv = mb.get("prompt_pixel_values")
    gr = mb.get("prompt_image_grid_thw")
    if pv is not None:
        kwargs["pixel_values"] = pv.to(device)
    if gr is not None:
        kwargs["image_grid_thw"] = gr.to(device)
    out = model(**kwargs)
    loss = getattr(out, "loss", None)
    logits = getattr(out, "logits", None)
    assert loss is not None or logits is not None
    if loss is not None:
        assert torch.isfinite(loss).item()


@pytest.mark.integration
def test_prepare_data_parquet_sft_then_ipo_collator_forward_e2e(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiny_sft_json: Path
) -> None:
    if not _e2e_enabled():
        pytest.skip("Set OCELOT_TRAINING_E2E=1 to run (downloads processor/model, GPU forward).")

    torch = pytest.importorskip("torch")
    has_cuda = torch.cuda.is_available()
    has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if not has_cuda and not has_mps:
        pytest.skip("CUDA or Apple MPS is required for this integration test.")

    pytest.importorskip("pyarrow")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("datasets")
    pytest.importorskip("qwen_vl_utils")
    import prepare_data

    if has_mps:
        monkeypatch.setenv("OCELOT_LOAD_IN_4BIT", "0")

    load_4bit = os.environ.get("OCELOT_LOAD_IN_4BIT", "1").strip().lower() in {"1", "true", "yes"}
    if load_4bit:
        pytest.importorskip("bitsandbytes")

    model_name = os.environ.get("OCELOT_E2E_MODEL_NAME", "Qwen/Qwen3-VL-2B-Instruct").strip()
    prepared_dir = tmp_path / "prepared"

    monkeypatch.setenv("TRAIN_IMAGE_DROP_RATIO", "0")
    monkeypatch.setenv("STORE_VISION_DTYPE", "float16")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_data",
            "--data-path",
            str(tiny_sft_json),
            "--output-dir",
            str(prepared_dir),
            "--model-name",
            model_name,
            "--max-length",
            "1024",
            "--max-prompt-length",
            "512",
            "--tokenize-batch-size",
            "2",
            "--chunk-size",
            "10",
            "--vision-max-pixels",
            "262144",
        ],
    )
    prepare_data.main()

    train_shards = sorted(prepared_dir.glob("train-*.parquet"))
    val_shards = sorted(prepared_dir.glob("validation-*.parquet"))
    assert train_shards, f"expected train-*.parquet under {prepared_dir}"
    assert val_shards, f"expected validation-*.parquet under {prepared_dir}"
    assert (prepared_dir / "config.json").is_file()

    monkeypatch.setenv("OCELOT_ATTN_IMPLEMENTATION", "sdpa")

    from core.config import RunConfig
    from data.collators import TokenizedDPOCollator, TokenizedSFTCollator
    from data.pipeline import load_and_prepare_datasets
    from modeling.factory import build_model_and_processor

    cfg = RunConfig(
        trainer="sft",
        epochs=1,
        model_name=model_name,
        data_path=str(tiny_sft_json),
        output_dir=str(tmp_path / "run"),
        resume_from=None,
        deepspeed=None,
        prepared_data_dir=str(prepared_dir),
        sft_max_length=1024,
        sft_max_prompt_length=512,
        tokenize_batch_size=1,
        store_vision_dtype="float16",
        vision_max_pixels=None,
        lora_rank=8,
        sft_warmup_epochs=0,
        sft_learning_rate=3e-4,
        pref_learning_rate=5e-6,
        ipo_beta=0.1,
        dpo_beta=0.1,
        enable_qat=False,
    )

    bundle = build_model_and_processor(cfg)
    train_ds, _val_ds = load_and_prepare_datasets(cfg, processor=bundle.processor)
    assert len(train_ds) >= 1, "prepared Parquet produced an empty train split after pipeline filters"

    n = min(2, len(train_ds))
    features = [train_ds[i] for i in range(n)]
    device = torch.device("cuda" if has_cuda else "mps")
    bundle.model.eval()
    bundle.model.to(device)

    sft_collator = TokenizedSFTCollator(vision_dtype=cfg.store_vision_dtype)
    sft_batch = sft_collator(features)
    sft_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sft_batch.items()}

    with torch.no_grad():
        out_sft = bundle.model(**sft_inputs)

    loss_sft = getattr(out_sft, "loss", None)
    logits_sft = getattr(out_sft, "logits", None)
    assert loss_sft is not None or logits_sft is not None
    if loss_sft is not None:
        assert torch.isfinite(loss_sft).item()
    if logits_sft is not None:
        assert logits_sft.shape[0] == n

    # IPO uses TokenizedDPOCollator (CPOTrainer); same layout as DPO — exercise prepared Parquet + vision bytes.
    ipo_collator = TokenizedDPOCollator(vision_dtype=cfg.store_vision_dtype)
    ipo_batch = ipo_collator(features)

    with torch.no_grad():
        _forward_chosen_or_rejected(bundle.model, ipo_batch, device, branch="chosen")
        _forward_chosen_or_rejected(bundle.model, ipo_batch, device, branch="rejected")
