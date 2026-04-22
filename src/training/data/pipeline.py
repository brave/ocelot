from __future__ import annotations

import json
import os
import base64
import random
from io import BytesIO
from typing import Any

import torch
from datasets import Dataset, Features, Sequence, Value, concatenate_datasets, load_dataset
from PIL import Image
from tqdm import tqdm
from qwen_vl_utils import process_vision_info

from core.config import RunConfig
from core.hf_args import _per_device_eval_batch_size, _per_device_batch_size
from data.vision import maybe_downscale_images


def _tokenized_sft_map_features() -> Features:
    """
    Fixed schema for tokenize_all_once_batched outputs.

    Without this, the first batch (often text-only) yields pixel_values=[] and HF/Arrow
    infers an inner null type; the next vision batch then fails to cast list<float> into null.
    """
    return Features(
        {
            "input_ids": Sequence(Value("int64")),
            "attention_mask": Sequence(Value("int64")),
            "labels": Sequence(Value("int64")),
            "mm_token_type_ids": Sequence(Value("int64")),
            "prompt_input_ids": Sequence(Value("int64")),
            "prompt_attention_mask": Sequence(Value("int64")),
            "prompt_mm_token_type_ids": Sequence(Value("int64")),
            "chosen_input_ids": Sequence(Value("int64")),
            "chosen_attention_mask": Sequence(Value("int64")),
            "chosen_mm_token_type_ids": Sequence(Value("int64")),
            "rejected_input_ids": Sequence(Value("int64")),
            "rejected_attention_mask": Sequence(Value("int64")),
            "rejected_mm_token_type_ids": Sequence(Value("int64")),
            "pixel_values": Sequence(Sequence(Value("float32"))),
            "image_grid_thw": Sequence(Sequence(Value("int64"))),
        }
    )


def _normalize_pixel_values_nested(pv: list | None) -> list[list[float]]:
    """
    Store pixel_values as list[list[float]] (patch rows). Processor tolist() can be 1D for a single flat patch.
    """
    if not pv:
        return []
    if not isinstance(pv, list):
        return []
    if len(pv) > 0 and not isinstance(pv[0], list):
        return [pv]  # type: ignore[list-item]
    return pv  # type: ignore[return-value]


def _trim_to_batch_multiple(dataset: Dataset, batch_size: int, world_size: int = 1) -> tuple[Dataset, int]:
    """Trim dataset so len is divisible by batch_size * world_size; return (dataset, num_dropped)."""
    effective = max(1, batch_size * world_size)
    n = len(dataset)
    remainder = n % effective
    if remainder == 0:
        return dataset, 0
    new_len = n - remainder
    return dataset.select(range(new_len)), remainder


def _example_has_image(ex: dict, *, column_names: set[str]) -> bool:
    # Prepared-data path: image grids are present after tokenization.
    if "image_grid_thw" in column_names:
        g = ex.get("image_grid_thw")
        return bool(g) and len(g) > 0
    # Raw/preprocessed path: concrete images list.
    if "images" in column_names:
        imgs = ex.get("images")
        return bool(imgs) and len(imgs) > 0
    # Raw JSON path: inspect prompt parts for image placeholders.
    if "prompt" in column_names:
        for msg in ex.get("prompt") or []:
            for part in msg.get("content") or []:
                if isinstance(part, dict) and part.get("type") in {"image_url", "image"}:
                    return True
        return False
    return False


def _drop_image_samples(dataset: Dataset, *, drop_ratio: float, seed: int) -> tuple[Dataset, int, int]:
    """
    Randomly drop `drop_ratio` of image-containing samples from a dataset.
    Returns (new_dataset, image_count, dropped_count).
    """
    if drop_ratio <= 0:
        return dataset, 0, 0
    rng = random.Random(int(seed))
    cols = set(dataset.column_names)
    keep_idx: list[int] = []
    image_count = 0
    dropped = 0
    for i, ex in tqdm(enumerate(dataset)):
        has_img = _example_has_image(ex, column_names=cols)
        if has_img:
            image_count += 1
            if rng.random() < float(drop_ratio):
                dropped += 1
                continue
        keep_idx.append(i)
    return dataset.select(keep_idx), image_count, dropped


def _num_grid_tokens(grid) -> int:
    if not grid:
        return 0
    total = 0
    for row in grid:
        if not row or len(row) != 3:
            return -1
        try:
            t, h, w = int(row[0]), int(row[1]), int(row[2])
        except Exception:
            return -1
        total += t * h * w
    return total


def _tokenized_row_is_valid(ex: dict) -> bool:
    # Required SFT fields must have matching lengths.
    for a, b in [
        ("input_ids", "attention_mask"),
        ("input_ids", "labels"),
        ("input_ids", "mm_token_type_ids"),
    ]:
        if a in ex and b in ex and ex.get(a) is not None and ex.get(b) is not None:
            if len(ex[a]) != len(ex[b]):
                return False

    # Required preference fields must have matching lengths when present.
    for a, b in [
        ("prompt_input_ids", "prompt_attention_mask"),
        ("prompt_input_ids", "prompt_mm_token_type_ids"),
        ("chosen_input_ids", "chosen_attention_mask"),
        ("chosen_input_ids", "chosen_mm_token_type_ids"),
        ("rejected_input_ids", "rejected_attention_mask"),
        ("rejected_input_ids", "rejected_mm_token_type_ids"),
    ]:
        if a in ex and b in ex and ex.get(a) is not None and ex.get(b) is not None:
            if len(ex[a]) != len(ex[b]):
                return False

    # Vision consistency: grid token count must match pixel_values token count.
    grid = ex.get("image_grid_thw")
    grid_tokens = _num_grid_tokens(grid)
    if grid_tokens < 0:
        return False

    shape = ex.get("pixel_values_shape")
    if shape:
        try:
            pv_tokens = int(shape[0])
        except Exception:
            return False
        if grid_tokens and pv_tokens != grid_tokens:
            return False
        if grid_tokens == 0 and pv_tokens > 0:
            return False
        return True

    pv = ex.get("pixel_values")
    if pv is not None:
        try:
            pv_tokens = len(pv)
        except Exception:
            return False
        if grid_tokens and pv_tokens != grid_tokens:
            return False
        if grid_tokens == 0 and pv_tokens > 0:
            return False
        return True

    # No pixel_values present: grid must also be empty.
    if grid_tokens > 0:
        return False
    return True


def _get_chat_messages(ex: dict) -> list:
    """Return chat messages from ex['text'], parsing JSON if stored as string (for Arrow compatibility)."""
    raw = ex.get("text")
    if raw is None:
        return []
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def sanitize_chat_messages(prompt_msgs):
    """
    Build a safe copy of chat messages (keeps only system/user, normalizes multimodal parts).
    IMPORTANT: never mutate dataset examples in-place.
    """
    chat_messages = []
    for msg in prompt_msgs or []:
        role = msg.get("role")
        if role not in {"system", "user"}:
            continue
        content = msg.get("content") or []
        content2 = [p.copy() if isinstance(p, dict) else p for p in content]
        msg2 = {"role": role, "content": content2}

        if role == "system":
            # Template can mis-detect images when system content is a list of parts; use a single string
            text_parts = []
            for part in msg2["content"]:
                if isinstance(part, dict) and part.get("type") != "image_url":
                    t = part.get("text") or part.get("content")
                    if t:
                        text_parts.append(t)
                elif isinstance(part, str):
                    text_parts.append(part)
            msg2["content"] = "\n".join(text_parts) if text_parts else ""
        else:
            for part in msg2["content"]:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "image_url":
                    # {"type":"image_url","image_url":{"url":"..."}} -> {"type":"image_url","image_url":"..."}
                    iu = part.get("image_url")
                    url = None
                    if isinstance(iu, dict):
                        url = iu.get("url")
                    elif isinstance(iu, str):
                        url = iu
                    if isinstance(url, str) and url:
                        part["image_url"] = url
                    part.pop("text", None)
                else:
                    # Text or any other part: drop image_url so template never sees it
                    part.pop("image_url", None)

        chat_messages.append(msg2)
    return chat_messages


def pad_1d_list(seq, *, length: int, pad_value: int, side: str = "right"):
    seq = [int(x) for x in (seq or [])]
    if len(seq) > length:
        seq = seq[-length:] if side == "left" else seq[:length]
    pad_n = length - len(seq)
    if pad_n <= 0:
        return seq
    pad = [int(pad_value)] * pad_n
    return (pad + seq) if side == "left" else (seq + pad)


def tokenize_all_once(
    ex,
    *,
    processor,
    max_length: int,
    max_prompt_length: int,
    store_vision_dtype: str = "float16",
    store_vision_as_bytes: bool = False,
    skip_if_oversized: bool = False,
):
    chat_messages = _get_chat_messages(ex)
    if chat_messages and any(m.get("role") not in {"system", "user"} for m in chat_messages):
        chat_messages = sanitize_chat_messages(chat_messages)
    images_raw = ex.get("images") or []
    images = images_raw if images_raw else None  # processor expects None when no images

    tok = processor.tokenizer
    pad_id = int(tok.pad_token_id)
    eos_id = tok.eos_token_id

    prompt_str = processor.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = processor(
        text=prompt_str,
        images=images,
        add_special_tokens=False,
        return_tensors="pt",
        padding=False,
        truncation=False,
    )
    p_ids = prompt_tokens["input_ids"].squeeze(0).to(torch.int64).tolist()
    mm_prompt = prompt_tokens.get("mm_token_type_ids", None)
    if isinstance(mm_prompt, torch.Tensor):
        if mm_prompt.dim() >= 2 and mm_prompt.size(0) == 1:
            mm_prompt = mm_prompt.squeeze(0)
        mm_prompt_ids = mm_prompt.to(torch.int64).tolist()
    else:
        mm_prompt_ids = [0] * len(p_ids)
    if len(p_ids) > int(max_prompt_length):
        raise RuntimeError(
            f"[tok] prompt_len={len(p_ids)} exceeds max_prompt_length={int(max_prompt_length)}. "
            f"Filter prompts more aggressively or increase SFT_MAX_PROMPT_LENGTH."
        )

    pv = prompt_tokens.get("pixel_values", None)
    grid = prompt_tokens.get("image_grid_thw", None)

    # Normalize vision outputs. Depending on checkpoint/processor version, Qwen can return:
    # - (N, D) flattened visual tokens, or
    # - (N, C, H, W) image tensors.
    # Keep both representations; only drop truly unsupported shapes.
    if isinstance(pv, torch.Tensor):
        while pv.dim() > 4 and pv.size(0) == 1:
            pv = pv.squeeze(0)
        if pv.dim() == 3:
            pv = pv.unsqueeze(0)
        if pv.dim() not in {2, 4}:
            pv = None
            grid = None  # drop vision for this example
    if images and pv is None:
        img_types = [type(x).__name__ for x in (images[:4] if isinstance(images, list) else [images])]
        proc_name = processor.__class__.__name__
        model_name = getattr(getattr(processor, "tokenizer", None), "name_or_path", "<unknown>")
        raise RuntimeError(
            "[tok] images provided but processor returned no pixel_values. "
            f"processor={proc_name} model/tokenizer={model_name}. "
            f"image_types={img_types}. "
            "Likely model/processor mismatch (e.g. text-only model for multimodal data) "
            "or incompatible image input format."
        )
    if isinstance(grid, torch.Tensor):
        while grid.dim() > 2 and grid.size(0) == 1:
            grid = grid.squeeze(0)
        if grid.dim() == 2 and grid.size(0) == 1:
            grid = grid.squeeze(0)  # (1, 3) -> (3,)
        if grid.dim() == 1 and grid.numel() == 3:
            grid = grid.view(1, 3)
        if grid.dim() != 2 or grid.size(1) != 3:
            grid = None

    # Invariant: never keep grid without concrete visual tokens.
    if not isinstance(pv, torch.Tensor) or pv.shape[0] <= 0:
        pv = None
        grid = None

    # Always derive grid from actual patch count so pv/grid never mismatch (preprocessing source of truth)
    if isinstance(pv, torch.Tensor) and pv.shape[0] > 0:
        n = int(pv.shape[0])
        if isinstance(grid, torch.Tensor) and grid.numel() > 0:
            g = grid.to(torch.int64)
            expected = (g[:, 0] * g[:, 1] * g[:, 2]).sum().item()
            if expected != n:
                if g.shape[0] == 1:
                    h = max(1, int(n ** 0.5))
                    while n % h != 0 and h > 1:
                        h -= 1
                    grid = g.new_tensor([[1, h, n // h]])
                else:
                    raise RuntimeError(
                        f"[tok] pv/grid mismatch: {n} patches vs grid sum {expected} (multi-image). "
                        "Check processor output or skip example."
                    )
        else:
            # No grid from processor; set from patch count (single image)
            h = max(1, int(n ** 0.5))
            while n % h != 0 and h > 1:
                h -= 1
            grid = pv.new_tensor([[1, h, n // h]], dtype=torch.int64)

    pixel_values = None
    pixel_values_bytes = None
    pixel_values_shape = None
    if isinstance(pv, torch.Tensor):
        if store_vision_dtype in {"fp16", "float16"}:
            pv = pv.to(dtype=torch.float16)
        else:
            pv = pv.to(dtype=torch.float32)
        if store_vision_as_bytes:
            arr = pv.cpu().numpy()
            pixel_values_bytes = arr.tobytes()
            pixel_values_shape = list(arr.shape)
        else:
            pixel_values = _normalize_pixel_values_nested(pv.cpu().tolist())

    image_grid_thw = None
    if isinstance(pv, torch.Tensor) and isinstance(grid, torch.Tensor):
        image_grid_thw = grid.to(torch.int64).cpu().tolist()

    chosen_text = ex.get("chosen") or ""
    answer_ids = tok(chosen_text, add_special_tokens=False, truncation=False).input_ids
    if eos_id is not None:
        answer_ids = [int(x) for x in answer_ids] + [int(eos_id)]
    else:
        answer_ids = [int(x) for x in answer_ids]

    max_answer = max(0, int(max_length) - len(p_ids))
    if len(answer_ids) > max_answer:
        answer_ids = answer_ids[:max_answer]
    sft_ids = p_ids + answer_ids
    sft_labels = ([-100] * len(p_ids)) + answer_ids
    sft_mm = mm_prompt_ids + ([0] * len(answer_ids))

    sft_input_ids = pad_1d_list(sft_ids, length=int(max_length), pad_value=pad_id, side="right")
    sft_attention = pad_1d_list([1] * len(sft_ids), length=int(max_length), pad_value=0, side="right")
    sft_labels = pad_1d_list(sft_labels, length=int(max_length), pad_value=-100, side="right")
    sft_mm = pad_1d_list(sft_mm, length=int(max_length), pad_value=0, side="right")

    prompt_input_ids = pad_1d_list(p_ids, length=int(max_prompt_length), pad_value=pad_id, side="left")
    prompt_attention = pad_1d_list([1] * len(p_ids), length=int(max_prompt_length), pad_value=0, side="left")
    prompt_mm = pad_1d_list(mm_prompt_ids, length=int(max_prompt_length), pad_value=0, side="left")

    chosen_conv = chat_messages + [{"role": "assistant", "content": [{"type": "text", "text": chosen_text}]}]
    rejected_conv = chat_messages + [
        {"role": "assistant", "content": [{"type": "text", "text": ex.get("rejected") or ""}]}
    ]
    chosen_str = processor.apply_chat_template(chosen_conv, tokenize=False)
    rejected_str = processor.apply_chat_template(rejected_conv, tokenize=False)

    chosen_tokens = processor(
        text=chosen_str,
        images=images,
        return_tensors="pt",
        padding=False,
        truncation=False,
        add_special_tokens=False,
    )
    rejected_tokens = processor(
        text=rejected_str,
        images=images,
        return_tensors="pt",
        padding=False,
        truncation=False,
        add_special_tokens=False,
    )
    c_ids = chosen_tokens["input_ids"].squeeze(0).to(torch.int64).tolist()
    r_ids = rejected_tokens["input_ids"].squeeze(0).to(torch.int64).tolist()
    mm_chosen = chosen_tokens.get("mm_token_type_ids", None)
    if isinstance(mm_chosen, torch.Tensor):
        if mm_chosen.dim() >= 2 and mm_chosen.size(0) == 1:
            mm_chosen = mm_chosen.squeeze(0)
        c_mm = mm_chosen.to(torch.int64).tolist()
    else:
        c_mm = [0] * len(c_ids)
    mm_rejected = rejected_tokens.get("mm_token_type_ids", None)
    if isinstance(mm_rejected, torch.Tensor):
        if mm_rejected.dim() >= 2 and mm_rejected.size(0) == 1:
            mm_rejected = mm_rejected.squeeze(0)
        r_mm = mm_rejected.to(torch.int64).tolist()
    else:
        r_mm = [0] * len(r_ids)
    if len(c_ids) > int(max_length) or len(r_ids) > int(max_length):
        if skip_if_oversized:
            return None
        raise RuntimeError(
            f"[tok] dpo sequence exceeded max_length={int(max_length)} (chosen={len(c_ids)}, rejected={len(r_ids)}). "
            f"Filter these examples or increase SFT_MAX_LENGTH."
        )
    chosen_input_ids = pad_1d_list(c_ids, length=int(max_length), pad_value=pad_id, side="right")
    rejected_input_ids = pad_1d_list(r_ids, length=int(max_length), pad_value=pad_id, side="right")
    chosen_attention = pad_1d_list([1] * len(c_ids), length=int(max_length), pad_value=0, side="right")
    rejected_attention = pad_1d_list([1] * len(r_ids), length=int(max_length), pad_value=0, side="right")
    chosen_mm = pad_1d_list(c_mm, length=int(max_length), pad_value=0, side="right")
    rejected_mm = pad_1d_list(r_mm, length=int(max_length), pad_value=0, side="right")

    out = {
        "input_ids": sft_input_ids,
        "attention_mask": sft_attention,
        "labels": sft_labels,
        "mm_token_type_ids": sft_mm,
        "prompt_input_ids": prompt_input_ids,
        "prompt_attention_mask": prompt_attention,
        "prompt_mm_token_type_ids": prompt_mm,
        "chosen_input_ids": chosen_input_ids,
        "chosen_attention_mask": chosen_attention,
        "chosen_mm_token_type_ids": chosen_mm,
        "rejected_input_ids": rejected_input_ids,
        "rejected_attention_mask": rejected_attention,
        "rejected_mm_token_type_ids": rejected_mm,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }
    if store_vision_as_bytes and pixel_values_bytes is not None:
        out["pixel_values_bytes"] = pixel_values_bytes
        out["pixel_values_shape"] = pixel_values_shape
    return out


def tokenize_all_once_batched(
    batch,
    *,
    processor,
    max_length: int,
    max_prompt_length: int,
    store_vision_dtype: str = "float16",
    store_vision_as_bytes: bool = False,
):
    n = len(batch["chosen"])
    outs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "mm_token_type_ids": [],
        "prompt_input_ids": [],
        "prompt_attention_mask": [],
        "prompt_mm_token_type_ids": [],
        "chosen_input_ids": [],
        "chosen_attention_mask": [],
        "chosen_mm_token_type_ids": [],
        "rejected_input_ids": [],
        "rejected_attention_mask": [],
        "rejected_mm_token_type_ids": [],
        "pixel_values": [],
        "image_grid_thw": [],
    }
    if store_vision_as_bytes:
        outs["pixel_values_bytes"] = []
        outs["pixel_values_shape"] = []
    for i in range(n):
        ex_i = {
            "text": batch["text"][i],
            "chosen": batch["chosen"][i],
            "rejected": batch["rejected"][i],
            "images": batch.get("images", [None] * n)[i],
        }
        out_i = tokenize_all_once(
            ex_i,
            processor=processor,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            store_vision_dtype=store_vision_dtype,
            store_vision_as_bytes=store_vision_as_bytes,
        )
        for k in outs:
            # Append exactly once per key per example to keep all columns aligned.
            v = out_i.get(k)
            # Arrow cannot mix null with list<list<...>> in one column when batching text+vision rows.
            if k in ("pixel_values", "image_grid_thw") and v is None:
                v = []
            outs[k].append(v)
    return outs


def tokenize_all_once_batched_allow_oversize_skip(
    batch,
    *,
    processor,
    max_length: int,
    max_prompt_length: int,
    store_vision_dtype: str = "float16",
    store_vision_as_bytes: bool = False,
):
    """
    Like tokenize_all_once_batched, but drops rows where full chosen/rejected sequences exceed max_length
    (prepare_data only — HF dataset.map expects fixed batch size from tokenize_all_once_batched).
    Returns (outs, chosen_kept, rejected_kept, n_skipped).
    """
    n = len(batch["chosen"])
    outs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "mm_token_type_ids": [],
        "prompt_input_ids": [],
        "prompt_attention_mask": [],
        "prompt_mm_token_type_ids": [],
        "chosen_input_ids": [],
        "chosen_attention_mask": [],
        "chosen_mm_token_type_ids": [],
        "rejected_input_ids": [],
        "rejected_attention_mask": [],
        "rejected_mm_token_type_ids": [],
        "pixel_values": [],
        "image_grid_thw": [],
    }
    if store_vision_as_bytes:
        outs["pixel_values_bytes"] = []
        outs["pixel_values_shape"] = []
    chosen_kept: list = []
    rejected_kept: list = []
    n_skipped = 0
    for i in range(n):
        ex_i = {
            "text": batch["text"][i],
            "chosen": batch["chosen"][i],
            "rejected": batch["rejected"][i],
            "images": batch.get("images", [None] * n)[i],
        }
        out_i = tokenize_all_once(
            ex_i,
            processor=processor,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            store_vision_dtype=store_vision_dtype,
            store_vision_as_bytes=store_vision_as_bytes,
            skip_if_oversized=True,
        )
        if out_i is None:
            n_skipped += 1
            continue
        chosen_kept.append(batch["chosen"][i])
        rejected_kept.append(batch["rejected"][i])
        for k in outs:
            v = out_i.get(k)
            if k in ("pixel_values", "image_grid_thw") and v is None:
                v = []
            outs[k].append(v)
    return outs, chosen_kept, rejected_kept, n_skipped


def _preprocess_example(example: dict, *, vision_max_pixels: int | None) -> dict:
    prompt = example.get("prompt") or []
    chat_messages = sanitize_chat_messages(prompt)
    try:
        images, _ = process_vision_info(chat_messages)
    except ValueError as e:
        if "aspect ratio" in str(e).lower() or "max_ratio" in str(e).lower():
            return {
                "text": json.dumps(chat_messages),
                "chosen": example.get("chosen", ""),
                "rejected": example.get("rejected", ""),
                "images": [],
                "_skip": True,
            }
        raise
    # Normalize data-URL strings to PIL so downstream processor always receives concrete image objects.
    normalized_images = []
    bad_image = False
    for img in (images or []):
        if isinstance(img, Image.Image):
            normalized_images.append(img)
            continue
        if isinstance(img, str) and img.startswith("data:image/"):
            try:
                b64 = img.split(",", 1)[1]
                data = base64.b64decode(b64)
                normalized_images.append(Image.open(BytesIO(data)).convert("RGB"))
            except Exception:
                bad_image = True
            continue
        normalized_images.append(img)
    if bad_image:
        return {
            "text": json.dumps(chat_messages),
            "chosen": example.get("chosen", ""),
            "rejected": example.get("rejected", ""),
            "images": [],
            "_skip": True,
        }
    images = maybe_downscale_images(normalized_images, vision_max_pixels=vision_max_pixels)

    num_placeholders = sum(
        1 for msg in chat_messages for c in msg["content"] if isinstance(c, dict) and c.get("type") == "image_url"
    )
    num_images = len(images) if images else 0
    assert num_images == num_placeholders, f"Mismatch: {num_images} images vs {num_placeholders} placeholders"

    chosen = example["chosen"]
    if isinstance(chosen, str) and chosen.startswith("["):
        chosen = chosen[1:]
    if isinstance(chosen, str) and chosen.endswith("]"):
        chosen = chosen[:-1]

    return {
        "text": json.dumps(chat_messages),
        "chosen": chosen,
        "rejected": example["rejected"],
        "images": images if images else [],
    }


def _prompt_within_limit(ex: dict, *, processor, max_prompt_length: int) -> bool:
    chat_messages = _get_chat_messages(ex)
    if chat_messages and any(m.get("role") not in {"system", "user"} for m in chat_messages):
        chat_messages = sanitize_chat_messages(chat_messages)
    prompt_str = processor.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
    images_raw = ex.get("images") or []
    images = images_raw if images_raw else None
    toks = processor(
        text=[prompt_str],
        images=images,
        return_tensors="pt",
        padding=False,
        truncation=False,
        add_special_tokens=True,
    )
    prompt_len = int(toks.input_ids.shape[1])
    margin = int(os.environ.get("SFT_PROMPT_FILTER_MARGIN", "0"))
    limit = int(max_prompt_length) - margin
    return prompt_len < limit


def _load_json_field(path: str, field: str, *, chunk_size: int = 50_000) -> Dataset:
    """Load a JSON field into a Dataset in chunks to avoid Arrow 32-bit offset overflow."""
    with open(path) as f:
        data = json.load(f)
    rows = data.get(field) or []
    if not rows:
        return Dataset.from_list([])
    if len(rows) <= chunk_size:
        return Dataset.from_list(rows)
    chunks = [Dataset.from_list(rows[i : i + chunk_size]) for i in range(0, len(rows), chunk_size)]
    return concatenate_datasets(chunks)


def load_and_prepare_datasets(cfg: RunConfig, *, processor: Any):
    try:
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    except Exception:
        rank = 0
        world_size = 1
    shuffle_seed = int(os.environ.get("DATA_SHUFFLE_SEED", "42"))
    image_drop_ratio = float(os.environ.get("TRAIN_IMAGE_DROP_RATIO", "0.5"))

    # Stage 2: load from prepared Parquet (Stage 1 = prepare_data.py)
    if cfg.prepared_data_dir:
        def _parquet_files(prefix: str) -> list[str]:
            dirpath = cfg.prepared_data_dir
            all_f = [f for f in os.listdir(dirpath) if f.startswith(prefix) and f.endswith(".parquet")]
            shards = sorted(f for f in all_f if f != f"{prefix}.parquet" and f[len(prefix)] == "-")
            if shards:
                return [os.path.join(dirpath, f) for f in shards]
            single = os.path.join(dirpath, f"{prefix}.parquet")
            return [single] if os.path.isfile(single) else []

        train_files = _parquet_files("train")
        val_files = _parquet_files("validation")
        if train_files and val_files:
            if rank == 0:
                print(f"[data] Stage 2: loading from prepared data {cfg.prepared_data_dir}", flush=True)
            ds = load_dataset(
                "parquet",
                data_files={"train": train_files, "validation": val_files},
            )
            dataset = ds["train"]
            val_dataset = ds["validation"]
            dataset, n_img, n_drop = _drop_image_samples(
                dataset, drop_ratio=image_drop_ratio, seed=shuffle_seed
            )
            if rank == 0:
                print(
                    f"[data] train image downsample: images={n_img}, dropped={n_drop}, "
                    f"drop_ratio={image_drop_ratio}",
                    flush=True,
                )
            dataset = dataset.shuffle(seed=shuffle_seed)
            if rank == 0:
                print(f"[data] shuffled train split (seed={shuffle_seed})", flush=True)
            torch_cols = [
                "input_ids",
                "attention_mask",
                "labels",
                "mm_token_type_ids",
                "prompt_input_ids",
                "prompt_attention_mask",
                "prompt_mm_token_type_ids",
                "chosen_input_ids",
                "chosen_attention_mask",
                "chosen_mm_token_type_ids",
                "rejected_input_ids",
                "rejected_attention_mask",
                "rejected_mm_token_type_ids",
            ]
            train_cols = [c for c in torch_cols if c in dataset.column_names]
            val_cols = [c for c in torch_cols if c in val_dataset.column_names]
            before_train = len(dataset)
            before_val = len(val_dataset)
            dataset = dataset.filter(
                _tokenized_row_is_valid,
                desc="Validate tokenized train rows",
                load_from_cache_file=False,
                writer_batch_size=1000,
            )
            val_dataset = val_dataset.filter(
                _tokenized_row_is_valid,
                desc="Validate tokenized val rows",
                load_from_cache_file=False,
                writer_batch_size=1000,
            )
            if rank == 0:
                dt = before_train - len(dataset)
                dv = before_val - len(val_dataset)
                print(f"[data] dropped invalid tokenized rows: train={dt}, val={dv}", flush=True)
            train_bs, eval_bs = _per_device_batch_size(), _per_device_eval_batch_size()
            dataset, train_drop = _trim_to_batch_multiple(dataset, train_bs, world_size)
            val_dataset, val_drop = _trim_to_batch_multiple(val_dataset, eval_bs, world_size)
            if rank == 0 and (train_drop or val_drop):
                print(
                    f"[data] trimmed for batch divisibility: train_dropped={train_drop}, val_dropped={val_drop} "
                    f"(train_bs={train_bs}, eval_bs={eval_bs}, world_size={world_size})",
                    flush=True,
                )
            # Drop assistant-only strings so TRL CPOTrainer does not run maybe_extract_prompt on them.
            _str_pref = [c for c in ("chosen", "rejected") if c in dataset.column_names]
            if _str_pref:
                dataset = dataset.remove_columns(_str_pref)
                _val_str = [c for c in _str_pref if c in val_dataset.column_names]
                if _val_str:
                    val_dataset = val_dataset.remove_columns(_val_str)
            dataset.set_format(type="torch", columns=train_cols, output_all_columns=True)
            val_dataset.set_format(type="torch", columns=val_cols, output_all_columns=True)
            return dataset, val_dataset
        if rank == 0:
            print(f"[data] PREPARED_DATA_DIR set but no train/validation Parquet found.", flush=True)
        raise FileNotFoundError(
            f"Prepared data dir {cfg.prepared_data_dir} missing train*.parquet and validation*.parquet. "
            "Run prepare_data.py first or unset PREPARED_DATA_DIR to load from JSON."
        )

    dataset = _load_json_field(cfg.data_path, "train")
    val_dataset = _load_json_field(cfg.data_path, "validation")
    dataset, n_img, n_drop = _drop_image_samples(
        dataset, drop_ratio=image_drop_ratio, seed=shuffle_seed
    )
    if rank == 0:
        print(
            f"[data] train image downsample: images={n_img}, dropped={n_drop}, "
            f"drop_ratio={image_drop_ratio}",
            flush=True,
        )

    # Optional: for testing, use only first N and last N samples (env DATA_HEAD_TAIL_SAMPLES=10)
    head_tail = os.environ.get("DATA_HEAD_TAIL_SAMPLES", "").strip()
    if head_tail:
        n = max(0, int(head_tail))
        if n > 0:
            def _head_tail_indices(length: int) -> list[int]:
                if length <= 2 * n:
                    return list(range(length))
                return sorted({*range(0, n), *range(length - n, length)})
            train_idx = _head_tail_indices(len(dataset))
            val_idx = _head_tail_indices(len(val_dataset))
            dataset = dataset.select(train_idx)
            val_dataset = val_dataset.select(val_idx)
            if rank == 0:
                print(f"[data] test subset: train {len(train_idx)}, val {len(val_idx)} (DATA_HEAD_TAIL_SAMPLES={n})", flush=True)

    # writer_batch_size=1000 avoids PyArrow "offset overflow" when combining list columns (token ids, images)
    _map_kw = {"load_from_cache_file": False, "writer_batch_size": 1000}
    dataset = dataset.map(
        lambda ex: _preprocess_example(ex, vision_max_pixels=cfg.vision_max_pixels),
        remove_columns=dataset.column_names,
        desc="Preprocess train",
        **_map_kw,
    )
    val_dataset = val_dataset.map(
        lambda ex: _preprocess_example(ex, vision_max_pixels=cfg.vision_max_pixels),
        remove_columns=val_dataset.column_names,
        desc="Preprocess val",
        **_map_kw,
    )

    # Drop examples with extreme-aspect-ratio images (qwen_vl_utils MAX_RATIO=200)
    if "_skip" in dataset.column_names:
        n_before = len(dataset) + len(val_dataset)
        dataset = dataset.filter(
            lambda ex: not ex.get("_skip", False),
            desc="Drop extreme aspect ratio",
            load_from_cache_file=False,
            writer_batch_size=1000,
        )
        val_dataset = val_dataset.filter(
            lambda ex: not ex.get("_skip", False),
            desc="Drop extreme aspect ratio (val)",
            load_from_cache_file=False,
            writer_batch_size=1000,
        )
        n_after = len(dataset) + len(val_dataset)
        dataset = dataset.remove_columns(["_skip"])
        val_dataset = val_dataset.remove_columns(["_skip"])
        if rank == 0 and n_before != n_after:
            print(f"[data] dropped {n_before - n_after} examples with extreme image aspect ratio (>200)", flush=True)

    orig_train_len = len(dataset)
    orig_val_len = len(val_dataset)

    dataset = dataset.filter(
        lambda ex: _prompt_within_limit(ex, processor=processor, max_prompt_length=cfg.sft_max_prompt_length),
        load_from_cache_file=False,
        writer_batch_size=1000,
    )
    val_dataset = val_dataset.filter(
        lambda ex: _prompt_within_limit(ex, processor=processor, max_prompt_length=cfg.sft_max_prompt_length),
        load_from_cache_file=False,
        writer_batch_size=1000,
    )
    if rank == 0:
        dropped_train = orig_train_len - len(dataset)
        dropped_val = orig_val_len - len(val_dataset)
        print(
            f"[sft] dropped long prompts > {int(cfg.sft_max_prompt_length)} tokens: "
            f"train {orig_train_len}->{len(dataset)}, val {orig_val_len}->{len(val_dataset)}",
            flush=True,
        )
        print(
            f"[sft] dropped counts: train_dropped={dropped_train}, val_dropped={dropped_val}, "
            f"filter_limit={int(cfg.sft_max_prompt_length)} margin={int(os.environ.get('SFT_PROMPT_FILTER_MARGIN', '0'))}",
            flush=True,
        )

    if rank == 0:
        print("[tok] precomputing padded SFT+DPO inputs via dataset.map(...)", flush=True)

    _tok_features = _tokenized_sft_map_features()
    # Drop string labels from the *output* table. If we only remove text/images, HF still forwards
    # chosen/rejected into ArrowWriter; with explicit `features=` that omits them → KeyError('chosen').
    _tok_remove = ["text", "images", "chosen", "rejected"]
    dataset = dataset.map(
        lambda batch: tokenize_all_once_batched(
            batch,
            processor=processor,
            max_length=int(cfg.sft_max_length),
            max_prompt_length=int(cfg.sft_max_prompt_length),
            store_vision_dtype=cfg.store_vision_dtype,
        ),
        remove_columns=_tok_remove,
        batched=True,
        batch_size=max(1, int(cfg.tokenize_batch_size)),
        desc="Tokenize+pad train (SFT+DPO)",
        load_from_cache_file=False,
        writer_batch_size=1000,
        features=_tok_features,
    )
    val_dataset = val_dataset.map(
        lambda batch: tokenize_all_once_batched(
            batch,
            processor=processor,
            max_length=int(cfg.sft_max_length),
            max_prompt_length=int(cfg.sft_max_prompt_length),
            store_vision_dtype=cfg.store_vision_dtype,
        ),
        remove_columns=_tok_remove,
        batched=True,
        batch_size=max(1, int(cfg.tokenize_batch_size)),
        desc="Tokenize+pad val (SFT+DPO)",
        load_from_cache_file=False,
        writer_batch_size=1000,
        features=_tok_features,
    )
    before_train = len(dataset)
    before_val = len(val_dataset)
    dataset = dataset.filter(
        _tokenized_row_is_valid,
        desc="Validate tokenized train rows",
        load_from_cache_file=False,
        writer_batch_size=1000,
    )
    val_dataset = val_dataset.filter(
        _tokenized_row_is_valid,
        desc="Validate tokenized val rows",
        load_from_cache_file=False,
        writer_batch_size=1000,
    )
    if rank == 0:
        dt = before_train - len(dataset)
        dv = before_val - len(val_dataset)
        print(f"[data] dropped invalid tokenized rows: train={dt}, val={dv}", flush=True)
    dataset = dataset.shuffle(seed=shuffle_seed)
    if rank == 0:
        print(f"[data] shuffled train split (seed={shuffle_seed})", flush=True)

    train_bs, eval_bs = _per_device_batch_size(), _per_device_eval_batch_size()
    dataset, train_drop = _trim_to_batch_multiple(dataset, train_bs, world_size)
    val_dataset, val_drop = _trim_to_batch_multiple(val_dataset, eval_bs, world_size)
    if rank == 0 and (train_drop or val_drop):
        print(
            f"[data] trimmed for batch divisibility: train_dropped={train_drop}, val_dropped={val_drop} "
            f"(train_bs={train_bs}, eval_bs={eval_bs}, world_size={world_size})",
            flush=True,
        )

    torch_cols = [
        "input_ids",
        "attention_mask",
        "labels",
        "mm_token_type_ids",
        "prompt_input_ids",
        "prompt_attention_mask",
        "prompt_mm_token_type_ids",
        "chosen_input_ids",
        "chosen_attention_mask",
        "chosen_mm_token_type_ids",
        "rejected_input_ids",
        "rejected_attention_mask",
        "rejected_mm_token_type_ids",
    ]
    train_cols = [c for c in torch_cols if c in dataset.column_names]
    val_cols = [c for c in torch_cols if c in val_dataset.column_names]
    dataset.set_format(type="torch", columns=train_cols, output_all_columns=True)
    val_dataset.set_format(type="torch", columns=val_cols, output_all_columns=True)

    return dataset, val_dataset


