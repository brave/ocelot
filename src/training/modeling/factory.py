from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from callbacks.qat import QATCallback, StableSimulatedQuant, apply_noise_to_base_only
from core.config import RunConfig
from core.torch_patches import patch_torch_linspace_steps
from data.vision import configure_qwen_processor_image_limits
from modeling.liger_kernels import maybe_apply_liger_kernel


@dataclass(frozen=True)
class ModelBundle:
    model: torch.nn.Module
    processor: object
    callbacks: list[object]


def _local_rank() -> int:
    try:
        return int(os.environ.get("LOCAL_RANK", "0"))
    except Exception:
        return 0


def build_model_and_processor(cfg: RunConfig) -> ModelBundle:
    patch_torch_linspace_steps()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    local_rank = _local_rank()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
    # Pin each distributed rank to its own GPU during from_pretrained (default cuda:0 otherwise).
    load_device_map = {"": local_rank} if use_cuda else None

    processor = AutoProcessor.from_pretrained(cfg.model_name)
    configure_qwen_processor_image_limits(processor, vision_max_pixels=cfg.vision_max_pixels)

    config = AutoConfig.from_pretrained(cfg.model_name)

    maybe_apply_liger_kernel(cfg)

    attn_impl = os.environ.get("OCELOT_ATTN_IMPLEMENTATION", "flash_attention_2").strip() or "flash_attention_2"
    load_4bit = os.environ.get("OCELOT_LOAD_IN_4BIT", "1").strip().lower() in {"1", "true", "yes"}

    if load_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            attn_implementation=attn_impl,
            device_map=load_device_map,
            config=config,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            device_map=load_device_map,
            config=config,
        )
    
    target_substrings = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "merger.linear_fc",
        "deepstack_merger_list",
    ]
    if "Qwen/Qwen3.5" in cfg.model_name:
        target_substrings = [
            "q_proj", 
            "k_proj", 
            "v_proj", 
            "o_proj",
            "out_proj", 
            "in_proj_qkv", 
            "in_proj_z", 
            "in_proj_b", 
            "in_proj_a",
            "gate_proj", 
            "up_proj", 
            "down_proj",
            "merger.linear_fc1", 
            "merger.linear_fc2",
        ]
    target_modules = [
        name
        for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear) and any(k in name for k in target_substrings) and "visual" not in name
    ]
    lora_config = LoraConfig(
        r=int(cfg.lora_rank),
        lora_alpha=int(cfg.lora_rank) * 2,
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    if cfg.resume_from:
        model = PeftModel.from_pretrained(model, cfg.resume_from, is_trainable=True)
        if (os.environ.get("LOCAL_RANK", "0")) == "0":
            print(f"[model] loaded PEFT adapter from {cfg.resume_from}", flush=True)
    else:
        model = get_peft_model(model, lora_config)

    # Match the original script: ensure LoRA params are trainable.
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True

    # Enable checkpointing only if requested (saves memory but slows steps; set GRADIENT_CHECKPOINTING=0 for speed).
    if os.environ.get("GRADIENT_CHECKPOINTING", "1").strip().lower() in {"1", "true", "yes"}:
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass

    callbacks: list[object] = []

    if cfg.enable_qat:
        quant_sim = StableSimulatedQuant(target_bits=cfg.qat_target_bits, warmup_steps=cfg.qat_warmup_steps)
        injected = apply_noise_to_base_only(model, target_substrings, quant_sim=quant_sim)
        callbacks.append(QATCallback(quant_sim))

    # Ensure model is on an accelerator before Trainer init (mirrors script).
    if os.environ.get("MOVE_MODEL_TO_CUDA_BEFORE_TRAINER", "1").strip().lower() in {"1", "true", "yes"}:
        if torch.cuda.is_available():
            model.to(torch.device(f"cuda:{local_rank}"))
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            model.to("mps")

    # Freeze vision encoder and MoE routers/gates (mirrors script).
    for name, param in model.named_parameters():
        if any(k in name for k in ["vision_encoder", "visual_encoder", "router", "gate"]):
            param.requires_grad = False

    try:
        model.config.use_cache = False
    except Exception:
        pass

    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    # TRL CPOTrainer (IPO) does model.warnings_issued["estimate_tokens"] = True; PEFT wrappers and
    # some model classes do not define this (PreTrainedModel does). Attach on the outer module.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    return ModelBundle(model=model, processor=processor, callbacks=callbacks)


