from __future__ import annotations

import os

from transformers import TrainerCallback


def resolve_periodic_eval_steps(trainer, *, default: int = 50) -> int:
    """
    Interval for PeriodicEvalCallback. EVAL_STEPS env overrides; otherwise use
    TrainingArguments.eval_steps so hf_args.py settings apply without exporting env.
    Use EVAL_STEPS=0 to disable this callback entirely.
    """
    raw = os.environ.get("EVAL_STEPS", "").strip()
    if raw != "":
        return max(0, int(raw))
    steps = getattr(trainer.args, "eval_steps", None)
    if steps is not None:
        return max(0, int(steps))
    return max(0, int(default))


class PeriodicEvalCallback(TrainerCallback):
    """Run evaluation every `eval_steps` (see resolve_periodic_eval_steps for how that value is chosen)."""

    def __init__(self, trainer, eval_steps: int):
        self.trainer = trainer
        self.eval_steps = int(eval_steps)

    def on_step_end(self, args, state, control, **kwargs):
        if self.eval_steps <= 0:
            return control
        if state.global_step > 0 and (state.global_step % self.eval_steps) == 0:
            self.trainer.evaluate()
        return control


def enable_gradient_checkpointing_for_lora(model) -> None:
    """
    With LoRA + gradient checkpointing, PyTorch can warn that none of the inputs require grads.
    This mirrors the helper from `train_script.py`.
    """
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass


