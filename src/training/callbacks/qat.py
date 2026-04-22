from __future__ import annotations

import torch
from transformers import TrainerCallback

# Experimental QAT callback for simulated quantisation during training

class StableSimulatedQuant:
    def __init__(self, target_bits: int = 4, warmup_steps: int = 120):
        self.target_bits = int(target_bits)
        self.warmup_steps = int(warmup_steps)
        self.current_step = 0
        self.momentum = 0.95
        self.running_max = {}
        self.ste_dampener = 0.5
        self.burn_in_steps = 20

    def hook(self, module, input, output):
        if not module.training:
            return output

        mod_id = id(module)

        with torch.no_grad():
            batch_std = output.std() + 1e-5
            batch_range = batch_std * 3.5
            if mod_id not in self.running_max:
                self.running_max[mod_id] = batch_range
            else:
                self.running_max[mod_id] = (self.momentum * self.running_max[mod_id]) + (
                    (1 - self.momentum) * batch_range
                )

            current_max = self.running_max[mod_id]

        if self.current_step < self.burn_in_steps:
            return output

        effective_step = max(0, self.current_step - self.burn_in_steps)
        progress = min(1.0, effective_step / float(self.warmup_steps))
        current_bits = 16 - (progress * (16 - self.target_bits))
        levels = 2 ** (current_bits - 1)

        scale = (current_max / levels).clamp(min=1e-6)
        rounded = torch.round(output / scale) * scale
        if current_bits < 14:
            rounded = torch.clamp(rounded, -current_max, current_max)

        return output + (rounded - output).detach() * self.ste_dampener


class QATCallback(TrainerCallback):
    def __init__(self, qat_instance: StableSimulatedQuant):
        self.qat = qat_instance

    def on_step_end(self, args, state, control, **kwargs):
        self.qat.current_step = state.global_step
        return control


def apply_noise_to_base_only(model, target_substrings: list[str], *, quant_sim: StableSimulatedQuant) -> int:
    """
    Registers the QAT hook on the *base* layers inside LoRA-wrapped modules, since the LoRA will be kept in the 
    same precision as it was trained in.
    """
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, "base_layer") and any(k in name for k in target_substrings) and "visual" not in name:
            module.base_layer.register_forward_hook(quant_sim.hook)
            count += 1
    return count


