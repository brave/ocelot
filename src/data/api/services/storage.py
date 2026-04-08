import json
import uuid
from pathlib import Path

from ..config.paths import OUTPUT_DIR
from .messages import CompletionPlan, messages_for_json_storage


def _write_example(path: Path, prompt_msgs: list[dict], chosen: str, rejected: str) -> None:
    with open(path, "w") as f:
        json.dump(
            {
                "prompt": messages_for_json_storage(prompt_msgs),
                "chosen": chosen,
                "rejected": rejected,
            },
            f,
            indent=2,
        )
        f.write("\n")


def store_examples(
    plan: CompletionPlan, chosen_t: str, chosen_v: str, rej_t: str, rej_v: str
) -> list[Path]:
    """
    One JSON per modality, same shape: { prompt, chosen, rejected }.
    prompt is the chosen-side messages (page + good instruction, or images + good instruction).
    If both text and image arms ran: {uuid}_text.json and {uuid}_image.json.
    Otherwise a single {uuid}.json.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = uuid.uuid4().hex
    paths: list[Path] = []
    has_text = plan.chosen_text_msgs is not None
    has_image = plan.chosen_vision_msgs is not None

    if has_text and has_image:
        p_t = OUTPUT_DIR / f"{base}_text.json"
        p_i = OUTPUT_DIR / f"{base}_image.json"
        _write_example(p_t, plan.chosen_text_msgs, chosen_t, rej_t)
        _write_example(p_i, plan.chosen_vision_msgs, chosen_v, rej_v)
        paths.extend([p_t, p_i])
    elif has_text:
        p = OUTPUT_DIR / f"{base}.json"
        _write_example(p, plan.chosen_text_msgs, chosen_t, rej_t)
        paths.append(p)
    elif has_image:
        p = OUTPUT_DIR / f"{base}.json"
        _write_example(p, plan.chosen_vision_msgs, chosen_v, rej_v)
        paths.append(p)
    return paths
