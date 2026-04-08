from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parent
API_ROOT = _CONFIG_DIR.parent
CONFIG_PATH = _CONFIG_DIR / "vllm_config.yaml"
OUTPUT_DIR = API_ROOT / "output"
