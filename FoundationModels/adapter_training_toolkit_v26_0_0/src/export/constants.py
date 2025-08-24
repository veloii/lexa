from pathlib import Path

BASE_SIGNATURE = "9799725ff8e851184037110b422d891ad3b92ec1"
EXPORT_ASSETS = Path(__file__).absolute().parents[1] / "assets"
WEIGHT_SPEC = EXPORT_ASSETS / "checkpoint_spec.yaml"
WEIGHTS_TEMPLATE = EXPORT_ASSETS / "weights_template.bin"
MIL_PATH = EXPORT_ASSETS / "draft.mil"
METADATA_FILENAME = "metadata.json"
