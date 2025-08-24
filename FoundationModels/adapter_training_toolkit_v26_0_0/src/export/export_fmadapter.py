import argparse
import enum
import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .constants import BASE_SIGNATURE, MIL_PATH
from .export_utils import AdapterConverter, AdapterSpec, DraftModelConverter, camelize

logger = logging.getLogger(__name__)


class MetadataKeys(enum.StrEnum):
    ADAPTER_IDENTIFIER = "adapter_identifier"
    BASE_SIGNATURE = "base_model_signature"
    LORA_RANK = "lora_rank"
    CREATOR_DEFINED = "creator_defined"


@dataclass
class Metadata:
    """
    Represents metadata associated with a trained adapter.

    Attributes:
        author: The developer's name.
        description: A brief description of the adapter's purpose.
        license: The license under which the adapter is distributed.
        speculative_decoding_draft_token_count: The number of tokens that the draft
            model predicts.
        creator_defined: A dictionary for storing any additional
            metadata fields defined by developer.
    """

    author: str = ""
    description: str = ""
    license: str = ""
    speculative_decoding_draft_token_count: int = 5
    creator_defined: dict[str, Any] = field(default_factory=dict)


def create_adapter_identifier(adapter_name: str) -> str:
    version_prefix = BASE_SIGNATURE[:7]
    return f"fmadapter-{adapter_name}-{version_prefix}"


def write_metadata(save_path: Path, metadata: Metadata, adapter_name: str, lora_rank: int) -> None:
    self_dict = asdict(metadata)
    self_dict[MetadataKeys.ADAPTER_IDENTIFIER] = create_adapter_identifier(adapter_name=adapter_name)
    self_dict[MetadataKeys.BASE_SIGNATURE] = BASE_SIGNATURE
    self_dict[MetadataKeys.LORA_RANK] = lora_rank

    creator_defined = None
    if MetadataKeys.CREATOR_DEFINED in self_dict:
        creator_defined = self_dict[MetadataKeys.CREATOR_DEFINED]

    camelized_dict = camelize(self_dict)
    if creator_defined is not None:
        camelized_dict[camelize(MetadataKeys.CREATOR_DEFINED)] = creator_defined

    json_obj = json.dumps(camelized_dict, sort_keys=True, indent=4, default=str, ensure_ascii=False)
    with save_path.open("w", encoding="utf-8") as f:
        f.write(json_obj)


def export_fmadapter(
    output_dir: str | Path,
    adapter_name: str,
    metadata: Metadata,
    checkpoint: str | Path | dict,
    *,
    draft_checkpoint: str | Path | dict | None = None,
) -> str:
    """
    Export trained checkpoints to .fmadapter.

    Parameters:
        output_dir (str | Path): The path to save weights
        adapter_name: (str): Adapter name
        checkpoint (str | Path): The checkpoint saved during training
        draft_checkpoint (str | Path | None): The checkpoint to trained draft model

    Returns:
        str: The path to the .fmadapter.
    """
    filename = f"{adapter_name}.fmadapter" if not adapter_name.endswith(".fmadapter") else adapter_name
    fmadapter_path = Path(output_dir) / filename
    if fmadapter_path.exists():
        logging.warning("%s exists and will be overwritten", fmadapter_path)
        shutil.rmtree(fmadapter_path)

    try:
        fmadapter_path.mkdir(parents=True, exist_ok=True)
        metadata_path = fmadapter_path / "metadata.json"
    except OSError:
        logging.exception("Error creating fmadapter.")

    adapter_converter = AdapterConverter(ckpt=checkpoint, config=AdapterSpec)
    adapter_converter.export_adapter(save_dir=fmadapter_path)

    write_metadata(
        save_path=metadata_path,
        metadata=metadata,
        adapter_name=adapter_name,
        lora_rank=adapter_converter.config.lora_rank,
    )

    if draft_checkpoint is not None:
        draft_weight_output = fmadapter_path / "draft_weights.bin"
        fmadapter_mil_path = fmadapter_path / "draft.mil"
        shutil.copy2(MIL_PATH, fmadapter_mil_path)
        draft_converter = DraftModelConverter(ckpt=draft_checkpoint)
        draft_converter.export_draft_model(
            output_weights_path=draft_weight_output,
        )
    logging.info(f"Adapter saved at {fmadapter_path}.")
    return str(fmadapter_path)


def run_from_cli() -> None:
    """
    Entry point when running this file from the CLI.

    Parse arguments from the CLI and call the `export` function.
    """

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output-dir", required=True, help="Directory where the adapter will be saved")
    parser.add_argument("-n", "--adapter-name", type=str, help="Name of the adapter", required=True)
    parser.add_argument("-c", "--checkpoint", required=True, help="Path to the checkpoint file")
    parser.add_argument(
        "-d",
        "--draft-checkpoint",
        required=False,
        help="Path to the draft checkpoint file (optional)",
    )
    parser.add_argument(
        "--author", type=str, default="3P developer", help="Author of the adapter (default: '3P developer')"
    )
    parser.add_argument("--description", type=str, default="", help="Description of the adapter, defaults to empty")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("-q", "--quiet", action="count", default=0)
    args = parser.parse_args()

    metadata = Metadata(author=args.author, description=args.description)

    result = export_fmadapter(
        output_dir=args.output_dir,
        adapter_name=args.adapter_name,
        metadata=metadata,
        checkpoint=args.checkpoint,
        draft_checkpoint=args.draft_checkpoint,
    )

    logger.info(f"Adapter exported successfully to: {result}")


if __name__ == "__main__":
    run_from_cli()
