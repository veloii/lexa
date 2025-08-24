# Copyright © 2025 Apple Inc.

import gc
import logging
import random
import re
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, ClassVar, Literal  # noqa: UP035

import numpy as np
import tamm
import torch
from pydantic import BaseModel
from tamm.layers.transformer import TransformerStack
from tamm.tokenizers.afm import AFMTokenizer
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    checkpoint_wrapper,
)
from torch.utils.checkpoint import checkpoint

from src.train.messages import InstructMessages

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_model_assets_dir() -> Path:
    return PROJECT_ROOT / "assets"


def get_logger(level: str | int | None = logging.DEBUG) -> logging.Logger:
    """
    Get a logger with a stream handler.

    Args:
        level (str | int | None): The logging level.

    Returns:
        logging.Logger: The logger.
    """
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logger.addHandler(logging.StreamHandler())
    if level is not None:
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        logger.setLevel(level)
    return logger


def supports_bfloat16(device: torch.device | None) -> bool:
    """Returns a Boolean value indicating whether the given device supports bfloat16 or not."""
    if device is None:
        return False
    if device.type == "cuda":
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if device.type == "mps":
        return torch.backends.mps.is_available() and torch.backends.mps.is_built()
    if device.type == "cpu":
        return False

    msg = f"Unsupport device `{device}`"
    raise ValueError(msg)


@contextmanager
def empty_cache() -> any:
    """
    Context manager that empties the CUDA or MPS cache and forces garbage collection
    on exiting the wrapped function.
    """
    try:
        yield
    finally:
        if get_device().type == "cuda":
            torch.cuda.empty_cache()
        if get_device().type == "mps":
            torch.mps.empty_cache()
        _ = gc.collect()


def apply_activation_checkpointing(
    model: TransformerStack,
    checkpoint_module_names: list[str] | None = None,
    checkpoint_every_n: int = 2,
) -> None:
    """
    Applies activation checkpointing to any module that matches any of the names in the given list
    at the specified frequency.

    The function is a noop if `checkpoint_frequency` is less than or equal to zero.

    Args:
        model (TransformerStack): The model to apply activation checkpointing to.
        checkpoint_module_names (list[str] | None): List of names of modules to wrap.
        checkpoint_every_n (int): An integer representing the frequency of modules to wrap for activation
            checkpointing (default: 2).
    """
    if checkpoint_every_n <= 0:
        return

    checkpoint_module_names = checkpoint_module_names or ["TransformerAttention", "TransformerFeedForward"]

    # Track how many times each module is visited.
    checkpoint_module_counts = {checkpoint_module: 0 for checkpoint_module in checkpoint_module_names}

    def _apply_activation_checkpointing(module: torch.nn.Module) -> None:
        nonlocal checkpoint_module_counts
        for child_key, child in module.named_children():
            for target_module_name in checkpoint_module_names:
                if not re.search(target_module_name, child.__class__.__name__):
                    continue
                checkpoint_module_count = checkpoint_module_counts.get(target_module_name, 0)
                checkpoint_module_counts[target_module_name] = checkpoint_module_count + 1
                if checkpoint_module_count % checkpoint_every_n != 0:
                    continue
                wrapped_child = checkpoint_wrapper(
                    child,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                    checkpoint_fn=checkpoint,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
                setattr(module, child_key, wrapped_child)

    model.apply(_apply_activation_checkpointing)

    # Ensure all given module names were matched
    for module_name, module_count in checkpoint_module_counts.items():
        if module_count == 0:
            msg = f"Failed to match module with name `{module_name}` when applying activation checkpointing"
            raise ValueError(msg)


def load_base_model(
    dtype: torch.dtype | None,
    device: torch.device | None,
    checkpoint_path: str | Path | None = None,
) -> TransformerStack:
    """
    Load and return the on-device foundation model.

    Args:
        dtype (torch.dtype | None): The model floating-point data-type to cast the model
        device (torch.device | None): Device to place the model
        checkpoint_path (str | Path | None = None): Optional path to a checkpoint of the model weights

    Returns:
        TransformerStack: The on-device foundation model.
    """
    model_config_path = get_model_assets_dir() / "base-model-config.json"
    with model_config_path.open() as f:
        model_config = tamm.utils.json.load(f)
    model_config.dtype = dtype or model_config.dtype
    model = model_config.create_model()

    # Always load the base model first...
    base_model_checkpoint_path = get_model_assets_dir() / "base-model.pt"
    with Path(base_model_checkpoint_path).open("rb") as f:
        sd = torch.load(f, map_location=device, weights_only=False)
        _ = model.load_state_dict(sd, strict=True)

    # And override adapter layers if checkpoint is given.
    if checkpoint_path:
        with Path(checkpoint_path).open("rb") as f:
            sd = torch.load(f, map_location=device, weights_only=False)
            # Relaxed to support loading only the adapters from a checkpoint
            _ = model.load_state_dict(sd, strict=False)

    # Freeze all parameters expect the adapters
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "adapter" in name

    return model.to(device=device, dtype=model_config.dtype)


def load_draft_model(
    dtype: torch.dtype | None,
    device: torch.device | None,
    checkpoint_path: str | Path | None = None,
) -> TransformerStack:
    """
    Load and return the on-device draft model.

    Args:
        dtype (torch.dtype | None): The model floating-point data-type to cast the model
        device (torch.device | None): Device to place the model
        checkpoint_path (str | Path | None = None): Optional path to a checkpoint of the model weights
        pretrained (bool): A Boolean value indicating whether to restore the
            model from pretrained weights or not (default: True)

    Returns:
        TransformerStack: The on-device draft model.
    """
    model_config_path = get_model_assets_dir() / "draft-model-config.json"
    with model_config_path.open() as f:
        model_config = tamm.utils.json.load(f)
    model_config.device = device or model_config.device
    model_config.dtype = dtype or model_config.dtype
    model = model_config.create_model()
    if checkpoint_path is None:
        checkpoint_path = get_model_assets_dir() / "draft-model.pt"
    with Path(checkpoint_path).open("rb") as f:
        sd = torch.load(f, map_location=device, weights_only=False)
        model.load_state_dict(sd, strict=True)
    return model


def load_tokenizer() -> AFMTokenizer:
    """Load and return the tokenizer.

    Returns:
        AFMTokenizer: A new instance of the tokenizer.
    """
    config_path = get_model_assets_dir() / "tokenizer-config.json"
    with config_path.open() as f:
        config = tamm.utils.json.load(f)
    vocab_path = get_model_assets_dir() / "tokenizer.model"
    tokenizer = config.create_tokenizer(vocab_path=vocab_path)

    # Set the ignore token id which is used for ignoring tokens in labels.
    if not hasattr(tokenizer, "ignore_id"):
        tokenizer.ignore_id = -100
    return tokenizer


def set_seed(seed: int | None = None) -> int:
    """
    Sets the seed for pseudo-random number generators.

    Args:
        seed (int | None): An integer value representing the seed for pseudo-random number
            generators. If `None`, a random seed will be generated.

    Returns:
        int: The seed

    Raises:
        ValueError: If the input seed value is outside a valid range.
    """
    max_val = np.iinfo(np.uint32).max
    min_val = np.iinfo(np.uint32).min
    seed = np.random.randint(min_val, max_val) if seed is None else max(min(max_val, seed), min_val)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    return seed


def get_device() -> torch.device:
    """Returns the default device for the current platform."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@contextmanager
def autocast(device: torch.device, dtype: torch.dtype | None = torch.bfloat16) -> any:
    """
    Context manager to faciliate autocast.

    - Calls `torch.autocast(device_type=device.type, dtype=dtype)` for supported devices and
        compatible data types.

    Args:
        device (torch.device): The device to use
        dtype (torch.dtype | None): The precision type, or `None` to disable (default: `torch.bfloat16`)
    """
    autocast_enabled = dtype is not None and dtype in (torch.float16, torch.bfloat16)
    if autocast_enabled and torch.amp.autocast_mode.is_autocast_available(device.type):
        with torch.autocast(device.type, dtype=dtype):
            yield
    else:
        yield


PrecisionType = Literal["f32", "bf16", "bf16-mixed", "f16-mixed"]


class PrecisionConfig:
    """
    Parses a pre-defined precision type to obtain the model and compute type.

    Deep learning models can use either a single precision (f32) or mixed precision.
    Mixed precision significantly speeds up training and reduces memory demands by primarily using
    lower-precision formats like f16 for computations. Crucially, it maintains model accuracy by
    selectively employing higher precision (f32) for sensitive operations and parameter updates,
    delivering performance gains without sacrificing results.

    The following list the supported precision types used in the examples:
    - "f32" : Model parameters and compute use Float32
    - "bf16" : Model parameters and compute use BFloat16
    - "bf16-mixed" : Model parameters are stored as Float32 and compute uses a combination of BFloat16 and Float32
    - "f16-mixed" : Model parameters are stored as Float32 and compute uses a combination of Float16 and Float32
    """

    def __init__(self, precision: PrecisionType) -> None:
        """
        Creates a new precision configuration.

        Args:
            precision (PrecisionType): Precision type
        """
        self.precision_type = precision.strip()
        self.model_dtype = PrecisionConfig._convert_model_dtype_from_str(precision)
        self.compute_dtype = PrecisionConfig._convert_compute_dtype_from_str(precision)

    def __str__(self) -> str:
        return self.precision_type

    @property
    def use_autocast(self) -> bool:
        """A Boolean value indicating whether autocast is enabled or not"""
        return self.precision_type in ("bf16-mixed", "f16-mixed")

    @staticmethod
    def _convert_model_dtype_from_str(precision: PrecisionType) -> torch.dtype:
        """Parses the precision type to a Pytorch data type"""
        if precision == "bf16":
            return torch.bfloat16
        if precision == "f16":
            return torch.float16
        if precision in ("f32", "bf16-mixed", "f16-mixed"):
            return torch.float32
        msg = f"Unsupported precision `{precision}` for model parameters"
        raise ValueError(msg)

    @staticmethod
    def _convert_compute_dtype_from_str(precision: PrecisionType) -> torch.dtype:
        """Parses the precision type to a Pytorch data type"""
        if precision == "bf16":
            return torch.bfloat16
        if precision == "f32":
            return torch.float32
        if precision == "bf16-mixed":
            return torch.bfloat16
        if precision in ("f16", "f16-mixed"):
            return torch.float16
        msg = f"Unsupported precision `{precision}` for compute"
        raise ValueError(msg)


class CheckpointSaver(ABC):
    checkpoint_dir: Path
    prefix: str
    best_checkpoint_count: int
    best_checkpoint_mode: str
    save_last: bool
    best_checkpoints: list[tuple[float, Path]]

    def __init__(
        self,
        checkpoint_dir: Path | str,
        prefix: str,
        best_checkpoint_count: int = 2,
        best_checkpoint_mode: Literal["min", "max"] = "min",
        save_last: bool = True,
    ) -> None:
        """
        Creates a new checkpoint saver.

        Args:
            checkpoint_dir (Path | str): Directory where checkpoints are saved
            prefix (str): Prefix assigned to the filename of the checkpoint
            best_checkpoint_count (int): Number of 'best' (best loss) checkpoints to keep
            best_checkpoint_mode (Literal["min", "max"]): Direction of metric comparison,
                ("min" for loss, "max" for accuracy)
            save_last (bool): Boolean value indicating whether to save the last checkpoint
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        assert best_checkpoint_count > 0
        self.best_checkpoint_count = best_checkpoint_count
        assert best_checkpoint_mode in ["min", "max"]
        self.best_checkpoint_mode = best_checkpoint_mode
        self.save_last = save_last
        self.best_checkpoints = []

    @abstractmethod
    def _is_saving(self, key: str, value: torch.Tensor) -> bool:
        pass

    def save(self, state_dict: dict[str, torch.Tensor], epoch: int, metric: float) -> None:
        """
        Saves the given state dictionary if it ranks among the top N checkpoints according to
        the corresponding metric.

        Args:
            state_dict (dict[str, torch.Tensor]): Model's weights to save
            epoch (int): Current epoch
            metric (float): Current loss
        """

        state_dict = {k: v for k, v in state_dict.items() if self._is_saving(k, v)}

        if self.save_last:
            torch.save(state_dict, self.checkpoint_dir / f"{self.prefix}-final.pt")

        is_saving = False
        if len(self.best_checkpoints) < self.best_checkpoint_count:
            is_saving = True
        else:
            best_checkpoint_metric = self.best_checkpoints[-1][0]
            is_saving = (metric < best_checkpoint_metric and self.best_checkpoint_mode == "min") or (
                metric > best_checkpoint_metric and self.best_checkpoint_mode == "max"
            )

        if not is_saving:
            return

        checkpoint_path = self.checkpoint_dir / f"{self.prefix}-epoch{epoch}.pt"
        torch.save(state_dict, checkpoint_path)

        logging.info(f"Saving checkpoint {checkpoint_path} with metric {metric}")

        # Track checkpoint
        self.best_checkpoints.append((metric, checkpoint_path))
        self.best_checkpoints = sorted(
            self.best_checkpoints, key=lambda x: x[0], reverse=(self.best_checkpoint_mode == "max")
        )

        # Remove checkpoint with lowest score if best_checkpoint_count is exceeded
        if len(self.best_checkpoints) > self.best_checkpoint_count:
            _, checkpoint_path = self.best_checkpoints.pop()
            checkpoint_path.unlink(missing_ok=True)


class DraftModelCheckpointSaver(CheckpointSaver):
    def _is_saving(self, key: str, value: torch.Tensor) -> bool:  # noqa: ARG002
        return True


class AdapterCheckpointSaver(CheckpointSaver):
    non_default_parameters: set[str]
    required_parameters: set[str]

    def __init__(
        self,
        checkpoint_dir: Path | str,
        prefix: str,
        model: torch.nn.Module,
        best_checkpoint_count: int = 2,
        best_checkpoint_mode: Literal["min", "max"] = "min",
        save_last: bool = True,
        layer_filter_fn: Callable[[str], bool] | None = None,
    ) -> None:
        super().__init__(checkpoint_dir, prefix, best_checkpoint_count, best_checkpoint_mode, save_last)
        self.layer_filter_fn = layer_filter_fn
        self.non_default_parameters = {param[0] for param in model.named_parameters() if param[1].requires_grad}

    def _is_saving(self, key: str, value: torch.Tensor) -> bool:  # noqa: ARG002
        return key in self.non_default_parameters


class SchemaAugmenter:
    FIELD_ORDER: ClassVar[list[str]] = [
        "type",
        "description",
        "enum",
        "items",
        "properties",
        "required",
        "additionalProperties",
        "$defs",
        "anyOf",
    ]

    @classmethod
    def _get_schema_from_data_model(cls, data_model: BaseModel) -> dict[str, Any]:
        base_schema = data_model.model_json_schema(ref_template="#/$defs/{model}")
        data_model_name = data_model.__name__

        x_order = list(base_schema.get("required", []))

        extras = {"title": data_model_name, "additionalProperties": False, "x-order": x_order}

        base_schema.update(extras)

        defs = base_schema.get("$defs")
        if defs:
            for model_ref in defs.values():
                model_ref.update(extras)

        cls._enforce_property_key_order(base_schema)

        for def_schema in base_schema.get("$defs", {}).values():
            cls._enforce_property_key_order(def_schema)

        return {
            "type": "json_schema",
            "json_schema": {
                "name": f"Response<{data_model_name}>",
                "strict": "true",
                "schema": {
                    **base_schema,
                    "$defs": base_schema.get("$defs", {}),
                },
            },
        }

    @classmethod
    def _enforce_property_key_order(cls, schema: dict) -> None:
        if "properties" in schema:
            for key, prop in schema["properties"].items():
                schema["properties"][key] = {k: prop[k] for k in cls.FIELD_ORDER if k in prop} | {
                    k: v for k, v in prop.items() if k not in cls.FIELD_ORDER
                }

    @classmethod
    def _validate_field_order(cls, schema: dict, path: str = "root_lvl") -> bool:
        res = True

        if isinstance(schema, dict):
            seen = [key for key in schema if key in cls.FIELD_ORDER]
            expected = sorted(seen, key=cls.FIELD_ORDER.index)

            if seen != expected:
                res = False

            for key, value in schema.items():
                new_path = f"{path}.{key}"
                if isinstance(value, dict):
                    res &= cls._validate_field_order(value, new_path)

        return res

    @classmethod
    def _reorder_schema_fields(cls, schema: dict) -> dict:
        if not isinstance(schema, dict):
            return schema

        reordered = {}

        for key in cls.FIELD_ORDER:
            if key in schema:
                value = schema[key]
                reordered[key] = cls._recurse(value)

        for key in schema:
            if key not in reordered:
                reordered[key] = cls._recurse(schema[key])

        return reordered

    @classmethod
    def _recurse(cls, value: dict | list | str) -> dict | list | str:
        if isinstance(value, dict):
            return cls._reorder_schema_fields(value)
        if isinstance(value, list):
            return [cls._reorder_schema_fields(v) if isinstance(v, dict) else v for v in value]
        return value

    @classmethod
    def _convert_schema_data(cls, data_model: BaseModel) -> dict[str, any]:
        schema = cls._get_schema_from_data_model(data_model)
        if not SchemaAugmenter._validate_field_order(schema):
            schema = SchemaAugmenter._reorder_schema_fields(schema)
        return schema

    @staticmethod
    def augment_schema_by_sample(data_model: BaseModel) -> Callable[[InstructMessages, int], InstructMessages]:
        """
        Generate a Callable for guided generation (single-turn).

        Args:
            input (BaseModel): A Pydantic BaseModel class that defines the expected
                output schema for the assistant's response.

        Returns:
            Callable[[InstructMessages, int], InstructMessages]: A transformation
                function to augment instruction messages with schema to guide the model's response.

        Example:
            from pydantic import BaseModel

            class Date(BaseModel):
                day: int = Field(description="The day")
                month: int = Field(description="The month")
                year: int = Field(description="A four digit year")

            dataset = InstructMessagesDataset(
                ...
                transform=augment_schema_guided_generation(Date)
            )
        """
        schema = SchemaAugmenter._convert_schema_data(data_model=data_model)

        def _transform_data(sample: InstructMessages, sample_idx: int) -> InstructMessages:
            sample = sample[0] if sample and isinstance(sample[0], list) else sample

            for message in sample:
                if message["role"] == "user":
                    message.update({"response_format": schema})
            return sample

        return _transform_data

    @staticmethod
    def _augment_schema_guided_generation(data_model_mapping: dict) -> Callable:
        """
        Utility for guided generation schema injection.

        Args:
            data_model_mapping (dict): A map of sample index or range of indices as key,
                message or range of messages as value.

        Returns:
            Callable[[InstructMessages, int], InstructMessages]: A transformation
                function to augment instruction messages with schema to guide the model's response.
        """
        message_level_mapping, sample_level_mapping = SchemaAugmenter._parse_mappings(data_model_mapping)
        sample_level_transforms = SchemaAugmenter._create_sample_level_transforms(sample_level_mapping)

        def _transform_data(sample: InstructMessages, sample_idx: int) -> InstructMessages:
            if sample_idx in sample_level_transforms:
                return sample_level_transforms[sample_idx](sample, sample_idx)

            return SchemaAugmenter._apply_message_level_transforms(sample, sample_idx, message_level_mapping)

        return _transform_data

    @staticmethod
    def _parse_mappings(data_model_mapping: dict) -> tuple[dict, dict]:
        """Parse schema mappings into message-level and sample-level mappings."""
        message_level_mapping = {}
        sample_level_mapping = {}

        for key, data_model in data_model_mapping.items():
            match key:
                # Sample level mapping supported for either "all" and/or range of samples
                case (sample_key, "all") if isinstance(sample_key, range):
                    for sample_idx in sample_key:
                        sample_level_mapping[sample_idx] = data_model
                case (sample_key, "all"):
                    sample_level_mapping[sample_key] = data_model
                # Message level mapping
                case (int(), int()):
                    message_level_mapping[key] = data_model
                case _:
                    raise ValueError(f"Unsupported mapping format: {key}")

        return message_level_mapping, sample_level_mapping

    @staticmethod
    def _create_sample_level_transforms(sample_level_mapping: dict) -> dict:
        """Function mappings from data model to precompiled callable function"""
        # Optimization to avoid recreating functions every epoch
        return {
            sample_idx: SchemaAugmenter.augment_schema_by_sample(data_model)
            for sample_idx, data_model in sample_level_mapping.items()
        }

    @staticmethod
    def _apply_message_level_transforms(
        sample: InstructMessages, sample_idx: int, message_level_mapping: dict
    ) -> InstructMessages:
        """Apply schema transforms to specific messages in a sample."""
        # Normalize data
        sample = sample[0] if sample and isinstance(sample[0], list) else sample

        user_msg_idx = 0
        for message in sample:
            if message["role"] == "user":
                key = (sample_idx, user_msg_idx)
                if key in message_level_mapping:
                    schema = SchemaAugmenter._convert_schema_data(message_level_mapping[key])
                    message.update({"response_format": schema})
                user_msg_idx += 1

        return sample


def augment_schema_guided_generation(
    data_model_mapping: dict,
) -> Callable[[InstructMessages, int], InstructMessages]:
    """
    Generate a Callable for guided generation.

    Args:
        data_model_mapping (dict): A map of sample index or range of sample indices as key,
            message index or range of message indices as value. See examples for details.

    Returns:
        Callable[[InstructMessages, int], InstructMessages]: A transformation
            function to augment instruction messages with schema to guide the model's response.

    Handle schema mapping with support for:
    - (sample_idx, user_msg_idx): specific message.
        i.e. (0, 0): Character
        # inject Character into sample 0, user message 0.
    - (sample_idx, "all"): all user messages in sample
        i.e. (2, "all"): Date
        # inject Date into all user messages of sample 2.
    - (range(start, end), "all"): all user messages in sample range
        i.e. (range(5, 8), "all"): Character
        # inject Character into all user messages of samples 5-7.

    Example:
        # Sample level data model schema injection
        from pydantic import BaseModel

        class Date(BaseModel):
            day: int = Field(description="The day")
            month: int = Field(description="The month")
            year: int = Field(description="A four digit year")

        singleturn_map = {
            (range(0, 1), "all"): Date
        }

        dataset = InstructMessagesDataset(
            ...
            transform=augment_schema_guided_generation(singleturn_map)
        )

        # Message level data model schema injection (multiturn)
        class Date(BaseModel):
            day: int = Field(description="The day")
            month: int = Field(description="The month")
            year: int = Field(description="A four digit year")

        class Character(BaseModel):
            name: str = Field(description="A creative name of the animal")
            age: int = Field(description="Age of the animal, from 1 to 100")


        multiturn_map = {
            (0, 0): Character, # Sample 0, user message 0.
                (sample_idx, user_msg_idx): specific message
            (1, 0): Date,
            (1, 1): Character,
            (2, "all"): Date, # Sample 2, Date.
                (sample_idx, "all"): all user messages in sample
            (range(5, 8), "all"): Character, # Samples 5-7, Character.
                (range(start, end), "all"): all user messages in sample range
        }

        dataset = InstructMessagesDataset(
            ...
            transform=augment_schema_guided_generation(multiturn_map)
        )

    """
    if isinstance(data_model_mapping, dict):
        return SchemaAugmenter._augment_schema_guided_generation(data_model_mapping)
    err = "Input must be a dictionary mapping according to documentation."
    raise TypeError(err)

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    if denominator == 0 or denominator == 0.0:
        return default
    try:
        return numerator / denominator
    except (ZeroDivisionError, TypeError):
        return default

def safe_int_divide(numerator: int, denominator: int, default: int = 1) -> int:
    """Safely divide two integers, returning default if denominator is zero"""
    if denominator == 0:
        return default
    try:
        return numerator // denominator
    except (ZeroDivisionError, TypeError):
        return default
