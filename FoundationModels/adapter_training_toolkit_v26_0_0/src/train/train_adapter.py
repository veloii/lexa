# Copyright © 2025 Apple Inc.

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from tamm.layers.transformer import TransformerStack
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import BatchKey, create_dataloader
from .messages import InstructMessages
from .utils import (
    AdapterCheckpointSaver,
    PrecisionConfig,
    PrecisionType,
    apply_activation_checkpointing,
    autocast,
    empty_cache,
    get_device,
    get_logger,
    load_base_model,
    load_tokenizer,
    safe_divide,
    safe_int_divide,
    supports_bfloat16,
)

logger = get_logger()

MAX_CONTEXT_LEN = 4096


def _create_optimizer(model: torch.nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    """
    Creates a AdamW optimizer.

    Args:
        model (torch.nn.Module): Model's parameters to optimize
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay coefficient

    Returns:
        torch.optim.Optimizer: Optimizer
    """
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def _create_learning_rate_scheduler(
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    gradient_accumulation_steps: int,
    training_samples: int,
    epochs: int,
    linear_warmup_epochs: int = 1,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Creates a cosine learning rate scheduler with a linear warmup.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer whose learning rate will be scheduled
        batch_size (int): Mini-batch size
        gradient_accumulation_steps (int): Number of steps over which gradients should be
            accumulated before an optimization step is performed
        training_samples (int): Total training samples
        epochs (int): Training iterations (or epochs)
        linear_warmup_epochs (int): Number of epochs to warmup.

    Returns:
        torch.optim.lr_scheduler.LRScheduler: Learning rate scheduler
    """
    scaled_batch_size = batch_size * max(1, gradient_accumulation_steps)
    steps_per_epochs = max(1, safe_int_divide(training_samples, scaled_batch_size, 1))
    warmup_steps = int(steps_per_epochs * linear_warmup_epochs)
    total_training_steps = max(1, int(steps_per_epochs * epochs))

    def lambda_lr(step: int) -> float:
        """Cosine learning rate scheduler with a linear warmup"""
        if step < warmup_steps:
            return safe_divide(float(step), float(max(1, warmup_steps)), 0.0)
        t = safe_divide(float(step - warmup_steps), float(max(1, total_training_steps - warmup_steps)), 0.0)
        lr_multiplier = 0.5 * (1.0 + math.cos(math.pi * t))
        return max(0.0, lr_multiplier)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)


class CrossEntropyLoss(torch.nn.Module):
    """Cross-entropy loss function with that supports ignoring the padding and ignore index."""

    def __init__(self, pad_id: int, ignore_id: int, reduction: Literal["mean", "sum"] = "mean") -> None:
        """Create a new instance of `CrossEntropyLoss`.

        Args:
            pad_id (int): Padding token ID
            ignore_id (int): Ignore token ID
            reduction (Literal["mean", "sum"]): Specifies how to aggregate the loss:
                - 'mean': Average the loss over valid tokens, then average over the batch.
                - 'sum': Sum the loss over valid tokens, then average over the batch.
        """
        super().__init__()
        self.pad_id = pad_id
        self.ignore_id = ignore_id
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Computes the cross-entropy from the logits produced by the model and given labels.

        Args:
            logits (torch.Tensor): A [batch size, seq length, vocab_size] tensor produced by the model.
            labels (torch.Tensor): A [batch size, seq length] integer tensor of the
                corresponding labels.

        Returns:
            torch.Tensor: The mean loss for the given batch.
        """
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)

        labels = torch.where(labels == self.pad_id, self.ignore_id, labels).to(torch.int64)

        return torch.nn.functional.cross_entropy(
            logits,
            labels,
            ignore_index=self.ignore_id,
            reduction=self.reduction,
        )


@dataclass
class AdapterTrainingConfiguration:
    epochs: int
    """The number of times to iterate over the training set"""

    learning_rate: float
    """The initial step size to update the parameters"""

    batch_size: int = 4
    """The size of the mini-batch (default: 4)"""

    linear_warmup_epochs: int = 1
    """
    The warmup period (in epochs) to gradually increase the learning rate from an initial value
    to its target (default: 1)
    """

    gradient_accumulation_steps: int = 1
    """An integer representing how many steps to accumulate gradients before updating parameters (default: 1)"""

    enable_activation_checkpointing: bool = False
    """
    A Boolean value indicating whether to enable activation checkpointing or not (default: False)

    When `True`, some activations are recomputed during the backward pass to reduce memory usage at the
    expense of compute.
    """

    precision: PrecisionType = "bf16-mixed"
    """
    Precision type (default: bf16-mixed)
    """

    compile_model: bool = False
    """
    A Boolean value indicating whether to compile the model and loss function (default: False)

    This will be ignored if not supported by the specified device.
    """

    weight_decay: float = 1e-2
    """The weight decay coefficient (default: 1e-2)"""

    clip_grad_norm: float = 1.0
    """The maximum norm of the gradients (default: 1.0)"""

    max_sequence_length: int | None = None
    """
    The maximum sequence length. Must be set if `pack_sequences` is enabled (default: None)
    """

    fixed_sized_sequences: bool = False
    """
    A Boolean value indicating whether unpacked sequences should be padded to `max_sequence_length` -
    `max_sequence_length` must be set (default: False)
    """

    pack_sequences: bool = False
    """
    A Boolean value indicating whether to pack multiple sequences together to improve
    throughput - `max_sequence_length` must be set (default: False)
    """

    loss_update_frequency: int = 3
    """An integer representing the frequency (or interval) which the average loss is updated."""

    @property
    def use_gradient_scaling(self) -> bool:
        """Boolean value indicating whether gradient scaling is enabled or not"""
        return self.precision == "f16"

    @property
    def model_dtype(self) -> torch.dtype:
        """Data-type to store the model parameters"""
        return PrecisionConfig(self.precision).model_dtype

    @property
    def autocast_dtype(self) -> torch.dtype | None:
        """Autocast data-type for compute, returning `None` if autocast should be disabled"""
        if PrecisionConfig(self.precision).use_autocast:
            return PrecisionConfig(self.precision).compute_dtype
        return None

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> "AdapterTrainingConfiguration":
        return AdapterTrainingConfiguration(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            linear_warmup_epochs=args.warmup_epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            enable_activation_checkpointing=args.activation_checkpointing,
            precision=args.precision,
            compile_model=args.compile_model,
            weight_decay=args.weight_decay,
            clip_grad_norm=args.clip_grad_norm,
            max_sequence_length=args.max_sequence_length,
            fixed_sized_sequences=args.fixed_sized_sequences,
            pack_sequences=args.pack_sequences,
            loss_update_frequency=args.loss_update_frequency,
        )


def _train_one_epoch(
    model: TransformerStack,
    scaler: torch.GradScaler,
    loss_fn: CrossEntropyLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: AdapterTrainingConfiguration,
) -> float:
    """
    Train (/tune) the model on the given data.

    Args:
        model (TransformerStack): The decoder model to train
        scaler (torch.GradScaler): Gradient scaler (used for mixed-precision training)
        loss_fn (CrossEntropyLoss): Loss function.
        dataloader (DataLoader): Data loader to train the model
        optimizer (torch.optim.Optimizer): Optimizer used to update the model
        scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler
        config (AdapterTrainingConfiguration): Fine-tuning configuration

    Returns:
        float: The average loss
    """
    model.train()
    accumulated_loss = 0.0  # Fixed: Initialize to 0.0 instead of None
    processed_batches = 0   # Fixed: Track processed batches
    
    pbar = tqdm(dataloader, desc="Training", leave=True, ascii=" =")
    for batch_idx, batch in enumerate(pbar):
        inputs = batch[BatchKey.INPUT].to(get_device(), non_blocking=True)
        labels = batch[BatchKey.LABEL].to(get_device(), non_blocking=True)
        segment_ids = batch[BatchKey.SEGMENT_ID].to(get_device(), non_blocking=True)

        with autocast(device=get_device(), dtype=config.autocast_dtype):
            logits = model(inputs, segment_ids=segment_ids).logits
            loss = loss_fn(logits, labels)

        train_loss = scaler.scale(loss)

        if config.gradient_accumulation_steps > 1:
            # Normalize loss to account for batch accumulation
            train_loss = train_loss / max(1, config.gradient_accumulation_steps)
            
        train_loss.backward()

        is_performing_update = ((batch_idx + 1) % config.gradient_accumulation_steps == 0) or (
            (batch_idx + 1) == len(dataloader)
        )
        if is_performing_update:
            if config.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.clip_grad_norm,
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        # Fixed: Accumulate loss safely
        with torch.no_grad():
            loss_value = loss.detach().item()
            if not (math.isnan(loss_value) or math.isinf(loss_value)):
                accumulated_loss += loss_value
                processed_batches += 1

        # Periodically update loss to avoid blocking the GPU at each iteration
        if processed_batches > 0 and (batch_idx % config.loss_update_frequency == 0 or (batch_idx + 1) == len(dataloader)):
            average_loss = safe_divide(accumulated_loss, processed_batches, 0.0)
            pbar.set_postfix(loss=average_loss)

    # Fixed: Return safe division result
    if processed_batches > 0:
        return safe_divide(accumulated_loss, processed_batches, 0.0)
    else:
        logger.warning("No batches were processed successfully")
        return 0.0


@torch.inference_mode()
def _evaluate_one_epoch(
    model: TransformerStack,
    loss_fn: CrossEntropyLoss,
    dataloader: DataLoader,
    config: AdapterTrainingConfiguration,
) -> float:
    """
    Evaluate the model on the given data.

    Args:
        model (TransformerStack): The decoder model to train
        loss_fn (CrossEntropyLoss): Loss function.
        dataloader (DataLoader): Data loader to train the model
        config (AdapterTrainingConfiguration): Fine-tuning configuration

    Returns:
        float: The average loss
    """
    model.eval()
    accumulated_loss = 0.0  # Fixed: Initialize to 0.0 instead of None
    processed_batches = 0   # Fixed: Track processed batches
    
    pbar = tqdm(dataloader, desc="Evaluation", leave=True, ascii=" =")
    for batch_idx, batch in enumerate(pbar):
        inputs = batch[BatchKey.INPUT].to(get_device())
        labels = batch[BatchKey.LABEL].to(get_device())
        segment_ids = batch[BatchKey.SEGMENT_ID].to(get_device())
        
        with autocast(device=get_device(), dtype=config.autocast_dtype):
            logits = model(inputs, segment_ids=segment_ids).logits
            loss = loss_fn(logits, labels)

        # Fixed: Accumulate loss safely
        loss_value = loss.item()
        if not (math.isnan(loss_value) or math.isinf(loss_value)):
            accumulated_loss += loss_value
            processed_batches += 1
        
        # Periodically update loss to avoid blocking the GPU at each iteration
        if processed_batches > 0 and (batch_idx % config.loss_update_frequency == 0 or (batch_idx + 1) == len(dataloader)):
            average_loss = safe_divide(accumulated_loss, processed_batches, 0.0)
            pbar.set_postfix(loss=average_loss)

    # Fixed: Return safe division result
    if processed_batches > 0:
        return safe_divide(accumulated_loss, processed_batches, 0.0)
    else:
        logger.warning("No evaluation batches were processed successfully")
        return 0.0


@empty_cache()
def train_adapter(
    train_data: str | Path,
    eval_data: str | Path | None,
    config: AdapterTrainingConfiguration,
    checkpoint_dir: str | Path,
    checkpoint_frequency: int = 1,
    data_transform: Callable[[InstructMessages, int], InstructMessages] | None = None,
) -> None:
    """
    Train (/tune) the LoRA adapters of the 3B Apple foundation language model.

    Args:
        train_data (str | Path): Path to the jsonl file containing the training dataset.
        eval_data: (str | Path | None): Path to the jsonl file containing the evaluation dataset.
        config (AdapterTrainingConfiguration): Fine-tuning configuration.
        checkpoint_dir (str | Path): Path to a directory to save the checkpoints. Checkpoints
            will be saved at the end of each epoch
        checkpoint_frequency (int): The frequency, in epochs, to save a checkpoint.
        data_transform: Callable[[InstructMessages, int], InstructMessages] | None = None:
            Function to transform each data sample.
            See augment_schema_guided_generation in utils.py for example implementation.
    """
    
    logger.info(f"Fine-tuning adapters with configuration: \n{config}")

    train_data = train_data if isinstance(train_data, Path) else Path(train_data)
    if not train_data.exists():
        msg = f"Training data file does not exist: {train_data}"
        raise ValueError(msg)
    if train_data.suffix != ".jsonl":
        msg = f"Expecting `train_data` ({train_data}) to point to a jsonl file"
        raise ValueError(msg)

    if eval_data:
        eval_data = eval_data if isinstance(eval_data, Path) else Path(eval_data)
        if not eval_data.exists():
            msg = f"Evaluation data file does not exist: {eval_data}"
            raise ValueError(msg)
        if eval_data.suffix != ".jsonl":
            msg = f"Expecting `eval_data` ({eval_data}) to point to a jsonl file"
            raise ValueError(msg)

    # Check data type compatibility
    device = get_device()
    if config.precision == "bf16" and not supports_bfloat16(device):
        msg = f"{device} does not support `BFloat16`, set `precision` to either `Float16` or `Float32`"
        raise ValueError(msg)

    # Check support for device compilation
    if config.compile_model and device.type != "cuda":
        logger.warning("Model compilation only supported on CUDA, disabling compilation")
        config.compile_model = False

    # Load model
    logger.info(f"Loading base model on {device} with precision {config.model_dtype}")
    model = load_base_model(dtype=config.model_dtype, device=device)

    checkpoint_saver = AdapterCheckpointSaver(checkpoint_dir=checkpoint_dir, prefix="adapter", model=model)

    # Activation checkpointing
    if config.enable_activation_checkpointing:
        apply_activation_checkpointing(model)

    logger.info(f"Total parameters {sum(param.numel() for param in model.parameters())}")
    logger.info(
        f"Total trainable parameters {sum(param.numel() for param in model.parameters() if param.requires_grad)}"
    )

    if config.compile_model:
        logger.info("Compiling model")
        model = torch.compile(model, mode="reduce-overhead")

    # Load tokenizer
    tokenizer = load_tokenizer()

    # Validate configuration
    if config.fixed_sized_sequences and config.max_sequence_length is None:
        logger.warning("`max_sequence_length` must be set for `fixed_sized_sequences` to be enabled")
        config.fixed_sized_sequences = False

    if config.pack_sequences and config.max_sequence_length is None:
        logger.warning("`max_sequence_length` must be set for `pack_sequences` to be enabled")
        config.pack_sequences = False

    if config.max_sequence_length and not 1 <= config.max_sequence_length < MAX_CONTEXT_LEN:
        msg = f"Context length must be within [1, {MAX_CONTEXT_LEN})."
        raise ValueError(msg)

    # Load data using original create_dataloader
    train_dataloader = create_dataloader(
        data=train_data,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_sequence_length=config.max_sequence_length,
        fixed_sized_sequences=config.fixed_sized_sequences,
        packing=config.pack_sequences,
        shuffle=True,
        data_transform=data_transform,
    )
    
    eval_dataloader = None
    if eval_data:
        eval_dataloader = create_dataloader(
            data=eval_data,
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            max_sequence_length=config.max_sequence_length,
            fixed_sized_sequences=config.fixed_sized_sequences,
            packing=config.pack_sequences,
            shuffle=False,
            data_transform=data_transform,
        )

    # Setup optimizer, learning rate scheduler, and loss function
    optimizer = _create_optimizer(model=model, learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    
    # Calculate training samples safely
    training_samples = max(1, len(train_dataloader) * config.batch_size)
    
    scheduler = _create_learning_rate_scheduler(
        optimizer=optimizer,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        training_samples=training_samples,
        epochs=config.epochs,
        linear_warmup_epochs=config.linear_warmup_epochs,
    )
    
    pad_id = getattr(tokenizer, 'pad_id', 0)
    loss_fn = CrossEntropyLoss(pad_id=pad_id, ignore_id=-100)
    
    if config.compile_model:
        logger.info("Compiling loss function")
        loss_fn = torch.compile(loss_fn)

    scaler = torch.GradScaler(device=device, enabled=config.use_gradient_scaling)
    logger.info(f"Gradient scaling is enabled: {config.use_gradient_scaling}")

    # Training loop
    for epoch in range(config.epochs):
        logger.info(f"Epoch {epoch + 1}/{config.epochs}")
        
        training_loss = _train_one_epoch(
            model=model,
            scaler=scaler,
            loss_fn=loss_fn,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
        )
        
        logger.info(f"Epoch {epoch + 1} training loss: {training_loss:.6f}")

        # Evaluate
        eval_loss = None
        if eval_dataloader:
            eval_loss = _evaluate_one_epoch(
                model=model, 
                loss_fn=loss_fn, 
                dataloader=eval_dataloader, 
                config=config
            )
            logger.info(f"Epoch {epoch + 1} evaluation loss: {eval_loss:.6f}")

        # Save checkpoint
        is_saving_checkpoint = (epoch + 1) % checkpoint_frequency == 0 or (epoch + 1) == config.epochs
        if is_saving_checkpoint:
            checkpoint_saver.save(
                state_dict=model.state_dict(), 
                epoch=epoch + 1, 
                metric=eval_loss if eval_loss is not None else training_loss
            )
            logger.info(f"Checkpoint saved for epoch {epoch + 1}")

    logger.info("Training completed successfully!")


def run_from_cli() -> None:
    """
    Train adapter with enhanced compatibility.

    Example Usage:
    ```
    # Standard training
    python -m examples.train_adapter \
    --train-data train.jsonl \
    --eval-data valid.jsonl \
    --epochs 5 \
    --learning-rate 1e-4 \
    --batch-size 4 \
    --checkpoint-dir /checkpoints/

    # Memory-optimized training for constrained systems
    python -m examples.train_adapter \
    --train-data train.jsonl \
    --epochs 5 \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --max-sequence-length 512 \
    --activation-checkpointing \
    --checkpoint-dir /checkpoints/
    ```

    Data is expected to be in the format:
    ```
    [{"role": "system", "content": "<INSTRUCTION>"}, {"role": "user", "content": "<USER PROMPT>"}, {"role": "assistant", "content": "<ASSISTANT RESPONSE>"}]
    [{"role": "system", "content": "<INSTRUCTION>"}, {"role": "user", "content": "<USER PROMPT>"}, {"role": "assistant", "content": "<ASSISTANT RESPONSE>"}]
    ...
    ```

    Where each line represents a single training sample.
    """
    parser = argparse.ArgumentParser(description="Fine-tune the LoRA adapters of the 3B language model")

    parser.add_argument("--train-data", type=str, help="Path to the training dataset (jsonl)", required=True)
    parser.add_argument("--eval-data", type=str, help="Path to the evaluation dataset (jsonl)", required=False)
    parser.add_argument("--epochs", type=int, help="The number of epochs to train for", default=2)
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        help="The initial step size to update the parameters (default: 1e-3)",
        default=1e-3,
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        help=(
            "The warmup period (in epochs) to gradually increase the learning rate "
            "from an initial value to its target (default: 1)"
        ),
        default=1,
    )
    parser.add_argument("--batch-size", type=int, help="The size of the mini-batch (default: 4)", default=4)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        help=(
            "An integer representing how many steps to accumulate gradients "
            "before updating parameters (default: 1)"
        ),
        default=1,
    )
    parser.add_argument(
        "--activation-checkpointing",
        help=(
            "When set, some activations are recomputed during the backward pass to reduce memory usage at the "
            "expense of compute."
        ),
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--precision",
        choices=["f32", "bf16", "bf16-mixed", "f16-mixed"],
        help="Precision type (default: bf16-mixed)",
        default="bf16-mixed",
    )
    parser.add_argument(
        "--compile-model",
        help=(
            "A Boolean value indicating whether to compile the model and loss function (default: False) "
            "This will be ignored if not supported by the specified device."
        ),
        action="store_true",
        default=False,
    )
    parser.add_argument("--weight-decay", type=float, help="The weight decay coefficient (default: 1e-2)", default=1e-2)
    parser.add_argument(
        "--clip-grad-norm", type=float, help="The maximum norm of the gradients (default: 1.0)", default=1.0
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        help="The maximum sequence length (default: None)",
        default=None,
    )
    parser.add_argument(
        "--fixed-sized-sequences",
        help="A Boolean value indicating whether unpacked sequences should be padded to `max_sequence_length`",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--pack-sequences",
        help="A Boolean value indicating whether to pack multiple sequences together to improve throughput",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--loss-update-frequency",
        help="the frequency (or interval) which the average loss is updated (default: 3)",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, help="Path to a directory to save the checkpoints.", required=True
    )
    parser.add_argument("--checkpoint-frequency", type=int, help="Epoch frequency to save a checkpoint.", default=1)

    args = parser.parse_args()

    train_adapter(
        train_data=args.train_data,
        eval_data=args.eval_data,
        config=AdapterTrainingConfiguration.create_from_args(args),
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_frequency=args.checkpoint_frequency,
    )


if __name__ == "__main__":
    run_from_cli()
