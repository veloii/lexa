# Copyright © 2025 Apple Inc.

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from tamm.layers.transformer import TransformerStack
from tamm.tokenizers.afm import AFMTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import BatchKey, create_dataloader
from .messages import InstructMessages
from .utils import (
    DraftModelCheckpointSaver,
    PrecisionConfig,
    PrecisionType,
    apply_activation_checkpointing,
    autocast,
    empty_cache,
    get_device,
    get_logger,
    load_base_model,
    load_draft_model,
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
    linear_warmup_epochs: int = 1,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Creates a learning rate scheduler with a linear warmup.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer whose learning rate will be scheduled
        batch_size (int): Mini-batch size
        gradient_accumulation_steps (int): Number of steps over which gradients should be
            accumulated before an optimization step is performed
        training_samples (int): Total training samples
        linear_warmup_epochs (int): Number of epochs to warmup.

    Returns:
        torch.optim.lr_scheduler.LRScheduler: Learning rate scheduler
    """
    scaled_batch_size = batch_size * max(1, gradient_accumulation_steps)
    steps_per_epochs = max(1, safe_int_divide(training_samples, scaled_batch_size, 1))
    warmup_steps = int(steps_per_epochs * linear_warmup_epochs)

    def lambda_lr(step: int) -> float:
        """Learning rate scheduler with a linear warmup"""
        if step < warmup_steps:
            return safe_divide(float(step), float(max(1, warmup_steps)), 0.0)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)


class InverseKLLoss(torch.nn.Module):
    """Inverse Kullback-Leibler (KL) divergence loss"""

    def __init__(
        self,
        pad_id: int,
        ignore_id: int,
        temperature: float = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
    ) -> None:
        """Create a new instance of `InverseKLLoss`.

        Args:
            pad_id (int): Padding token ID
            ignore_id (int): Ignore token ID
            temperature (float): Controls the sharpness of the distribution. Higher values produce softer
                distributions, while lower values make the distribution sharper.
            reduction (Literal["mean", "sum"]): Specifies how to aggregate the loss:
                - 'mean': Average the loss over valid tokens, then average over the batch.
                - 'sum': Sum the loss over valid tokens, then average over the batch.
        """
        super().__init__()
        self.pad_id = pad_id
        self.ignore_id = ignore_id
        self.temperature = temperature
        self.reduction = reduction

    def _masked_cross_entropy(
        self,
        logits: torch.Tensor,
        target_probs: torch.Tensor,
        ignore_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, cl, vs = logits.shape  # bs: Batch size, cl: Context Length, vs: Vocab Size
        logits = logits.view(-1, vs)  # [B*L, D]
        target_probs = target_probs.view(-1, vs)  # [B*L, D]

        loss = torch.nn.functional.cross_entropy(logits, target_probs, reduction="none")
        loss = loss.view(bs, cl)

        # Apply the mask
        if ignore_mask is not None:
            loss = loss * ignore_mask  # Mask the loss
            num_valid_positions = ignore_mask.sum(dim=-1)  # Count valid positions
        else:
            num_valid_positions = torch.tensor(bs * cl, device=loss.device)

        # Aggregate the loss
        if self.reduction == "mean":
            num_valid_positions = torch.clamp(num_valid_positions, min=1)
            loss = loss.sum(dim=-1) / num_valid_positions
        elif self.reduction == "sum":
            loss = loss.sum(dim=-1)

        # Average over batch and return
        return loss.mean()

    def forward(self, draft_logits: torch.Tensor, target_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Computes the inverse Kullback-Leibler (KL) divergence for the loss using the given
        draft (/student) and target (/teacher) logits.

        Args:
            draft_logits (torch.Tensor): A [batch size, seq length] tensor produced by the
                draft (/student) model.
            target_logits (torch.Tensor): A [batch size, seq length] tensor produced by the
                target (/teacher) model.
            labels (torch.Tensor): A [batch size, seq length] integer tensor of the
                corresponding labels.

        Returns:
            torch.Tensor: The reduced loss for the given batch.
        """
        assert target_logits.shape == draft_logits.shape
        if self.temperature == 1.0:
            draft_probs = torch.nn.functional.softmax(draft_logits, dim=-1)
        else:
            draft_logits /= self.temperature + torch.finfo(draft_logits.dtype).eps
            target_logits /= self.temperature + torch.finfo(target_logits.dtype).eps
            draft_probs = torch.nn.functional.softmax(draft_logits, dim=-1)

        ignore_mask = torch.logical_and(
            labels != self.pad_id,
            labels != self.ignore_id,
        )

        # Compute inverse kl and return
        return self._masked_cross_entropy(target_logits, draft_probs, ignore_mask) - self._masked_cross_entropy(
            draft_logits, draft_probs, ignore_mask
        )


@dataclass
class DraftModelTrainingConfiguration:
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
    """An integer representing how many steps to accumulate gradients before updating parameters"""

    enable_activation_checkpointing: bool = False
    """
    A Boolean value indicating whether to enable activation checkpointing or not (default: False)

    When `True`, some activations are recomputed during the backward pass to reduce memory usage at the
    expense of compute.
    """

    target_precision: PrecisionType = "bf16-mixed"
    """Precision type of the target model (default: bf16-mixed)"""

    compile_target_model: bool = False
    """
    A Boolean value indicating whether to compile the target (/teacher) model (default: False)

    This will be ignored if not supported by the specified device.
    """

    draft_precision: PrecisionType = "bf16-mixed"
    """Precision type of the draft model (default: bf16-mixed)"""

    compile_draft_model: bool = False
    """
    A Boolean value indicating whether to compile the draft (/student) model and loss function (default: False)

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
    def target_model_dtype(self) -> torch.dtype:
        """Target model data-type"""
        return PrecisionConfig(self.target_precision).model_dtype

    @property
    def target_autocast_dtype(self) -> torch.dtype | None:
        """Target model autocast data-type for compute, returning None if autocast should be disabled"""
        if PrecisionConfig(self.target_precision).use_autocast:
            return PrecisionConfig(self.target_precision).compute_dtype
        return None

    @property
    def draft_model_dtype(self) -> torch.dtype:
        """Draft model data-type"""
        return PrecisionConfig(self.draft_precision).model_dtype

    @property
    def draft_autocast_dtype(self) -> torch.dtype | None:
        """Draft model autocast data-type for compute, returning None if autocast should be disabled"""
        if PrecisionConfig(self.draft_precision).use_autocast:
            return PrecisionConfig(self.draft_precision).compute_dtype
        return None

    @property
    def use_gradient_scaling_for_draft_model(self) -> bool:
        """Boolean value indicating whether gradient scaling is enabled or not"""
        return self.draft_precision == "f16"

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> "DraftModelTrainingConfiguration":
        return DraftModelTrainingConfiguration(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            linear_warmup_epochs=args.warmup_epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            enable_activation_checkpointing=args.activation_checkpointing,
            target_precision=args.target_precision,
            compile_target_model=args.compile_target_model,
            draft_precision=args.draft_precision,
            compile_draft_model=args.compile_draft_model,
            weight_decay=args.weight_decay,
            clip_grad_norm=args.clip_grad_norm,
            max_sequence_length=args.max_sequence_length,
            fixed_sized_sequences=args.fixed_sized_sequences,
            pack_sequences=args.pack_sequences,
            loss_update_frequency=args.loss_update_frequency,
        )


def _train_one_epoch(
    draft: TransformerStack,
    target: TransformerStack,
    scaler: torch.GradScaler,
    loss_fn: InverseKLLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: DraftModelTrainingConfiguration,
) -> float:
    """
    Knowledge distillation training for the draft model using inverse Kullback-Leibler (KL)
    divergence for the loss.

    Args:
        draft (TransformerStack): Draft (/student) model being trained
        target (TransformerStack): Target (/teacher) model to distill knowledge from
        scaler (torch.GradScaler): Gradient scaler
        loss_fn (InverseKLLoss): Loss function.
        dataloader (DataLoader): Data loader to train the model
        optimizer (torch.optim.Optimizer): Optimizer used to update the model
        scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler
        config (DraftModelTrainingConfiguration): Fine-tuning configuration

    Returns:
        float: The average loss
    """
    draft.train()
    accumulated_loss = 0.0  # Fixed: Initialize to 0.0 instead of None
    processed_batches = 0   # Fixed: Track processed batches
    
    pbar = tqdm(dataloader, desc="Training", leave=True, ascii=" =")
    for batch_idx, batch in enumerate(pbar):
        inputs = batch[BatchKey.INPUT].to(get_device(), non_blocking=True)
        labels = batch[BatchKey.LABEL].to(get_device(), non_blocking=True)
        segment_ids = batch[BatchKey.SEGMENT_ID].to(get_device(), non_blocking=True)
        
        with autocast(device=get_device(), dtype=config.draft_autocast_dtype):
            draft_logits = draft(inputs, segment_ids=segment_ids).logits
        with torch.no_grad(), autocast(device=get_device(), dtype=config.target_autocast_dtype):
            target_logits = target(inputs, segment_ids=segment_ids).logits
        with autocast(device=get_device(), dtype=config.draft_autocast_dtype):
            loss = loss_fn(draft_logits=draft_logits, target_logits=target_logits.to(draft_logits.dtype), labels=labels)

        train_loss = scaler.scale(loss)

        if config.gradient_accumulation_steps > 1:
            # Normalize loss to account for batch accumulation
            train_loss = train_loss / max(1, config.gradient_accumulation_steps)
        train_loss.backward()

        is_performing_update = ((batch_idx + 1) % config.gradient_accumulation_steps == 0) or (
            batch_idx + 1 == len(dataloader)
        )
        if is_performing_update:
            if config.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    draft.parameters(),
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
    draft: TransformerStack,
    target: TransformerStack,
    loss_fn: InverseKLLoss,
    dataloader: DataLoader,
    config: DraftModelTrainingConfiguration,
) -> float:
    """
    Evaluate the model on the given data.

    Args:
        draft (TransformerStack): Draft (/student) model being trained
        target (TransformerStack): Target (/teacher) model to distill knowledge from
        loss_fn (InverseKLLoss): Loss function.
        dataloader (DataLoader): Data loader to train the model
        config (DraftModelTrainingConfiguration): Fine-tuning configuration

    Returns:
        float: The average loss
    """
    draft.eval()
    accumulated_loss = 0.0  # Fixed: Initialize to 0.0 instead of None
    processed_batches = 0   # Fixed: Track processed batches
    
    pbar = tqdm(dataloader, desc="Evaluation", leave=True, ascii=" =")
    for batch_idx, batch in enumerate(pbar):
        inputs = batch[BatchKey.INPUT].to(get_device())
        labels = batch[BatchKey.LABEL].to(get_device())
        segment_ids = batch[BatchKey.SEGMENT_ID].to(get_device())

        with torch.no_grad(), autocast(device=get_device(), dtype=config.draft_autocast_dtype):
            draft_logits = draft(inputs, segment_ids=segment_ids).logits
        with autocast(device=get_device(), dtype=config.target_autocast_dtype):
            target_logits = target(inputs, segment_ids=segment_ids).logits
        with torch.no_grad(), autocast(device=get_device(), dtype=config.draft_autocast_dtype):
            loss = loss_fn(draft_logits=draft_logits, target_logits=target_logits.to(draft_logits.dtype), labels=labels)

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


def _load_train_eval_dataloaders(
    train_data: str | Path,
    eval_data: str | Path | None,
    tokenizer: AFMTokenizer,
    config: DraftModelTrainingConfiguration,
    data_transform: Callable[[InstructMessages, int], InstructMessages] | None = None,
) -> tuple[DataLoader, DataLoader | None]:
    """
    Returns a tuple containing the dataloader for the training data and, optionally, the dataloader for
    the evaluation data.

    Args:
        train_data (str | Path): Path to the jsonl file containing the training dataset
        eval_data: (str | Path | None): Path to the jsonl file containing the evaluation dataset
        tokenizer (AFMTokenizer): Tokenizer for encoding the samples
        config (DraftModelTrainingConfiguration): Fine-tuning configuration
        data_transform: Callable[[InstructMessages, int], InstructMessages] | None = None:
            Function to transform each data sample.
            See augment_schema_guided_generation in utils.py for example implementation.

    Returns:
        tuple[DataLoader, DataLoader | None]: A tuple containing the data loaders for the training and, optionally,
            evaluation data
    """
    train_data = train_data if isinstance(train_data, Path) else Path(train_data)
    if train_data.suffix != ".jsonl":
        msg = f"Expecting `train_data` ({train_data}) to point to a jsonl file"
        raise ValueError(msg)

    if eval_data:
        eval_data = eval_data if isinstance(eval_data, Path) else Path(eval_data)
        if eval_data.suffix != ".jsonl":
            msg = f"Expecting `eval_data` ({eval_data}) to point to a jsonl file"
            raise ValueError(msg)

    if config.max_sequence_length and not 1 <= config.max_sequence_length < MAX_CONTEXT_LEN:
        msg = f"Context length must be within [1, {MAX_CONTEXT_LEN})."
        raise ValueError(msg)

    # Load data
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

    return (train_dataloader, eval_dataloader)


def _load_models_and_tokenizer(
    checkpoint: str | Path | None,
    config: DraftModelTrainingConfiguration,
) -> tuple[TransformerStack, TransformerStack, AFMTokenizer]:
    """
    Returns a tuple containing the target (/teacher) model, draft (/student) model, and shared tokenizer.

    Args:
        checkpoint (str | Path | None): File path to saved checkpoint to restore the target
            (/teacher) model
        config (DraftModelTrainingConfiguration): Fine-tuning configuration.

    Returns:
        tuple[TransformerStack, TransformerStack, AFMTokenizer]: A tuple containing the loaded
            target (/teacher) model, draft (/student) model, and shared tokenizer
    """
    # Check data type compatibility
    if (config.target_precision == "bf16" or config.draft_precision == "bf16") and not supports_bfloat16(get_device()):
        msg = f"{get_device()} does not support `BFloat16`, set model precision to either `Float16` or `Float32`"
        raise ValueError(msg)

    # Check support for device compilation
    if (config.compile_target_model or config.compile_draft_model) and get_device().type != "cuda":
        msg = "Expecting CUDA for model compilation, please unset `compile_target_model` and `compile_draft_model`"
        raise RuntimeError(msg)

    logger.info(f"Loading target (/teacher) model on {get_device()} with precision {config.target_model_dtype}")
    target = load_base_model(dtype=config.target_model_dtype, device=get_device(), checkpoint_path=checkpoint)
    # Put the target model in eval mode.
    target.eval()
    if config.compile_target_model:
        logger.info("Compiling target model")
        target = torch.compile(target, mode="reduce-overhead")

    logger.info(f"Loading draft (/student) model on {get_device()} with precision {config.draft_model_dtype}")
    draft = load_draft_model(dtype=config.draft_model_dtype, device=get_device())
    if config.compile_draft_model:
        logger.info("Compiling draft model")
        draft = torch.compile(draft, mode="reduce-overhead")

    # Activation checkpointing
    if config.enable_activation_checkpointing:
        apply_activation_checkpointing(draft, checkpoint_every_n=1)

    # Load tokenizer
    tokenizer = load_tokenizer()

    return (target, draft, tokenizer)


@empty_cache()
def train_draft_model(
    checkpoint: str | Path | None,
    train_data: str | Path,
    eval_data: str | Path | None,
    config: DraftModelTrainingConfiguration,
    checkpoint_dir: str | Path,
    checkpoint_frequency: int = 1,
    data_transform: Callable[[InstructMessages, int], InstructMessages] | None = None,
) -> None:
    """
    Trains the draft (or student) model by distilling knowledge from the base (or teacher) model.

    Args:
        checkpoint (str | Path | None): File path to saved checkpoint to restore
            the base (/teacher) model.
        train_data (str | Path): Path to the jsonl file containing the training dataset.
        eval_data: (str | Path | None): Path to the jsonl file containing the evaluation dataset.
        config (DraftModelTrainingConfiguration): Fine-tuning configuration.
        checkpoint_dir (str | Path): Path to a directory to save the checkpoints. Checkpoints
            will be saved at the end of each epoch
        checkpoint_frequency (int): The frequency, in epochs, to save a checkpoint.
        data_transform: Callable[[InstructMessages, int], InstructMessages] | None = None:
            Function to transform each data sample.
            See augment_schema_guided_generation in utils.py for example implementation.
    """
    checkpoint_saver = DraftModelCheckpointSaver(checkpoint_dir=checkpoint_dir, prefix="draft-model")

    target, draft, tokenizer = _load_models_and_tokenizer(checkpoint=checkpoint, config=config)
    train_dataloader, eval_dataloader = _load_train_eval_dataloaders(
        train_data=train_data, eval_data=eval_data, tokenizer=tokenizer, config=config, data_transform=data_transform
    )

    # Setup optimizer, learning rate scheduler, and loss function
    optimizer = _create_optimizer(model=draft, learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    
    # Calculate training samples safely
    training_samples = max(1, len(train_dataloader) * config.batch_size)
    
    scheduler = _create_learning_rate_scheduler(
        optimizer=optimizer,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        training_samples=training_samples,
        linear_warmup_epochs=config.linear_warmup_epochs,
    )

    loss_fn = InverseKLLoss(pad_id=tokenizer.pad_id, ignore_id=-100)
    if config.compile_draft_model:
        logger.info("Compiling loss function")
        loss_fn = torch.compile(loss_fn)

    scaler = torch.GradScaler(device=get_device(), enabled=config.use_gradient_scaling_for_draft_model)
    logger.info(f"Gradient scaling is enabled: {config.use_gradient_scaling_for_draft_model}")

    for epoch in range(config.epochs):
        logger.info(f"Epoch {epoch + 1}/{config.epochs}")
        training_loss = _train_one_epoch(
            draft=draft,
            target=target,
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
                draft=draft, target=target, loss_fn=loss_fn, dataloader=eval_dataloader, config=config
            )
            logger.info(f"Epoch {epoch + 1} evaluation loss: {eval_loss:.6f}")

        # Save checkpoint
        is_saving_checkpoint = (epoch + 1) % checkpoint_frequency == 0 or (epoch + 1) == config.epochs
        if is_saving_checkpoint:
            checkpoint_saver.save(state_dict=draft.state_dict(), epoch=epoch + 1, metric=eval_loss or training_loss)
            logger.info(f"Checkpoint saved for epoch {epoch + 1}")

    logger.info("Training completed successfully!")


def run_from_cli() -> None:
    """
    Train (/tune) the draft model for speculative decoding.

    Example Usage:
    ```
    python -m examples.train_draft_model \
    --checkpoint teacher.pt \
    --train-data train.jsonl \
    --eval-data valid.jsonl \
    --epochs 5 \
    --learning-rate 1e-4 \
    --batch-size 4 \
    --checkpoint-dir /checkpoints/
    ```

    Data is expected to be in the format like:
    ```
    [{"role": "system", "content": "<INSTRUCTION>"}, {"role": "user", "content": "<USER PROMPT>"}, {"role": "assistant", "content": "<ASSISTANT RESPONSE>"}]
    [{"role": "system", "content": "<INSTRUCTION>"}, {"role": "user", "content": "<USER PROMPT>"}, {"role": "assistant", "content": "<ASSISTANT RESPONSE>"}]
    ...
    ```

    Where each line represents a single training sample.
    """  # noqa: E501
    parser = argparse.ArgumentParser(description="Fine-tune the draft model for speculative decoding")

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a checkpoint of a fine-tuned base (/teacher) model",
        required=False,
    )
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
            "before updating parameters (default 1)"
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
    )
    parser.add_argument(
        "--target-precision",
        choices=["f32", "bf16", "bf16-mixed", "f16-mixed"],
        help="Precision type for the target model (default: bf16-mixed)",
        default="bf16-mixed",
    )
    parser.add_argument(
        "--compile-target-model",
        help=(
            "A Boolean value indicating whether to compile the target (/teacher) model (default: False) "
            "This will be ignored if not supported by the specified device."
        ),
        action="store_true",
    )
    parser.add_argument(
        "--draft-precision",
        choices=["f32", "bf16", "bf16-mixed", "f16-mixed"],
        help="Precision type for the draft model (default: bf16-mixed)",
        default="bf16-mixed",
    )
    parser.add_argument(
        "--compile-draft-model",
        help=(
            "A Boolean value indicating whether to compile the draft (/student) model "
            "This will be ignored if not supported by the specified device."
        ),
        action="store_true",
    )
    parser.add_argument("--weight-decay", type=float, help="The weight decay coefficient (default: 1e-2)", default=1e-2)
    parser.add_argument(
        "--clip-grad-norm", type=float, help="The maximum norm of the gradients (default: 1.0)", default=1.0
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        help="The maximum sequence length. Must be set if `pack-sequences` is enabled",
        required=False,
    )
    parser.add_argument(
        "--fixed-sized-sequences",
        help="A Boolean value indicating whether unpacked sequences should be padded to `max_sequence_length`",
        action="store_true",
    )
    parser.add_argument(
        "--pack-sequences",
        help="A Boolean value indicating whether to pack multiple sequences together to improve throughput",
        action="store_true",
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

    train_draft_model(
        checkpoint=args.checkpoint,
        train_data=args.train_data,
        eval_data=args.eval_data,
        config=DraftModelTrainingConfiguration.create_from_args(args),
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_frequency=args.checkpoint_frequency,
    )


if __name__ == "__main__":
    run_from_cli()
