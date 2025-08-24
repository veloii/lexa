# Copyright © 2025 Apple Inc.

import copy
import json
import random
from collections.abc import Callable, Iterable
from enum import Enum
from functools import partial
from pathlib import Path

import torch
from tamm.preprocessors.instruct_lm.afm import AFMChatTemplateV6Preprocessor
from tamm.tokenizers.afm import AFMTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .messages import InstructMessages
from .utils import get_logger

logger = get_logger()


class BatchKey(str, Enum):
    """The batch key for features and labels."""

    INPUT = "input_ids"
    LABEL = "label_ids"
    SEGMENT_ID = "segment_ids"


def _pad_sequence(
    samples: list[torch.Tensor],
    padding_value: int,
    padding_side: str = "right",
    sequence_length: int | None = None,
) -> dict[list[int]]:
    """
    Pads or truncates a list of 1D tensors to a fixed length or maximum length in the list.

    Args:
        samples (list[torch.Tensor]): List of variable lengthed sequences
        padding_value (int): Token ID used for padding
        padding_side (str): The side to pad the sequences on (default: `right`)
        sequence_length (int | None): If set, all sequences will be padded (or truncated)
            to this value, otherwise padded to the length of the longest sequence in the list

    Returns:
        torch.Tensor: Tensor of shape [batch size, padded length]
    """
    if sequence_length:
        padded_samples = []
        for sample in samples:
            padded_sample = sample[:sequence_length]
            padding_length = sequence_length - padded_sample.size(0)
            if padding_length > 0:
                padded_sample = torch.nn.functional.pad(
                    sample, (0, padding_length) if padding_side == "right" else (padding_length, 0), value=padding_value
                )
            padded_samples.append(padded_sample)
        return torch.stack(padded_samples)
    return pad_sequence(samples, batch_first=True, padding_value=padding_value, padding_side=padding_side)


class InstructMessagesDataset(Dataset):
    """
    Dataset for instruct messages.


    """

    def __init__(
        self,
        data: Iterable[InstructMessages],
        preprocessor: AFMChatTemplateV6Preprocessor,
        data_transform: Callable[[InstructMessages, int], InstructMessages] | None = None,
    ) -> None:
        """
        Create an instance of `InstructMessagesDataset`.

        Data is expected to be in the format:
        ```
        [{"role": "system", "content": "<INSTRUCTION>"}, {"role": "user", "content": "<USER PROMPT>"}, {"role": "assistant", "content": "<ASSISTANT RESPONSE>"}]
        [{"role": "system", "content": "<INSTRUCTION>"}, {"role": "user", "content": "<USER PROMPT>"}, {"role": "assistant", "content": "<ASSISTANT RESPONSE>"}]
        ...
        ```

        Args:
            data (Iterable[InstructMessages]): The data samples.
            preprocessor (AFMChatTemplateV6Preprocessor): The sample processor.
            data_transform: Callable[[InstructMessages, int], InstructMessages] | None = None: Function to transform each data sample.
                See augment_schema_guided_generation in utils.py for example implementation.
        """  # noqa: E501
        self.data = copy.deepcopy(data)
        self.preprocessor = preprocessor
        self.data_transform = data_transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        sample = copy.deepcopy(self.data[index])
        if self.data_transform:
            sample = self.data_transform(sample, index)
        sample = self.preprocessor(sample)
        item = {BatchKey.INPUT: sample.input_ids}
        if sample.label_ids:
            item[BatchKey.LABEL] = sample.label_ids
        return item


class PackedInstructMessagesDataset(Dataset):
    """
    A dataset for instruct messages that combines multiple samples into
    a single sequence.

    The dataset is packed on initialization therefore should fit
    in-memory
    """

    def __init__(self, dataset: InstructMessagesDataset, max_sequence_length: int) -> None:
        """
        Create a new instance of `PackedInstructMessagesDataset`

        Args:
            dataset (InstructMessagesDataset): The dataset to pack.
            max_sequence_length (int): The maximum sequence length.
        """
        self.data = PackedInstructMessagesDataset.pack_sequences(dataset, max_sequence_length)

    @staticmethod
    def pack_sequences(
        dataset: InstructMessagesDataset, max_sequence_length: int
    ) -> dict[str, torch.Tensor | list[int]]:
        """
        Returns a packed representation of the given dataset up to `max_sequence_length`.

        Args:
            dataset (InstructMessagesDataset): The dataset to pack.
            max_sequence_length (int): The maximum sequence length. Must be set
                if `packing` is enabled.
        Returns:
            dict[str, list[int]]: The packed samples.
        """

        data = []

        packed_sample = {
            BatchKey.INPUT: [],
            BatchKey.LABEL: [],
        }

        for sample in dataset:
            input, label = sample[BatchKey.INPUT], sample[BatchKey.LABEL]
            seq_len = len(input)

            if seq_len > max_sequence_length:
                msg = (
                    f"Encountered sequence of length {seq_len} which "
                    f"is greater than the specified max sequence legnth {max_sequence_length}."
                )
                logger.warning(msg)
                input, label = input[:max_sequence_length], label[:max_sequence_length]
            if len(packed_sample[BatchKey.INPUT]) + seq_len >= max_sequence_length:
                data.append(packed_sample)
                packed_sample = {BatchKey.INPUT: [], BatchKey.LABEL: []}

            packed_sample[BatchKey.INPUT] += input
            packed_sample[BatchKey.LABEL] += label

        if len(packed_sample[BatchKey.INPUT]) > 0:
            data.append(packed_sample)

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.data[index]


def _validate_training_sample(sample: InstructMessages) -> None:
    """
    Validates a single training sample.

    Args:
        sample (InstructMessages): A list of messages to validate.

    Raises:
        ValueError: If the sample is invalid.
    """
    if not isinstance(sample, list):
        raise TypeError(f"Excepting `list` but got {type(sample)}")
    if len(sample) < 2:
        raise TypeError(f"A sample must contain at least two messages, but got {len(sample)}")
    last_turn = ""
    for i, message in enumerate(sample):
        if message["role"] == "system" and i > 0:
            msg = f"System message must be the message message but found at index {i}"
            raise ValueError(msg)
        if message["role"] == "assistant" and last_turn != "user":
            msg = "Assistant message must be preceded by user message"
            raise ValueError(msg)
        if message["role"] == last_turn:
            msg = f"Two consecutive messages at index {i} and {i - 1} in messages"
            raise ValueError(msg)
        last_turn = message["role"]

    if sample[-1]["role"] != "assistant":
        msg = "Expecting last message to originate from the assistant"
        raise ValueError(msg)


def _validate_inference_sample(sample: InstructMessages) -> None:
    """
    Validates a single inference sample.

    Args:
        sample (InstructMessages): A list of messages to validate.

    Raises:
        ValueError: If the sample is invalid.
    """
    if not isinstance(sample, list):
        raise TypeError(f"Excepting `list` but got {type(sample)}")
    if len(sample) < 1:
        raise TypeError(f"A sample must contain at least one messages, but got {len(sample)}")
    last_turn = ""
    for i, message in enumerate(sample):
        if message["role"] == "system" and i > 0:
            msg = f"System message must be the message message but found at index {i}"
            raise ValueError(msg)
        if message["role"] == last_turn:
            msg = f"Two consecutive messages at index {i} and {i - 1} in messages"
            raise ValueError(msg)
        last_turn = message["role"]

    if sample[-1]["role"] != "user":
        msg = "Expecting last message to originate from the user"
        raise ValueError(msg)


def _collate_batch(
    batch: list[dict[str, list[int]]],
    pad_id: int,
    eos_id: int,
    ignored_id: int | None = None,
    padding_side: str = "right",
    sequence_length: int | None = None,
) -> dict[str, torch.Tensor]:
    """
    Collate a list of samples into a single batch.

    Samples are padded to the longest sequence length with segment IDs computed for each sequence in the
    batch, i.e., for the input:
        [[2, 0, 2, 3, 0, 0],
         [2, 2, 1, 4, 0, 5],
         [0, 0, 0, 0, 0, 0]],
    The corresponding segment IDs would be:
        [[1, 1, 1, 1, 0, 0],
         [1, 1, 2, 2, 2, 2],
         [0, 0, 0, 0, 0, 0]].

    Args:
        batch (list[dict[str, list[int]]]): A list of dictionaries containing input and label pairs.
        pad_id (int): Token ID for padding.
        eos_id (int): Token ID for end-of-sequence (EOS)
        ignored_id (int | None): Token ID indicating elements to be ignored (padding value for labels),
            `pad_id` will be used if `None` (default: `None`)
        padding_side (str): The side to pad the sequences on (default: `right`)
        sequence_length (int | None): If set, sequences are padded (or truncated) to the specified value, otherwise
            sequences are padded to the sequence with the longest length (default: `None`)

    Returns:
        Dict[str, torch.Tensor]: Collated input, label, and segment id tensors.
    """
    if batch is None or len(batch) == 0:
        msg = "batch must be non-empty"
        raise ValueError(msg)

    input_ids = _pad_sequence(
        [torch.Tensor(x[BatchKey.INPUT]) for x in batch],
        padding_value=pad_id,
        padding_side=padding_side,
        sequence_length=sequence_length,
    ).to(torch.int64)

    segment_ids = 1 + (input_ids == eos_id).cumsum(dim=1, dtype=torch.int64)
    padding_mask = input_ids.ne(pad_id)
    segment_ids = (padding_mask * segment_ids).to(input_ids.dtype)

    collated_batch = {
        BatchKey.INPUT: input_ids,
        BatchKey.SEGMENT_ID: segment_ids,
    }

    if BatchKey.LABEL in batch[0]:
        collated_batch[BatchKey.LABEL] = _pad_sequence(
            [torch.Tensor(x[BatchKey.LABEL]) for x in batch],
            padding_value=ignored_id or pad_id,
            padding_side=padding_side,
            sequence_length=sequence_length,
        ).to(torch.int64)

    return collated_batch


def create_dataloader(
    data: str | Path | list[InstructMessages],
    tokenizer: AFMTokenizer,
    batch_size: int,
    padding_side: str = "right",
    max_sequence_length: int | None = None,
    fixed_sized_sequences: bool = False,
    packing: bool = False,
    shuffle: bool = False,
    data_transform: Callable[[InstructMessages, int], InstructMessages] | None = None,
) -> DataLoader:
    """
    Returns a dataloader for the given data.

    Args:
        data (str | Path | list[Turn]): Either a Path to a jsonl file or list of messages.
        tokenizer (AFMTokenizer): The tokenizer.
        batch_size (int): The size of the mini-batch.
        padding_side (str): The side to pad the sequences on (default: `right`)
        max_sequence_length (int | None): The maximum sequence length. Must be set if `packing`
            is enabled (default: `None`).
        fixed_sized_sequences (bool): A Boolean value indicating whether unpacked sequences should be
            padded to `max_sequence_length` or not (default: `False`).
        packing (bool): A Boolean value indicating whether short sequences are packed together (default: `False`).
        shuffle (bool): A Boolean value indicating whether the samples should be shuffled
            or not (default: `False`).
        data_transform: Callable[[InstructMessages, int], InstructMessages] | None = None:
            Transform function with signature (messages, sample index) -> messages.
            Sample index provides sample position for sample-specific processing, can be ignored if not sample-specific.
            See augment_schema_guided_generation in utils.py for example implementation.

    Returns:
        DataLoader: A dataloader.
    """
    if isinstance(data, str | Path):
        with Path.open(data, encoding="utf8") as f:
            data = [json.loads(line) for line in f]

    if len(data) < batch_size:
        msg = f"""
            batch size of {batch_size} is greater than the number \
            of samples {len(data)}
            """
        raise ValueError(msg)

    # Extract list from JSON object, i.e., transform: {"messages": [...]} to [...]
    data = [sample[next(iter(sample.keys()))] if isinstance(sample, dict) else sample for sample in data]

    # Generate random index to query sample for verification
    rnd_idx = random.randint(0, len(data) - 1)  # noqa: S311

    # Assume to be training if the response is found in a sample.
    is_training = data[rnd_idx][-1]["role"] == "assistant"

    # Check if the first message of the random sample is a system message
    # (assume all samples are consistent in structure).
    contains_system_message = data[rnd_idx][0]["role"] == "system"

    # Validate a single sample at the random index
    if is_training:
        _validate_training_sample(data[rnd_idx])
    else:
        _validate_inference_sample(data[rnd_idx])

    preprocessor = AFMChatTemplateV6Preprocessor(
        tokenizer_spec=tokenizer,
        generate_prefix=not is_training,
        max_sequence_length=max_sequence_length,
        prepend_system_message=not contains_system_message,
    )

    dataset = InstructMessagesDataset(data, preprocessor, data_transform)
    if packing and max_sequence_length:
        dataset = PackedInstructMessagesDataset(dataset, max_sequence_length)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=partial(
            _collate_batch,
            pad_id=tokenizer.pad_id,
            eos_id=tokenizer.eos_id,
            ignored_id=-100,
            padding_side=padding_side,
            sequence_length=max_sequence_length if (fixed_sized_sequences or packing) else None,
        ),
    )
