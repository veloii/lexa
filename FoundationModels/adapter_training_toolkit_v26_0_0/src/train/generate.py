# Copyright © 2025 Apple Inc.

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from tamm.generation.token_generators.base import TokenGenerator
from tamm.generation.token_generators.greedy import GreedyNextTokenLogic, GreedyTokenGenerator
from tamm.generation.token_generators.sampling import SamplingNextTokenLogic, SamplingTokenGenerator
from tamm.generation.token_generators.speculative import (
    SpeculativeDecodingTokenGenerator,
    SpeculativeDecodingTokenGeneratorOutput,
)
from tamm.generation.token_generators.utils import KVCacheType
from tamm.layers.transformer import TransformerStack
from tamm.tokenizers.afm import AFMTokenizer

from .data import BatchKey, create_dataloader
from .messages import InstructMessages, Message
from .utils import (
    PrecisionConfig,
    PrecisionType,
    autocast,
    empty_cache,
    get_device,
    get_logger,
    load_base_model,
    load_draft_model,
    load_tokenizer,
    supports_bfloat16,
)

logger = get_logger()

TEMPERATURE_EPSILON = 1e-5


@dataclass
class GenerationConfiguration:
    precision: PrecisionType = "bf16-mixed"
    """
    Precision type (default: bf16-mixed)

    Property is applied to the base and draft model.
    """

    temperature: float = 1.0
    """temperature (float): A scaling factor applied to the predicted logits (default: 1.0)"""

    top_k: int | None = None
    """top_k (int | None): Limit sampling to the top k most probable tokens (default: None)"""

    max_new_tokens: int = 1024
    """
    The maximum number of tokens to generate (default: 1024), which must not exceed 4096,
    including the prompt.
    """

    batch_size: int = 1
    """The number of samples to process at once, only applicable if processing a file (default: 1)"""

    compile_model: bool = False
    """A Boolean value indicating whether to compile the model (default: False)"""

    num_draft_tokens: int = 5
    """
    Number of steps to take before verifying output with the target model, also known as gamma (default: 15)

    Only applicable when running speculative decoding, i.e., when a checkpoint for the draft model
    is provided.
    """

    @property
    def model_dtype(self) -> torch.dtype:
        """Data-type to store the model parameters"""
        return PrecisionConfig(self.precision).model_dtype

    @property
    def compute_dtype(self) -> torch.dtype | None:
        """Data-type for compute"""
        return PrecisionConfig(self.precision).compute_dtype

    @property
    def autocast_dtype(self) -> torch.dtype | None:
        """Autocast data-type for compute, returning `None` if autocast should be disabled"""
        if PrecisionConfig(self.precision).use_autocast:
            return PrecisionConfig(self.precision).compute_dtype
        return None

    def _create_generator(
        self, model: TransformerStack, tokenizer: AFMTokenizer, output_probabilities: bool = False
    ) -> TokenGenerator:
        """
        Returns a new token generator using the given tokenizer and configuration.

        Args:
            model (TransformerStack): The model producing the tokens
            tokenizer (AFMTokenizer): The tokenizer
            output_probabilities (bool): When set to `True`, `generate` returns probability
                distributions corresponding to the generated `token_ids`.

        Returns:
            TokenGenerator: A new token generator.
        """
        temperature = max(self.temperature, 0)
        if temperature <= TEMPERATURE_EPSILON:
            return GreedyTokenGenerator(
                model=model,
                eos_id=[tokenizer.eos_id, tokenizer.eot_id],
                pad_id=tokenizer.pad_id,
                cache_dtype=self.compute_dtype,
                cache_type=KVCacheType.VANILLA,
                output_probabilities=output_probabilities,
            )
        return SamplingTokenGenerator(
            model=model,
            eos_id=[tokenizer.eos_id, tokenizer.eot_id],
            pad_id=tokenizer.pad_id,
            temperature=temperature,
            top_k=self.top_k,
            top_p=None,
            cache_dtype=self.compute_dtype,
            cache_type=KVCacheType.VANILLA,
            output_probabilities=output_probabilities,
        )

    def _create_speculative_generator(
        self, model: TransformerStack, draft: TransformerStack, tokenizer: AFMTokenizer
    ) -> TokenGenerator:
        """
        Returns a new speculative decoding token generator using the given tokenizer and configuration.

        Args:
            model (TransformerStack): The model producing the tokens
            draft (TransformerStack): The draft model used for speculative decoding
            tokenizer (AFMTokenizer): The tokenizer

        Returns:
            TokenGenerator: A new token generator.
        """
        next_token_logic = (
            GreedyNextTokenLogic()
            if self.temperature <= TEMPERATURE_EPSILON
            else SamplingNextTokenLogic(temperature=self.temperature, top_k=self.top_k)
        )
        return SpeculativeDecodingTokenGenerator(
            model=model,
            next_token_logic=next_token_logic,
            draft_generator=self._create_generator(model=draft, tokenizer=tokenizer, output_probabilities=True),
            max_draft_tokens=max(self.num_draft_tokens, 1),
            cache_type=KVCacheType.SPECULATIVE,
            cache_dtype=self.compute_dtype,
        )

    def create_generator(
        self, model: TransformerStack, draft: TransformerStack | None, tokenizer: AFMTokenizer
    ) -> TokenGenerator:
        """
        Returns a new token generator using the given tokenizer and configuration.

        Args:
            model (TransformerStack): The model producing the tokens
            draft (TransformerStack | None): The draft model used for speculative decoding
            tokenizer (AFMTokenizer): The tokenizer

        Returns:
            TokenGenerator: A new token generator.
        """
        if draft:
            return self._create_speculative_generator(model=model, draft=draft, tokenizer=tokenizer)
        return self._create_generator(model=model, tokenizer=tokenizer)

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> "GenerationConfiguration":
        return GenerationConfiguration(
            precision=args.precision,
            temperature=args.temperature,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            compile_model=args.compile_model,
            num_draft_tokens=args.num_draft_tokens,
        )


def _parse_outputs(tokenizer: AFMTokenizer, generated_token_ids: torch.Tensor) -> list[str]:
    """
    Returns the generated response as a list of strings.

    Args:
        tokenizer (AFMTokenizer): Tokenizer for decoding
        generated_token_ids (torch.Tensor): Generated token IDs

    Returns:
        list[str]: List of strings representing the generated content
    """
    parts = generated_token_ids.split(1)
    return [tokenizer.decode(token_ids=part.cpu().flatten().tolist()) for part in parts]


@dataclass
class GenerationOutput:
    """
    Per-sample output encapsulating the generated response, generation time, and
    acceptance ratio if running speculative decoding.
    """

    response: str
    """Response generated by the model"""

    inference_time: float
    """Inference time per-sample (in seconds)"""

    suggested_token_counts: list[int] | None = None

    accepted_token_counts: list[int] | None = None

    acceptance_ratio: float | None = None
    """
    Average ratio of accepted tokens over the sequence when performing speculative decoding,
    otherwise `None`
    """

    def __str__(self) -> str:
        return self.response


@empty_cache()
@torch.inference_mode()
def generate_content(
    inputs: list[InstructMessages],
    config: GenerationConfiguration,
    checkpoint: str | Path | None = None,
    draft_checkpoint: str | Path | None = None,
) -> list[GenerationOutput]:
    """
    Generate and return responses for each of the given inputs.

    Args:
        inputs (list[InstructMessages]): Samples to run inference
        config (GenerationConfiguration): Configuration details for running inference and decoding outputs
            produced by the model
        checkpoint (str | Path | None): File path to serialized checkpoint to restore the model (base or adapter)
        draft_checkpoint (str | Path | None): File path to serialized checkpoint to restore the draft model

    Returns:
        list[GenerationOutput]: Generated responses
    """
    assert inputs
    model = load_base_model(dtype=config.model_dtype, device=get_device(), checkpoint_path=checkpoint)
    model = model.eval()

    if config.compile_model and torch.cuda.is_available():
        logger.info("Compiling model")
        model = torch.compile(model, mode="reduce-overhead")

    # Check data type compatibility
    if config.precision == "bf16" and not supports_bfloat16(get_device()):
        msg = f"{get_device()} does not support `BFloat16`, set `precision` to either `Float16` or `Float32`"
        raise ValueError(msg)

    draft_model = None
    if draft_checkpoint:
        draft_model = load_draft_model(dtype=config.model_dtype, device=get_device())
        draft_model = draft_model.eval()

        sd = torch.load(draft_checkpoint, map_location=get_device(), weights_only=False)
        draft_model.load_state_dict(sd, strict=True)

        if config.compile_model and torch.cuda.is_available():
            logger.info("Compiling draft model")
            draft_model = torch.compile(model, draft_model="reduce-overhead")

    tokenizer = load_tokenizer()

    data_loader = create_dataloader(
        data=inputs,
        tokenizer=tokenizer,
        batch_size=min(config.batch_size, len(inputs)),
        padding_side="left",
    )

    generator = config.create_generator(model=model, draft=draft_model, tokenizer=tokenizer)
    outputs = []

    for batch in data_loader:
        input_ids = batch[BatchKey.INPUT].to(get_device())
        st = time.perf_counter()
        with autocast(device=get_device(), dtype=config.autocast_dtype):
            output = generator.generate(input_ids=input_ids, max_new_tokens=config.max_new_tokens)
        generated_token_ids = output.token_ids
        et = time.perf_counter()
        generated_responses = _parse_outputs(tokenizer=tokenizer, generated_token_ids=generated_token_ids)
        elapsed_time = (et - st) / len(generated_responses)
        if isinstance(output, SpeculativeDecodingTokenGeneratorOutput):
            accepted_mask = output.suggested_token_counts != 0
            accepted_token_counts, suggested_token_counts = (
                output.accepted_token_counts.float(),
                output.suggested_token_counts.float(),
            )
            acceptance_ratios = torch.zeros_like(suggested_token_counts)
            acceptance_ratios[accepted_mask] = (
                accepted_token_counts[accepted_mask] / suggested_token_counts[accepted_mask]
            )

            for i in range(len(generated_responses)):
                response = generated_responses[i]

                per_prompt_suggested = output.suggested_token_counts[i].cpu().tolist()
                per_prompt_accepted = output.accepted_token_counts[i].cpu().tolist()

                if sum(per_prompt_suggested) > 0:
                    acceptance_ratio = sum(per_prompt_accepted) / sum(per_prompt_suggested)
                else:
                    acceptance_ratio = None

                outputs.append(
                    GenerationOutput(
                        response=response,
                        inference_time=elapsed_time,
                        acceptance_ratio=acceptance_ratio,
                        suggested_token_counts=per_prompt_suggested,
                        accepted_token_counts=per_prompt_accepted,
                    )
                )

        else:
            outputs += [GenerationOutput(response, elapsed_time, None) for response in generated_responses]

    return outputs


def _parse_prompt(prompt: str) -> list[InstructMessages]:
    """
    Convert the given prompt into a formatted input.

    Expecting the `prompt` to be one of:
    - Prompt, e.g.
        "What was Apple founded?"
    - Templated prompt, e.g.
        "system<n>A conversation between a user and a helpful assistant.<turn_end> user<n> When was Apple founded?<turn_end> assistant<n>"
    - Path to a jsonl file where each line represents a structured input, e.g.,
        [{"role": "system", "content": "A conversation between a user and a helpful assistant."}, {"role": "user", "content": "When was Apple founded?"}]

    Args:
        prompt (str): Either a user prompt, templated prompt, or path to a jsonl file.

    Returns:
        list[InstructMessages]: Prompt in a structured format.
    """  # noqa: E501
    template_pattern = r"(system<n>(.+)<turn_end> )?user<n> (.+)<turn_end>"

    if prompt.endswith(".jsonl"):
        jsonl_path = Path(prompt)
        if not jsonl_path.exists():
            msg = f"`{jsonl_path}` does not exist"
            raise ValueError(msg)
        with jsonl_path.open("r") as f:
            return [json.loads(line) for line in f]
    elif re.match(template_pattern, prompt):
        (
            _,
            system_message,
            user_message,
        ) = re.match(template_pattern, prompt).groups()
        return [
            [
                Message.from_system(system_message) if system_message else Message.default_system_message(),
                Message.from_user(user_message),
            ]
        ]
    else:
        return [[Message.default_system_message(), Message.from_user(prompt)]]


def run_from_cli() -> None:
    """
    Generate a response conditioned on the given prompt.

    Example Usage:
    ```
    python -m examples.generate \
    --prompt "When was Apple founded?" \
    --temperature 0
    ```
    """
    parser = argparse.ArgumentParser(description="Run inference using the 3B language model")
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help=(
            "Prompt to condition the model for generation. ",
            "The prompt can be a user message, e.g., 'When was Apple founded?', ",
            "formated prompt with the system message (or instruction), ",
            "or path to a jsonl where each line describes a list of instruct messages.",
        ),
        required=True,
    )
    parser.add_argument(
        "--precision",
        choices=["f32", "bf16", "bf16-mixed", "f16-mixed"],
        help=("Precision type"),
        default="bf16-mixed",
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        help="The size of the mini-batch to process in parallel (default: 1)",
        default=1,
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        help=("A scaling factor applied to the predicted logits. " "Greedy decoding is used when set to 0."),
        default=1.0,
    )
    parser.add_argument("--top-k", type=int, help="Limit sampling to the top k most probable tokens", default=50)
    parser.add_argument("-m", "--max-new-tokens", type=int, help="Max generated tokens", default=50)
    parser.add_argument(
        "-c", "--checkpoint", type=str, help="Serialized model checkpoint of the base model or trained adapter"
    )
    parser.add_argument(
        "--compile-model",
        help="When set, the model will be compiled before running inference.",
        action="store_true",
    )
    parser.add_argument("--draft-checkpoint", type=str, help="Serialized model checkpoint of the draft model")
    parser.add_argument(
        "--num_draft_tokens",
        type=int,
        help=(
            "Number of steps to take before verifying output with the target model, also known as gamma (default: 5)",
            "This option is only applicable when running speculative decoding, i.e., when a checkpoint for the draft ",
            "model is provided.",
        ),
        default=5,
    )

    args = parser.parse_args()

    responses = generate_content(
        inputs=_parse_prompt(args.prompt),
        config=GenerationConfiguration.create_from_args(args),
        checkpoint=args.checkpoint,
        draft_checkpoint=args.draft_checkpoint,
    )
    for index, response in enumerate(responses):
        logger.info(f"Response for prompt {index}: {response}")


if __name__ == "__main__":
    run_from_cli()
