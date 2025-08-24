import logging
import re
import shutil
import struct
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import numpy.typing as npt
import torch
import yaml
from coremltools.libmilstoragepython import _BlobStorageWriter as BlobWriter
from coremltools.models.neural_network.quantization_utils import _get_kmeans_lookup_table_and_weight
from coremltools.optimize._utils import LutParams

from src.export.constants import WEIGHT_SPEC, WEIGHTS_TEMPLATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def camelize_key(snake_str: str) -> str:
    parts = re.split(r"[_\s]+", snake_str)
    return parts[0].lower() + "".join(word.capitalize() for word in parts[1:])


def camelize(word: str | dict) -> str | dict:
    if not isinstance(word, dict):
        return camelize_key(word)
    return {camelize_key(k): v for k, v in word.items()}


@dataclass
class DraftModelSpec:
    """
    num_heads: The number of attention heads.
    hidden_dim: The dimension of the hidden layers and embeddings.
    num_kv_heads: The number of attention heads of key and value
        projections.
    target_dtype: The data type for the draft model's performance
        optimization.
    """

    num_heads: int
    hidden_dim: int
    num_kv_heads: int
    target_dtype: str | None = None

    @staticmethod
    def from_checkpoint(sd: dict[str, torch.Tensor]) -> "DraftModelSpec":
        hidden_dim = sd["embedding.weight"].shape[1]

        k_proj_out_channels = sd["layers.layer_0.attention.qkv_transform.fused_linear.weight"].shape[1] // 2
        per_head_dim = sd["layers.layer_0.attention.qk_norm.key_norm.weight"].shape[0]
        num_kv_heads = k_proj_out_channels // per_head_dim
        num_heads = hidden_dim // per_head_dim

        return DraftModelSpec(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_kv_heads=num_kv_heads,
        )


@dataclass
class AdapterSpec:
    num_heads: int = 16
    num_kv_heads: int = 2
    lora_in_features: int = 2048
    lora_out_features: int = 256
    lora_rank: int = 32


class ModelConverter(ABC):
    """
    The abstract base class for model conversion.
    """

    def __init__(self, ckpt: str | Path) -> None:
        if isinstance(ckpt, dict):
            self.ckpt = ckpt
        else:
            self.ckpt = torch.load(ckpt, map_location=torch.device("cpu"), weights_only=True)

    @staticmethod
    def _permute_qk(w: torch.Tensor, n_heads: int, dim1: int, dim2: int) -> torch.Tensor:
        """
        Permute query or key weight. Convert GPT-J style RoPE to GPT-NeoX style RoPE

        Args:
            w: (torch.Tensor): A tensor of query or key norm weight.
            n_heads: number of heads
            dim1: dimension 0 of result
            dim2: dimension 1 of result

        Returns:
            A tensor of permuted query or key norm weight.
        """
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    @staticmethod
    def _cast(ckpt: dict[str : torch.Tensor], target_dtype: torch.dtype) -> dict:
        new_ckpt = {}
        for k, v in ckpt.items():
            if isinstance(v, torch.Tensor) and v.dtype is not target_dtype:
                new_ckpt[k] = v.to(target_dtype)
            else:
                new_ckpt[k] = v
        return new_ckpt

    @abstractmethod
    def _convert(self) -> None:
        """
        Convert checkpoint and apply transformations for export.
        """


class AdapterConverter(ModelConverter):
    """
    Converter class for adapters.
    """

    def __init__(self, ckpt: str | Path, config: AdapterSpec) -> None:
        super().__init__(ckpt)
        self.adapter_weights: dict[str, torch.Tensor] | None = None
        self.transformed_adapter: dict[str, torch.Tensor] | None = None
        self.config = config

    def _extract_adapter(self, cast_dtype: torch.dtype | None = torch.float16) -> None:
        """
        Extract adapter layers from TAMM checkpoint, move to CPU and convert to float16 NumPy arrays

        Args:
            checkpoint: dict[str, torch.Tensor]: PyToch TAMM state dictionary
            cast_dtype (torch.dtype | None): The PyTorch data type to cast the parameter
                to, or `None` to omit casting (default: `torch.float16`)

        Returns:
            dict[str, np.ndarray]: dictionary of adapter weights as NumPy arrays
        """
        self.adapter_weights = {
            k: v.to(cast_dtype or v.dtype, copy=True) for k, v in self.ckpt.items() if "adapter" in k
        }

    def _convert(self) -> None:
        """
        Apply transformations to positional transformation to query and key weights.

        Args:
            adapter_weights: dict[str, np.ndarray]: untransformed adapter weights

        Returns:
            dict: reshaped state dictionary of adapter weights
        """
        adapter_transformed_sd: dict[str, torch.Tensor] = {}
        for k, v in self.adapter_weights.items():
            transformed_v = v.t()
            if "q_transform" in k:
                if "segment_1" in k:
                    if "b_transpose" in k:
                        transformed_v = self._permute_qk(
                            transformed_v, self.config.num_heads, self.config.lora_in_features, self.config.lora_rank
                        )
                    if len(transformed_v.shape) == 2:
                        transformed_v = transformed_v[:, :, None, None]
                    adapter_transformed_sd[k] = transformed_v
            elif "qkv_transform" in k:
                if "lora_0.b_transpose" in k:
                    transformed_v = self._permute_qk(
                        transformed_v, self.config.num_heads, self.config.lora_in_features, self.config.lora_rank
                    )
                if "lora_1.b_transpose" in k:
                    transformed_v = self._permute_qk(
                        transformed_v, self.config.num_kv_heads, self.config.lora_out_features, self.config.lora_rank
                    )
                adapter_transformed_sd[k] = transformed_v
            else:
                adapter_transformed_sd[k] = transformed_v

        self.transformed_adapter = self._cast(adapter_transformed_sd, torch.float32)

    def save_adapter(self, save_dir: str | Path) -> None:
        """
        Save adapter weights as a binary blob as fp16 for .fmadapter.

        Args:
            checkpoint (dict[str, np.ndarray]): Adapter state dictionary.
            output_directory (Union[str, Path]): directory to save the adapter to.

        Returns:
            None
        """

        save_dir.mkdir(parents=True, exist_ok=True)
        output_file = save_dir / "adapter_weights.bin"

        writer = BlobWriter(output_file.as_posix())
        total_bytes_written = 0

        for v in self.transformed_adapter.values():
            weight = v.detach().cpu().numpy()
            if weight.dtype != np.float16:
                if weight.dtype == np.float32:
                    weight = weight.astype(np.float16)
                else:
                    raise ValueError(f"Unsupported dtype {weight.dtype}." "Only float16 and float32 are supported.")
            layer_size_bytes = weight.nbytes
            total_bytes_written += layer_size_bytes
            weight_uint16 = np.frombuffer(weight.tobytes(), np.uint16)
            writer.write_fp16_data(weight_uint16)
        assert total_bytes_written > 0

    def export_adapter(self, save_dir: str | Path) -> None:
        self._extract_adapter()
        self._convert()
        self.save_adapter(save_dir)


class DraftModelConverter(ModelConverter):
    """
    Converter class for draft models
    """

    def __init__(self, ckpt: str | Path) -> None:
        super().__init__(ckpt)
        self.transformed_sd: dict[str, torch.Tensor] = {}
        self.mapping: dict = {
            "norm.weight": "norm.scale",
            "hidden_transform.linear_0.weight": "linear1_0.kernel",
            "hidden_transform.linear_1.weight": "linear1_1.kernel",
            "output_transform.weight": "linear2.kernel",
        }
        self.converted_ckpt: dict[str, torch.Tensor] = {}
        self.ckpt_spec: DraftModelSpec | None = None
        self.exporter = DraftModelExporter()

    @staticmethod
    def _permute_qk_norm(w: torch.Tensor, head_dim: int) -> torch.Tensor:
        """Permute query or key norm weights from GPT-J style rop to GPT-NeoX style rope.

        Args:
            w (torch.Tensor): A tensor of query or key norm weight.
            head_dim (int): The dimension of head.
        Returns:
            A tensor of permuted query or key norm weight.
        """
        return w.view(head_dim // 2, 2).transpose(0, 1).reshape(head_dim)

    def _normalize_tamm_checkpoint(self) -> None:
        self.ckpt = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in self.ckpt.items()}

    def _convert_attention(self, k_orig: str, v_orig: torch.Tensor) -> None:
        korig_ls = k_orig.split(".")
        layer_no = korig_ls[1].replace("layer_", "")
        attn_name_prefix = f"decoder.transformer.layer{layer_no}.self_attention"

        if korig_ls[3] == "qk_norm":
            if korig_ls[4] == "query_norm":
                k = f"{attn_name_prefix}.attention.scale_query.scale"
                v = self._permute_qk_norm(v_orig, self.ckpt_spec.hidden_dim // self.ckpt_spec.num_heads)
                self.transformed_sd[k] = v
            elif korig_ls[4] == "key_norm":
                k = f"{attn_name_prefix}.attention.scale_key.scale"
                v = self._permute_qk_norm(v_orig, self.ckpt_spec.hidden_dim // self.ckpt_spec.num_heads)
                self.transformed_sd[k] = v
        elif korig_ls[3] == "norm":  # Convert norm in Attention layer
            k = f"{attn_name_prefix}.norm.scale"
            self.transformed_sd[k] = v_orig.to(self.ckpt_spec.target_dtype)
        elif korig_ls[3] == "output_transform":  # Convert output proj in Attention layer
            k = f"{attn_name_prefix}.attention.o_proj.conv.kernel"
            v = v_orig.reshape(v_orig.shape[0], self.ckpt_spec.num_heads, -1)
            self.transformed_sd[k] = v.reshape(v.shape[0], v.shape[1] * v.shape[2])[:, :, None, None].to(
                self.ckpt_spec.target_dtype
            )

        elif korig_ls[3] == "qkv_transform":  # Convert QKV in Attention layer
            kq = f"{attn_name_prefix}.attention.i_proj.qkv_proj.q_proj.kernel"
            kk = f"{attn_name_prefix}.attention.i_proj.qkv_proj.k_proj.kernel"
            kv = f"{attn_name_prefix}.attention.i_proj.qkv_proj.v_proj.kernel"
            sum_num_heads = self.ckpt_spec.num_heads + 2 * self.ckpt_spec.num_kv_heads
            v = v_orig.reshape(sum_num_heads, -1, self.ckpt_spec.hidden_dim).permute(2, 0, 1)
            v_q, v_k, v_v = torch.split(
                v, [self.ckpt_spec.num_heads, self.ckpt_spec.num_kv_heads, self.ckpt_spec.num_kv_heads], -2
            )

            v_q = self._permute_qk(
                w=v_q.permute(1, 2, 0).reshape(-1, self.ckpt_spec.hidden_dim),
                n_heads=self.ckpt_spec.num_heads,
                dim1=self.ckpt_spec.hidden_dim,
                dim2=self.ckpt_spec.hidden_dim,
            )
            v_k = self._permute_qk(
                w=v_k.permute(1, 2, 0).reshape(-1, self.ckpt_spec.hidden_dim),
                n_heads=self.ckpt_spec.num_kv_heads,
                dim1=self.ckpt_spec.num_kv_heads * self.ckpt_spec.hidden_dim // self.ckpt_spec.num_heads,
                dim2=self.ckpt_spec.hidden_dim,
            )
            self.transformed_sd[kq] = v_q[:, :, None, None]
            self.transformed_sd[kk] = v_k[:, :, None, None]

            self.transformed_sd[kv] = (
                v_v.reshape(v_v.shape[0], v_v.shape[1] * v_v.shape[2])[:, :, None, None].transpose(0, 1)
            ).to(self.ckpt_spec.target_dtype)

        elif korig_ls[3] == "kv_quantizer":
            kv_key = "i_proj.qkv_proj.v_proj" if korig_ls[4] == "key_quantizer" else "scale_key.rescale_multiply"
            kk = f"{attn_name_prefix}.attention.{kv_key}._activation._quantizer.range_tracker.{korig_ls[6]}"
            self.transformed_sd[kk] = v
        else:
            err = "Invalid key value in ckpt.attention"
            raise ValueError(err)

    def _convert_ff(self, k_orig: str, v_orig: torch.Tensor) -> None:
        korig_ls = k_orig.split(".")
        layer_no = korig_ls[1].replace("layer_", "")
        attn_name_prefix = f"decoder.transformer.layer{layer_no}.feed_forward."
        k = k_orig.replace(f"layers.layer_{layer_no}.feed_forward.", "")
        k = k.replace("wrapped.", "")
        if len(v_orig.shape) == 2:
            v = v_orig[:, :, None, None]
            self.transformed_sd[attn_name_prefix + self.mapping[k]] = v.to(self.ckpt_spec.target_dtype)
        self.transformed_sd[attn_name_prefix + self.mapping[k]] = v_orig.to(self.ckpt_spec.target_dtype)

    def _convert(self) -> dict[str, torch.Tensor]:
        """
        Converts weights for TAMM model to export friendly format.
        """
        self._normalize_tamm_checkpoint()
        self.ckpt_spec = DraftModelSpec.from_checkpoint(self.ckpt)

        assert self.ckpt_spec.num_kv_heads != self.ckpt_spec.num_heads

        for k_orig, v_orig in self.ckpt.items():
            if "embedding" in k_orig:  # Convert Embedding
                assert torch.all(self.ckpt["embedding.weight"] == self.ckpt["output_transform.weight"])
                self.transformed_sd["decoder.emb.token_emb._embedding"] = v_orig.to(self.ckpt[k_orig].dtype)
            elif k_orig == "output_transform.weight":
                pass

            elif "output_norm.weight" in k_orig:  # Convert output norm
                self.transformed_sd["decoder.output_norm.scale"] = v_orig.to(self.ckpt[k_orig].dtype)

            elif "attention" in k_orig:  # Convert Attention layer
                self._convert_attention(k_orig, v_orig)

            elif "feed_forward" in k_orig:  # Convert FF layer
                self._convert_ff(k_orig, v_orig)

            else:
                err = "Invalid key value in ckpt"
                raise ValueError(err)

        return self._cast(self.transformed_sd, torch.float32)

    def export_draft_model(
        self,
        *,
        output_weights_path: str,
    ) -> None:
        """
        Export a Tamm checkpoint into a MIL model.

        This requires running the `build_weights_mapping.ipynb` notebook ahead of time
        to produce the weights specifications and the weights template.

        Arguments:
            ckpt_path (str): the path to the Torch checkpoint
            weights_spec (dict): the weights specifications (from the notebook)
            weights_template_path (str): path to the weights template (from the notebook)
            output_weights_path (str): output path where the exported weights will be saved
        """
        spec_file = WEIGHT_SPEC
        weight_template = WEIGHTS_TEMPLATE

        with spec_file.open() as spec_file:
            weights_spec = yaml.safe_load(spec_file)

        if not isinstance(weights_spec, dict):
            err = "The weights spec must be a dict (not a path)"
            raise TypeError(err)

        weights_mapping = weights_spec["weights_mapping"]

        ckpt = self._convert()

        # Ensure that the checkpoint is reasonable
        missing_keys = sorted(set(weights_mapping) - set(ckpt))
        unknown_keys = sorted(set(ckpt) - set(weights_mapping))
        if missing_keys:
            logger.critical("Missing keys in torch checkpoint:\n%s", "\n".join(missing_keys))
            raise KeyError("\n".join(missing_keys))
        if unknown_keys:
            logger.warning("Unexpected keys in torch checkpoint:\n%s", "\n".join(missing_keys))

        # Copy and fill-in the weights template
        shutil.copyfile(weight_template, output_weights_path)
        with Path(output_weights_path).open("r+b") as output_weights_file:
            for key, spec in weights_mapping.items():
                if spec["format"] == "fp16":
                    logger.debug("exporting fp16 weights: %s", key)
                    self.exporter._write_blob(
                        output_weights_file,
                        array=np.asarray(ckpt[key], np.float16),
                        blob_spec=spec["blobs"]["values"],
                        expected_shape=spec["shape"],
                    )
                elif spec["format"] == "linear_4bits":
                    logger.debug("exporting 4 bits linear weights: %s", key)
                    q_weights = self.exporter._quantize_linearly(np.asarray(ckpt[key], np.float32))
                    self.exporter._write_blob(
                        output_weights_file,
                        array=q_weights.q_weights,
                        blob_spec=spec["blobs"]["values"],
                        expected_shape=spec["shape"],
                    )
                    self.exporter._write_blob(
                        output_weights_file,
                        array=q_weights.scales.astype(np.float16),
                        blob_spec=spec["blobs"]["scales"],
                        expected_shape=[spec["shape"][-1]],
                    )
                elif spec["format"] == "lut_4bits":
                    logger.debug("exporting 4 bits palettized weights: %s", key)
                    # TODO: We can make this way faster with a ProcessPoolExecutor
                    q_weights = self.exporter._palettize(np.asarray(ckpt[key], np.float64))
                    self.exporter._write_blob(
                        output_weights_file,
                        array=q_weights.lut_indices,
                        blob_spec=spec["blobs"]["lut_indices"],
                        expected_shape=spec["shape"],
                    )
                    self.exporter._write_blob(
                        output_weights_file,
                        array=q_weights.lut_palette.astype(np.float16),
                        blob_spec=spec["blobs"]["lut_palette"],
                    )
                    self.exporter._write_blob(
                        output_weights_file,
                        array=q_weights.scales.astype(np.float16),
                        blob_spec=spec["blobs"]["scales"],
                        expected_shape=[spec["shape"][0]],
                    )
                else:
                    err = f"Weights format not yet supported: {spec['format']}"
                    raise ValueError(err)


@dataclass
class LinearlyQuantizedArray:
    q_weights: npt.NDArray[np.int8]
    scales: npt.NDArray[np.float64]


@dataclass
class PalettizedArray:
    lut_indices: npt.NDArray[np.uint]
    lut_palette: npt.NDArray[np.float64]
    scales: npt.NDArray[np.float64]


@dataclass
class DraftModelExporter:
    @staticmethod
    def _quantize_linearly(weights: np.ndarray, axis: int = 0) -> LinearlyQuantizedArray:
        max_abs_weights = np.max(np.abs(weights), axis=axis, keepdims=True)
        scales = np.maximum(max_abs_weights / 7, 1e-6)

        q = np.round(weights / scales).astype(np.int8)
        q = np.clip(q, -8, 7)

        return LinearlyQuantizedArray(q, scales)

    @staticmethod
    def _grouped_channelwise_compress(
        weight: np.ndarray,
        nbits: int = 4,
        channel_axis: int = 0,
        channel_group_size: int = 16,
        cluster_dim: int = 1,
        vector_axis: int = 0,
    ) -> LutParams:
        """Compress weight into n-bit representation by grouped channelwise palettization."""
        if channel_axis < 0:
            channel_axis += len(weight.shape)
        channel_num = weight.shape[channel_axis]
        if channel_group_size == 0:
            channel_group_size = channel_num
        if channel_num % channel_group_size != 0:
            msg = (
                f"Can't perform palettization: The number of channels at {channel_axis}th axis"
                f"{channel_num}) is not divisible by channel_group_size ({channel_group_size})."
            )
            raise ValueError(msg)
        if channel_group_size % cluster_dim != 0:
            msg = (
                f"Can't perform palettization: The channel_group_size at {channel_axis}th axis "
                f"({channel_group_size}) is not divisible by cluster_dim ({cluster_dim})."
            )
            raise ValueError(msg)
        channel_group_num = channel_num // channel_group_size

        if channel_axis != 0:
            weight = np.swapaxes(weight, 0, channel_axis)
        grouped_channel_data = np.split(weight, channel_group_num, axis=0)
        lut, indices = zip(
            *[
                _get_kmeans_lookup_table_and_weight(
                    nbits, per_channel_group_data, force_kmeans1d=True, cluster_dim=cluster_dim, vector_axis=0
                )
                for per_channel_group_data in grouped_channel_data
            ],
            strict=False,
        )

        lut = np.stack(lut, axis=0)
        indices = np.stack(indices, axis=0)

        palette_num = 2**nbits
        indices_target_shape = list(weight.shape)
        if cluster_dim > 1:
            indices_target_shape[vector_axis] //= cluster_dim
        indices = indices.reshape(indices_target_shape)
        lut_target_shape = [1] * (len(weight.shape) + 2)
        lut_target_shape[0] = channel_group_num
        lut_target_shape[-1] = cluster_dim
        lut_target_shape[-2] = palette_num
        lut = lut.reshape(lut_target_shape)

        if channel_axis != 0:
            lut = np.swapaxes(lut, 0, channel_axis)
            indices = np.swapaxes(indices, 0, channel_axis)

        return LutParams(indices.astype(np.uint8), lut.astype(weight.dtype), channel_axis)

    @staticmethod
    def _squeezed_shape(shape: Sequence[int]) -> Sequence[int]:
        return [n for n in shape if n != 1]

    @staticmethod
    def _pack_uint4(array: np.ndarray) -> bytes:
        assert array.dtype == np.uint8
        assert np.all(array < 16)

        array = np.ravel(array)
        lower, upper = array[0::2], array[1::2]
        packed_bytes = ((upper << 4) | lower).astype(np.uint8)
        return packed_bytes.tobytes()

    @staticmethod
    def _pack_int4(array: np.ndarray) -> bytes:
        assert array.dtype == np.int8

        array = np.ravel(array)
        lower, upper = array[0::2], array[1::2]
        packed_bytes = ((upper << 4) | lower & 0x0F).astype(np.int8)
        return packed_bytes.tobytes()

    @classmethod
    def _palettize(cls, weights: np.ndarray, axis: int = 0) -> PalettizedArray:
        reduction_axes = tuple([x for x in range(len(weights.shape)) if x != axis])
        scales = np.max(np.abs(weights), axis=reduction_axes)
        normalized_weights = weights / np.expand_dims(scales, axis=reduction_axes)
        lut_params = cls._grouped_channelwise_compress(weight=normalized_weights, channel_axis=axis)
        return PalettizedArray(lut_indices=lut_params.indices, lut_palette=lut_params.lut, scales=scales)

    @classmethod
    def _write_blob(
        cls,
        weights_file: BinaryIO,
        array: np.ndarray,
        blob_spec: dict[str, any],
        expected_shape: Sequence[int] | None = None,
    ) -> None:
        weights_file.seek(blob_spec["offset"])
        magic, dtype_code, size_bytes, offset, _ = struct.unpack("=LLQQQ", weights_file.read(32))
        assert magic == 0xDEADBEEF

        if expected_shape is not None and cls._squeezed_shape(array.shape) != cls._squeezed_shape(expected_shape):
            raise ValueError(f"Incompatible shapes: {array.shape!s} <> {expected_shape!s}")

        _mil_to_np_dtypes = {1: np.float16, 2: np.float32, 3: np.uint8, 4: np.int8, 8: np.int8, 11: np.uint8}
        expected_dtype = _mil_to_np_dtypes[dtype_code]
        if expected_dtype != array.dtype:
            logger.warning("Expected %s blob, got %s", expected_dtype.name, array.dtype.name)

        _mil_dtypes_to_bytes_converters = {8: cls._pack_int4, 11: cls._pack_uint4}
        convert_to_bytes = _mil_dtypes_to_bytes_converters.get(dtype_code)
        array_bytes = convert_to_bytes(array) if convert_to_bytes is not None else array.tobytes()
        assert len(array_bytes) == size_bytes, "mismatching array size or dtype"

        weights_file.seek(offset)
        weights_file.write(array_bytes)
