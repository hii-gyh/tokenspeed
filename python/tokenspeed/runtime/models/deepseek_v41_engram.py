# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""DeepSeek V4.1 Engram, with caller-owned history and FP8-resident tables.

Integration contract:
* Construct one EngramHashState from the text config and the actual tokenizer.
  Its immutable input_parameters are borrowed by runtime preparation, which
  resolves raw previous-three IDs and computes hashes before model execution.
  -1 denotes a sequence boundary or a nonparticipating (e.g. image) token.
* Input preparation owns these small windows, including pending overlap tokens
  and the current speculative branch. Refresh pointer-stable input buffers for
  eager and graph forwards alike. Models receive prepared hashes and masks;
  never put request tensors in ForwardContext.
* Hidden states, hashes and masks must be replicated within mapping.attn.tp_group.
  GPU-resident tables are row-sharded across attention TP and all-reduced.
  Host tables stay in anonymous or shared host memory. FP8 gathers through UVA.
  INT8 gathers on the CPU table, then dequantizes on the activation device.
  ``shared`` keeps one full copy per node and skips the lookup all-reduce.
  ``sharded`` keeps a host shard per attention-TP rank and retains the all-reduce.
  ``auto`` selects sharded when attention TP > 1 on CUDA, and shared on Ascend
  (one-shot AR lanes are not armed there). Output matches the GPU path.
  Ascend W8A8 checkpoints may ship ``optional/quarot.safetensors``; the gate
  undoes the repeated block-32 ``global_rotation`` on the residual before the
  normed dot, matching vLLM-Ascend.
* Intercept embed.weight/scale BEFORE a generic GPU weight iterator. Open the
  safetensors file on CPU, pass get_slice(name) to embed.load_sharded(), and mark
  the corresponding runtime parameter name loaded. GPU and host-sharded tables
  copy local rows; shared host tables copy every row on the writer rank.
  Do not cast the model wholesale to BF16: table codes/scales must stay bytes.
* wkv uses the caller-selected V4.1 Linear method for checkpoint scale
  expansion, hardware storage and post-load preparation. Engram owns the
  embedding loader and projection aliases, not a second projection loader.
  checkpoint_weight_aliases() maps raw checkpoint names, including
  wkv.scale, without renaming the embedding's per-row embed.scale.
"""

from __future__ import annotations

import atexit
import ctypes
import ctypes.util
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass

import numpy as np
import torch
from sympy import isprime
from tokenizers import Regex, normalizers
from tokenspeed_kernel.ops.embedding import host_gather
from tokenspeed_kernel.ops.metadata.ngram import (
    NGramHashParams,
    engram_hash_reciprocals,
)
from tokenspeed_kernel.platform import current_platform
from torch import nn

from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.distributed.comm_ops import all_reduce, prepare_all_reduce_lane
from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.quantization.ascend import AscendQuantConfig
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.quantization.fp8 import Mxfp8Config
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.env import global_server_args_dict

logger = get_colorful_logger(__name__)

_ENGRAM_EMBED_SUFFIXES = (".engram.embed.weight", ".engram.embed.scale")
_ENGRAM_SCALE_GROUP = 32


def is_engram_embed_checkpoint_name(name: str) -> bool:
    """True for the two huge table tensors that must not be load_file'd."""
    return name.endswith(_ENGRAM_EMBED_SUFFIXES)


def _engram_table_dtypes(
    quant_config: QuantizationConfig | None,
) -> tuple[torch.dtype, torch.dtype]:
    """Weight/scale storage for the Engram embedding table."""
    if isinstance(quant_config, AscendQuantConfig):
        return torch.int8, torch.float32
    return torch.float8_e4m3fn, torch.uint8


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Return raw-ID to normalized-ID lookup and its multiplier-defining size.

    ``tokenizer`` must expose the fast ``backend_tokenizer``; decoding deliberately
    includes special tokens and does not use HF whitespace cleanup.
    """
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_id: dict[str, int] = {}
    lookup = []
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        key = (
            backend.id_to_token(token_id)
            if "\ufffd" in text
            else normalizer.normalize_str(text) or text
        )
        lookup.append(key_to_id.setdefault(key, len(key_to_id)))
    return lookup, len(key_to_id)


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, tokenizer_vocab_size: int
) -> torch.Tensor:
    """Return CPU int64 [layers, lookbacks] odd multipliers, using reference RNG."""
    if not layer_ids or max_ngram_size < 2 or tokenizer_vocab_size < 1:
        raise ValueError(
            "Engram requires layers, at least two lookbacks and a vocabulary"
        )
    bound = max(1, (np.iinfo(np.int64).max // tokenizer_vocab_size) // 2)
    return torch.from_numpy(
        np.stack(
            [
                np.random.default_rng(10007 * layer_id).integers(
                    low=0, high=bound, size=(max_ngram_size,), dtype=np.int64
                )
                * 2
                + 1
                for layer_id in layer_ids
            ]
        )
    )


@dataclass(frozen=True)
class EngramLayout:
    """Prime bucket geometry in checkpoint layer/ngram/head order."""

    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    max_ngram_size: int
    n_heads: int
    head_dim: int
    primes: tuple[tuple[tuple[int, ...], ...], ...]

    @classmethod
    def from_config(cls, config) -> EngramLayout:
        """Build and validate geometry from a V4.1 text config, not its wrapper."""
        layer_ids = tuple(config.engram_layer_ids)
        rows = tuple(config.engram_num_embeddings)
        if (
            not layer_ids
            or len(set(layer_ids)) != len(layer_ids)
            or len(rows) != len(layer_ids)
            or config.engram_max_ngram_size != 4
            or config.engram_n_heads < 1
            or config.engram_vocab_size < 2
            or config.engram_head_dim < 32
            or config.engram_head_dim % 32
        ):
            raise ValueError(
                "Invalid V4.1 Engram geometry (requires max ngram=4 and 32-wide blocks)"
            )
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(config.engram_max_ngram_size - 1):
                current, sizes = config.engram_vocab_size - 1, []
                for _ in range(config.engram_n_heads):
                    current += 1
                    while current in seen or not isprime(current):
                        current += 1
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        expected_rows = tuple(sum(sum(heads) for heads in layer) for layer in primes)
        if rows != expected_rows:
            raise ValueError(
                f"Engram table rows {rows} do not match prime buckets {expected_rows}"
            )
        return cls(
            layer_ids=layer_ids,
            num_embeddings=rows,
            max_ngram_size=config.engram_max_ngram_size,
            n_heads=config.engram_n_heads,
            head_dim=config.engram_head_dim,
            primes=tuple(primes),
        )


class EngramHashState(nn.Module):
    """Immutable tokenizer/hash constants only; no per-request token cache."""

    def __init__(self, config, tokenizer, device: torch.device | str):
        super().__init__()
        self.layout = EngramLayout.from_config(config)
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        if vocab_size != config.engram_compressed_vocab_size:
            raise ValueError(
                f"Engram compressed vocabulary: got {vocab_size}, expected {config.engram_compressed_vocab_size}"
            )
        if not 0 <= config.engram_pad_token_id < len(token_map):
            raise ValueError("Engram pad token is outside the tokenizer vocabulary")
        self.pad_id = token_map[config.engram_pad_token_id]
        host_primes = torch.tensor(self.layout.primes, dtype=torch.int64, device="cpu")
        primes = host_primes.to(device=device)
        flat = primes.flatten(1)
        self.register_buffer("primes", primes, persistent=False)
        self.register_buffer(
            "reciprocals",
            engram_hash_reciprocals(host_primes).to(device=device),
            persistent=False,
        )
        self.register_buffer("offsets", flat.cumsum(-1) - flat, persistent=False)
        self.register_buffer(
            "token_map",
            torch.tensor(token_map, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "multipliers",
            compute_hash_multipliers(self.layout.layer_ids, 4, vocab_size).to(
                device=device
            ),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        previous_token_ids: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Hash raw IDs [...] and lookbacks [..., 3] to [..., layers, hash columns].

        ``token_mask`` is bool [...] (False for padding/images). Past barriers and
        sequence starts are -1 in ``previous_token_ids``. Real IDs must be in the
        tokenizer vocabulary; callers must not pass clamped image placeholder IDs
        without their mask. All tensors live on the constants' device.
        """
        if (
            previous_token_ids.shape != (*input_ids.shape, 3)
            or token_mask.shape != input_ids.shape
        ):
            raise ValueError(
                "Engram IDs, previous-three window and mask shapes disagree"
            )
        if (
            token_mask.dtype != torch.bool
            or input_ids.dtype not in (torch.int32, torch.int64)
            or previous_token_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise TypeError("Engram expects int32/int64 IDs and a bool token mask")
        return self.input_parameters.hash(input_ids, previous_token_ids, token_mask)

    @property
    def input_parameters(self) -> NGramHashParams:
        """Expose the current immutable buffers to executor-owned input preparation."""
        return NGramHashParams(
            self.token_map,
            self.primes,
            self.reciprocals,
            self.offsets,
            self.multipliers,
            self.pad_id,
        )


def _host_table_dir(nbytes: int) -> str:
    """Pick a writable directory with room for one Engram host-table file.

    ``/dev/shm`` is preferred when it is large enough. Docker often caps it
    at 32–64 GiB, which cannot hold a V4.1 table, so fall back to scratch.
    """
    explicit = global_server_args_dict.get("engram_host_table_dir")
    if explicit:
        return explicit
    margin = 1 << 30
    candidates = []
    if os.path.isdir("/dev/shm"):
        candidates.append("/dev/shm")
    for path in ("/scratch", "/tmp", tempfile.gettempdir()):
        if path not in candidates:
            candidates.append(path)
    for directory in candidates:
        if not os.path.isdir(directory) or not os.access(directory, os.W_OK):
            continue
        if shutil.disk_usage(directory).free >= nbytes + margin:
            return directory
    raise OSError(
        f"No filesystem with {nbytes + margin} free bytes for an Engram host table; "
        "pass --engram-host-table-dir"
    )


_HOST_TABLE_JOB_ID: str | None = None
_CREATED_HOST_FILES: list[str] = []


def _host_table_job_id() -> str:
    """Return a per-job id shared by every rank, unique across concurrent jobs."""
    global _HOST_TABLE_JOB_ID
    if _HOST_TABLE_JOB_ID is not None:
        return _HOST_TABLE_JOB_ID
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        payload = [str(os.getpid())]
        torch.distributed.broadcast_object_list(payload, src=0)
        _HOST_TABLE_JOB_ID = payload[0]
    else:
        _HOST_TABLE_JOB_ID = str(os.getpid())
    return _HOST_TABLE_JOB_ID


def _engram_host_table_path(layer_id: int, field: str, nbytes: int) -> str:
    return os.path.join(
        _host_table_dir(nbytes),
        f"ts-engram-{_host_table_job_id()}-L{layer_id}-{field}",
    )


def _is_host_table_writer(mapping: Mapping) -> bool:
    local = os.environ.get("LOCAL_RANK")
    if local is not None:
        return int(local) == 0
    return mapping.rank % max(mapping.nprocs_per_node, 1) == 0


def host_load_barrier(group: tuple[int, ...]) -> None:
    """Wait until every rank has finished a host-side Engram load.

    vLLM-Ascend and SGLang synchronize this copy on the CPU process group.
    ``torch.distributed.barrier()`` with no group uses the NPU default, which
    is HCCL. The first such call creates the communicator, and
    ``HCCL_CONNECT_TIMEOUT`` expires while slower ranks are still reading the
    multi-tens-of-GiB tables. The gloo world group already exists and waits
    for the copy itself.
    """
    if (
        not torch.distributed.is_initialized()
        or torch.distributed.get_world_size() == 1
    ):
        return
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )

    if not pg_manager.has_process_group("gloo", group):
        pg_manager.init_process_group(group, backend="gloo")
    torch.distributed.barrier(group=pg_manager.get_process_group("gloo", group))


def resolve_engram_host_layout(host_table: bool, tp_size: int) -> str:
    """Return ``gpu``, ``shared``, or ``sharded`` from the CLI opt-in and TP."""
    if not host_table:
        return "gpu"
    if tp_size < 1:
        raise ValueError("Engram host layout requires a positive attention TP size")
    requested = global_server_args_dict["engram_host_table_layout"]
    if requested == "auto":
        # Ascend cannot arm the one-shot Engram embed lane; shared skips the
        # TP all-reduce that otherwise falls back to a slow device collective.
        if tp_size > 1 and not current_platform().is_npu:
            return "sharded"
        return "shared"
    if requested in ("shared", "sharded"):
        return requested
    raise ValueError(
        "engram_host_table_layout must be auto, shared, or sharded; "
        f"got {requested!r}"
    )


def load_engram_rotation_block(
    checkpoint_dir: str, hidden_size: int
) -> torch.Tensor | None:
    """Return the 32×32 QuaRot block for Engram gating, or None if unused.

    Ascend W8A8 packs store ``optional/quarot.safetensors`` with a
    ``global_rotation`` that is the same 32×32 block tiled across hidden_size.
    Missing file means the gate runs in the residual basis (BF16 / GPU packs).
    """
    if hidden_size < _ENGRAM_SCALE_GROUP or hidden_size % _ENGRAM_SCALE_GROUP:
        raise ValueError(
            "Engram QuaRot undo requires hidden_size divisible by 32; "
            f"got {hidden_size}"
        )
    rotation_path = os.path.join(checkpoint_dir, "optional", "quarot.safetensors")
    if not os.path.isfile(rotation_path):
        return None
    from safetensors import safe_open

    with safe_open(rotation_path, framework="pt") as checkpoint:
        rotation = checkpoint.get_tensor("global_rotation")
    if rotation.ndim != 2 or rotation.shape[0] != hidden_size:
        raise ValueError(
            "Engram global_rotation must be [hidden_size, hidden_size]; "
            f"got {tuple(rotation.shape)} for hidden_size={hidden_size}"
        )
    block = rotation[:_ENGRAM_SCALE_GROUP, :_ENGRAM_SCALE_GROUP].contiguous().float()
    expected = torch.block_diag(*([block] * (hidden_size // _ENGRAM_SCALE_GROUP)))
    if not torch.equal(rotation.float(), expected):
        raise ValueError("Engram gate requires repeated block-32 global_rotation")
    return block


def _cleanup_host_files() -> None:
    while _CREATED_HOST_FILES:
        path = _CREATED_HOST_FILES.pop()
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


atexit.register(_cleanup_host_files)


def _advise_hugepages(buffer, nbytes: int) -> None:
    """Best-effort Linux THP hint. Shared file maps often ignore it."""
    if sys.platform != "linux" or nbytes <= 0:
        return
    libc_name = ctypes.util.find_library("c")
    if libc_name is None:
        return
    libc = ctypes.CDLL(libc_name)
    madvise = getattr(libc, "madvise", None)
    if madvise is None:
        return
    madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    madvise.restype = ctypes.c_int
    # Linux MADV_HUGEPAGE=14. Failure is ignored: THP may be disabled.
    madvise(ctypes.c_void_p(int(buffer.ctypes.data)), ctypes.c_size_t(nbytes), 14)


def _create_host_file(path: str, nbytes: int) -> None:
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.ftruncate(fd, nbytes)
    finally:
        os.close(fd)
    _CREATED_HOST_FILES.append(path)


def _unlink_host_file(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    try:
        _CREATED_HOST_FILES.remove(path)
    except ValueError:
        pass


def _host_table_nbytes(shape: tuple[int, int], dtype: torch.dtype) -> int:
    return int(shape[0] * shape[1] * torch.empty((), dtype=dtype).element_size())


def _tensor_from_host_bytes(
    buffer, shape: tuple[int, int], dtype: torch.dtype
) -> torch.Tensor:
    tensor = torch.frombuffer(buffer, dtype=torch.uint8)
    if dtype != torch.uint8:
        tensor = tensor.view(dtype)
    return tensor.view(shape)


def _allocate_host_bytes(
    shape: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device | str,
    shared_file: bool,
    path: str,
) -> tuple[torch.Tensor, object | None]:
    """Return a host uint8/fp8 table and a keep-alive for the backing store.

    Shared multi-rank tables map a job-scoped file. Rank-local host shards use
    an anonymous buffer so Linux transparent huge pages can apply.
    """
    device = torch.device(device)
    if device.type == "meta":
        return torch.empty(shape, dtype=dtype, device="meta"), None
    nbytes = _host_table_nbytes(shape, dtype)
    if shared_file:
        mmap = np.memmap(path, dtype=np.uint8, mode="r+", shape=(nbytes,))
        _advise_hugepages(mmap, nbytes)
        return _tensor_from_host_bytes(mmap, shape, dtype), mmap
    buffer = np.empty(nbytes, dtype=np.uint8)
    _advise_hugepages(buffer, nbytes)
    return _tensor_from_host_bytes(buffer, shape, dtype), buffer


def _dequant_fp8_e8m0(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    values = codes.view(torch.uint8).view(torch.float8_e4m3fn).float()
    scale = scales.view(torch.uint8).view(torch.float8_e8m0fnu).float()
    return (
        (values.unflatten(-1, (-1, _ENGRAM_SCALE_GROUP)) * scale.unsqueeze(-1))
        .flatten(-2)
        .to(torch.bfloat16)
    )


def _dequant_int8_group32(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Symmetric INT8 group-32 with FP32 scales (Ascend / modelslim Engram)."""
    decoded = codes.to(torch.float32).unflatten(-1, (-1, _ENGRAM_SCALE_GROUP))
    decoded = decoded * scales.to(torch.float32).unsqueeze(-1)
    return decoded.flatten(-2).to(torch.bfloat16)


def _dequant_engram_rows(
    codes: torch.Tensor, scales: torch.Tensor, weight_dtype: torch.dtype
) -> torch.Tensor:
    if weight_dtype == torch.int8:
        return _dequant_int8_group32(codes, scales)
    return _dequant_fp8_e8m0(codes, scales)


def _host_int8_row_ids(
    indices: torch.Tensor, row_start: int, row_end: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU row ids into a host INT8 Engram table, and which ids were in range.

    Hash ids are int64. Subtracting and masking them on the NPU has no
    kernel; the copy that synchronizes afterward reports ACL 507034. Move
    the ids first and resolve the range on the CPU. Out-of-range positions
    are 0 in the id tensor and False in the mask.
    """
    cpu_ids = indices.detach()
    if cpu_ids.device.type != "cpu":
        cpu_ids = cpu_ids.contiguous().to(device="cpu")
    valid = (cpu_ids >= row_start) & (cpu_ids < row_end)
    shifted = (cpu_ids - row_start).masked_fill(~valid, 0).to(dtype=torch.long)
    return shifted, valid


class RowShardedEngramEmbedding(nn.Module):
    """Engram table codes + group-32 scales, GPU-sharded or host-resident.

    FP8/E8M0 is the native V4.1 checkpoint path. Ascend modelslim W8A8 uses
    INT8 codes with FP32 group-32 scales instead.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        mapping: Mapping,
        device: torch.device | str,
        host_table: bool,
        host_layout: str,
        layer_id: int,
        *,
        weight_dtype: torch.dtype = torch.float8_e4m3fn,
        scale_dtype: torch.dtype = torch.uint8,
    ):
        super().__init__()
        if (
            num_embeddings < mapping.attn.tp_size
            or embedding_dim < _ENGRAM_SCALE_GROUP
            or embedding_dim % _ENGRAM_SCALE_GROUP
        ):
            raise ValueError(
                "Engram embedding requires rows per TP rank and 32-wide scale blocks"
            )
        if (weight_dtype, scale_dtype) not in (
            (torch.float8_e4m3fn, torch.uint8),
            (torch.int8, torch.float32),
        ):
            raise ValueError(
                "Engram tables require FP8/E8M0 or INT8/FP32 group-32 storage"
            )
        if host_table:
            if host_layout not in ("shared", "sharded"):
                raise ValueError(
                    "Host Engram tables require host_layout shared or sharded"
                )
        elif host_layout != "gpu":
            raise ValueError("GPU Engram tables require host_layout='gpu'")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight_dtype = weight_dtype
        self.scale_dtype = scale_dtype
        self.tp_group = mapping.attn.tp_group
        self.host_table = host_table
        self.host_layout = host_layout
        self.layer_id = layer_id
        self._mmap_keep: list[object] = []
        self._host_gpu_registered = False
        row_sharded = (not host_table) or host_layout == "sharded"
        if row_sharded:
            self.part_num_embeddings = (
                num_embeddings + mapping.attn.tp_size - 1
            ) // mapping.attn.tp_size
            self.row_start = mapping.attn.tp_rank * self.part_num_embeddings
            self.row_end = min(
                self.row_start + self.part_num_embeddings, num_embeddings
            )
        else:
            self.part_num_embeddings = num_embeddings
            self.row_start = 0
            self.row_end = num_embeddings
        shared_file = (
            host_layout == "shared"
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        self._shared_writer = (not shared_file) or _is_host_table_writer(mapping)
        if host_table:
            weight_path = ""
            scale_path = ""
            weight_shape = (self.part_num_embeddings, embedding_dim)
            scale_shape = (
                self.part_num_embeddings,
                embedding_dim // _ENGRAM_SCALE_GROUP,
            )
            if shared_file:
                weight_bytes = _host_table_nbytes(weight_shape, weight_dtype)
                scale_bytes = _host_table_nbytes(scale_shape, scale_dtype)
                weight_path = _engram_host_table_path(layer_id, "weight", weight_bytes)
                scale_path = _engram_host_table_path(layer_id, "scale", scale_bytes)
                if self._shared_writer:
                    _create_host_file(weight_path, weight_bytes)
                    _create_host_file(scale_path, scale_bytes)
                host_load_barrier(mapping.world_group)
            weight, weight_mmap = _allocate_host_bytes(
                weight_shape,
                weight_dtype,
                device,
                shared_file,
                weight_path,
            )
            scale, scale_mmap = _allocate_host_bytes(
                scale_shape,
                scale_dtype,
                device,
                shared_file,
                scale_path,
            )
            for keep in (weight_mmap, scale_mmap):
                if keep is not None:
                    self._mmap_keep.append(keep)
            self.weight = nn.Parameter(weight, requires_grad=False)
            self.scale = nn.Parameter(scale, requires_grad=False)
            if shared_file:
                host_load_barrier(mapping.world_group)
                if self._shared_writer:
                    _unlink_host_file(weight_path)
                    _unlink_host_file(scale_path)
        else:
            self.weight = nn.Parameter(
                torch.empty(
                    self.part_num_embeddings,
                    embedding_dim,
                    dtype=weight_dtype,
                    device=device,
                ),
                requires_grad=False,
            )
            self.scale = nn.Parameter(
                torch.empty(
                    self.part_num_embeddings,
                    embedding_dim // _ENGRAM_SCALE_GROUP,
                    dtype=scale_dtype,
                    device=device,
                ),
                requires_grad=False,
            )
        for param in (self.weight, self.scale):
            param.weight_loader = self.weight_loader
            param.engram_row_sharded = row_sharded
            param.engram_host_table = host_table
            param.engram_row_start = self.row_start
            param.engram_row_end = self.row_end
        local_rows = self.row_end - self.row_start
        if torch.device(device).type != "meta":
            self.weight.data[local_rows:].zero_()
            # FP8 unused rows keep exponent 127 (scale 1.0); INT8 uses 1.0 FP32.
            pad = 1.0 if scale_dtype == torch.float32 else 127
            self.scale.data[local_rows:].fill_(pad)
            self._register_host_for_gpu()

    def _register_host_for_gpu(self) -> None:
        """cudaHostRegister once. Gather kernels must not register."""
        if self._host_gpu_registered or not self.host_table:
            return
        if (
            torch.device(self.weight.device).type != "cpu"
            or not torch.cuda.is_available()
        ):
            self._host_gpu_registered = True
            return
        platform = current_platform()
        weight_bytes = self.weight.view(torch.uint8)
        if not weight_bytes.is_pinned():
            platform.register_host_tensor_for_gpu_access(weight_bytes)
        if not self.scale.is_pinned():
            platform.register_host_tensor_for_gpu_access(self.scale)
        self._host_gpu_registered = True

    def _apply(self, fn, recurse=True):
        if self.host_table:
            return self
        return super()._apply(fn, recurse)

    def _copy_rows(
        self, param: nn.Parameter, rows: torch.Tensor, local_start: int
    ) -> None:
        if param is self.scale:
            if self.scale_dtype == torch.float32:
                if rows.dtype != torch.float32:
                    raise TypeError("Engram INT8 tables require FP32 group-32 scales")
            else:
                if rows.dtype not in (torch.uint8, torch.float8_e8m0fnu):
                    raise TypeError(
                        "Engram scales must be E8M0 or raw uint8 exponent bytes"
                    )
                rows = rows.view(torch.uint8)
        elif self.weight_dtype == torch.int8:
            if rows.dtype != torch.int8:
                raise TypeError("Engram INT8 table weights must stay INT8")
        elif rows.dtype != torch.float8_e4m3fn:
            raise TypeError("Engram table weights must stay FP8 E4M3")
        param.data[local_start : local_start + rows.shape[0]].copy_(rows)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """Load CPU full-table views or already-local tensors without dequantization.

        The streaming loader should prefer load_sharded(): this callback cannot
        undo a full-table allocation already made by a generic weight iterator.
        """
        if param is not self.weight and param is not self.scale:
            raise ValueError("Parameter does not belong to this Engram table")
        if loaded_weight.ndim != 2 or loaded_weight.shape[1] != param.shape[1]:
            raise ValueError("Engram table weight has the wrong row width")
        rows = loaded_weight.shape[0]
        local_rows = self.row_end - self.row_start
        if rows == self.num_embeddings:
            if loaded_weight.device.type != "cpu" and len(self.tp_group) > 1:
                raise ValueError(
                    "Do not materialize a full Engram table on GPU; use load_sharded"
                )
            loaded_weight = loaded_weight[self.row_start : self.row_end]
        elif rows in (local_rows, self.part_num_embeddings):
            loaded_weight = loaded_weight[:local_rows]
        else:
            raise ValueError(
                "Engram weight is neither a full table nor this rank's row shard"
            )
        self._copy_rows(param, loaded_weight, 0)

    def load_sharded(self, parameter_name: str, tensor_slice, chunk_rows: int) -> None:
        """Copy table rows from a CPU safetensors get_slice() in bounded chunks.

        GPU and host-sharded tables copy only this rank's rows. Shared host
        tables copy every row on the writer rank; followers map the file.
        Args: parameter_name is 'weight' or 'scale'; tensor_slice is the global
        checkpoint tensor's lazy CPU slice; chunk_rows bounds transient host
        storage. No full tensor, remote shard, or BF16 table is allocated.
        Returns: None; the caller records the corresponding parameter as loaded.
        """
        if parameter_name not in ("weight", "scale") or chunk_rows < 1:
            raise ValueError("Expected weight/scale and a positive row chunk size")
        if self.host_table and not self._shared_writer:
            return
        param = getattr(self, parameter_name)
        if tuple(tensor_slice.get_shape()) != (self.num_embeddings, param.shape[1]):
            raise ValueError("Checkpoint Engram shape does not match the global table")
        for start in range(self.row_start, self.row_end, chunk_rows):
            end = min(start + chunk_rows, self.row_end)
            rows = tensor_slice[start:end, :]
            if rows.device.type != "cpu":
                raise ValueError("Open Engram safetensors on CPU before slicing")
            self._copy_rows(param, rows, start - self.row_start)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Gather [..., hash columns] IDs into BF16 [..., hash columns, head dim].

        All TP peers must pass identical indices. Shared host tables gather the
        full copy and skip the all-reduce. GPU and host-sharded tables expand
        only local rows and reduce. Dequantization uses the same FP8/E8M0
        torch casts in both paths.
        """
        if (
            indices.is_cuda
            and not self.host_table
            and self.weight_dtype == torch.float8_e4m3fn
        ):
            from tokenspeed_kernel.ops.embedding import mxfp8_embedding

            # Each layer's 2D hash slice can be gathered directly; retain the
            # contiguous fallback for arbitrary leading dimensions.
            if indices.ndim != 2:
                indices = indices.contiguous()
            values = mxfp8_embedding(
                self.weight,
                self.scale,
                indices,
                self.row_start,
                self.row_end,
            )
        elif self.host_table and self.weight_dtype == torch.int8:
            # The INT8 table stays on CPU. Resolve ids on the CPU as well:
            # the NPU cannot run the int64 range check.
            table_ids, valid = _host_int8_row_ids(
                indices, self.row_start, self.row_end
            )
            codes = self.weight[table_ids]
            scales = self.scale[table_ids]
            if codes.device != indices.device:
                codes = codes.to(device=indices.device)
                scales = scales.to(device=indices.device)
                valid = valid.to(device=indices.device)
            values = _dequant_engram_rows(codes, scales, self.weight_dtype)
            values = values.masked_fill(~valid.unsqueeze(-1), 0)
        else:
            local = (indices >= self.row_start) & (indices < self.row_end)
            local_ids = (indices - self.row_start).masked_fill(~local, 0)
            if self.host_table and self.weight_dtype != torch.int8:
                codes = host_gather.uint8_row_gather(
                    self.weight.view(torch.uint8), local_ids, None
                )
                scales = host_gather.uint8_row_gather(self.scale, local_ids, None)
                values = _dequant_engram_rows(codes, scales, self.weight_dtype)
            else:
                local_ids = local_ids.long()
                if self.weight_dtype == torch.int8:
                    values = _dequant_engram_rows(
                        self.weight[local_ids],
                        self.scale[local_ids],
                        self.weight_dtype,
                    )
                else:
                    # Byte gathers work on CPU and CUDA, including dtypes without index kernels.
                    values = _dequant_engram_rows(
                        self.weight.view(torch.uint8)[local_ids],
                        self.scale[local_ids],
                        self.weight_dtype,
                    )
            values = values.masked_fill(~local.unsqueeze(-1), 0)
        if len(self.tp_group) > 1 and self.host_layout != "shared":
            # The workspace all-reduce sizes itself on rows x trailing width.
            # Reduce one row per token, [rows, columns * head_dim]: the lane is
            # widened for that at construction (DeepseekV41Engram), so a
            # decode batch stays inside the one-shot window. As
            # [rows * columns, head_dim] the same bytes exceed the window's
            # row capacity and drop to NCCL, which costs a ring latency plus a
            # launch stall on every step.
            flat = values.reshape(-1, values.shape[-2] * values.shape[-1])
            values = all_reduce(
                flat,
                group=self.tp_group,
                backend=None,
                op=torch.distributed.ReduceOp.SUM,
            ).view(values.shape)
        return values


def engram_reduce_lane_width(layout: EngramLayout) -> int:
    """Trailing width of the embedding all-reduce: one row per token.

    The gathered values are ``[tokens, n_hash_cols, head_dim]``; they are
    reduced as ``[tokens, n_hash_cols * head_dim]`` so the workspace
    all-reduce sees one row per token and a decode batch stays inside its
    one-shot row window. This is the lane width to arm for it.
    """
    return (layout.max_ngram_size - 1) * layout.n_heads * layout.head_dim


class DeepseekV41Engram(nn.Module):
    """Reference normalized-dot, signed-sqrt gate over four hyperconnection copies."""

    def __init__(
        self,
        config,
        layer_id: int,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        prefix: str,
        device: torch.device | str,
        host_table: bool,
        host_layout: str,
    ):
        """Construct with the V4.1 runtime quant config selected by the caller.

        prefix is the runtime module path (e.g. model.layers.1.engram). The
        checkpoint's wkv is [hidden_size * (hc_mult + 1), n_hash_cols * head_dim],
        replicated and quantized through the existing linear implementation.
        """
        super().__init__()
        layout = EngramLayout.from_config(config)
        if config.hc_mult != 4:
            raise ValueError("V4.1 Engram requires four hyperconnection copies")
        self.layer_id = layer_id
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.dim = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        self.prefix = prefix
        self.n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        if quant_config is not None:
            ascend = isinstance(quant_config, AscendQuantConfig)
            mxfp8 = (
                isinstance(quant_config, Mxfp8Config)
                and quant_config.is_checkpoint_fp8_serialized
                and quant_config.weight_block_size == [1, 32]
                and quant_config.scale_fmt == "ue8m0"
            )
            if not ascend and not mxfp8:
                raise ValueError(
                    "Engram wkv requires Ascend W8A8 or the V4.1 runtime 1x32 MXFP8 config"
                )
        weight_dtype, scale_dtype = _engram_table_dtypes(quant_config)
        with torch.device(device):
            self.wkv = ReplicatedLinear(
                input_size=self.n_hash_cols * layout.head_dim,
                output_size=self.dim * (self.hc_mult + 1),
                bias=False,
                skip_bias_add=False,
                params_dtype=torch.bfloat16,
                quant_config=quant_config,
                prefix=f"{prefix}.wkv",
            )
            self.q_weight = nn.Parameter(
                torch.ones(self.hc_mult, self.dim, dtype=torch.bfloat16),
                requires_grad=False,
            )
            self.k_weight = nn.Parameter(
                torch.ones(self.hc_mult, self.dim, dtype=torch.bfloat16),
                requires_grad=False,
            )
        if (
            quant_config is not None
            and not isinstance(quant_config, AscendQuantConfig)
            and not isinstance(self.wkv.quant_method, Fp8LinearMethod)
        ):
            raise ValueError(
                "Do not exclude Engram wkv from checkpoint FP8 quantization"
            )
        self.embed = RowShardedEngramEmbedding(
            layout.num_embeddings[self.layer_hash_index],
            layout.head_dim,
            mapping,
            device,
            host_table,
            host_layout,
            layer_id,
            weight_dtype=weight_dtype,
            scale_dtype=scale_dtype,
        )
        self.q_weight.weight_loader = default_weight_loader
        self.k_weight.weight_loader = default_weight_loader
        # Optional QuaRot undo for Ascend W8A8; bind_checkpoint_dir fills this.
        self.rotation_block: torch.Tensor | None = None
        # The embedding reduces [tokens, n_hash_cols * head_dim] across attention
        # TP; widen the one-shot lane to that width so decode batches take the
        # workspace kernel. Collective: every rank builds the same layers.
        self.reduce_lane_armed = False
        if (
            len(mapping.attn.tp_group) > 1
            and host_layout != "shared"
            and torch.distributed.is_initialized()
        ):
            width = engram_reduce_lane_width(layout)
            self.reduce_lane_armed = prepare_all_reduce_lane(
                mapping.attn.tp_group, width
            )
            if not self.reduce_lane_armed:
                logger.warning(
                    f"{prefix!s}: one-shot all-reduce lane of width {width:d} not "
                    "armed; the Engram embedding reduce falls back to the device "
                    "collective"
                )

    def set_rotation_block(self, block: torch.Tensor | None) -> None:
        """Install the 32×32 QuaRot block used to undo checkpoint rotation."""
        if block is None:
            self.rotation_block = None
            return
        if (
            block.ndim != 2
            or block.shape[0] != block.shape[1]
            or block.shape[0] != _ENGRAM_SCALE_GROUP
        ):
            raise ValueError(
                "Engram rotation_block must be [32, 32]; "
                f"got {tuple(block.shape)}"
            )
        self.rotation_block = block.detach().float().contiguous()

    def checkpoint_weight_aliases(self) -> dict[str, str]:
        """Return raw layers.<id>.engram checkpoint names to runtime param paths.

        Use these before the generic '.scale' rename. Embedding params expose
        engram_row_sharded/row_start/row_end attributes for streaming dispatch.
        """
        names = {
            "embed.weight": "embed.weight",
            "embed.scale": "embed.scale",
            "wkv.weight": "wkv.weight",
            "q_weight": "q_weight",
            "k_weight": "k_weight",
        }
        if hasattr(self.wkv, "weight_scale_inv"):
            names["wkv.scale"] = "wkv.weight_scale_inv"
        return {
            f"layers.{self.layer_id}.engram.{source}": f"{self.prefix}.{target}"
            for source, target in names.items()
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        hash_ids: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Inject Engram into [..., 4, hidden_size], returning the same shape/dtype.

        hash_ids contains this layer's [..., n_hash_cols] IDs; token_mask is bool
        [...]. False positions pass through unchanged, including image/pad rows.
        Norms and gates accumulate in FP32, independently for each HC copy.
        When ``rotation_block`` is set, the residual is undone into the original
        basis for the normed-dot only; the value add stays in the residual basis.
        """
        shape = hidden_states.shape[:-2]
        if (
            hidden_states.shape[-2:] != (self.hc_mult, self.dim)
            or hash_ids.shape != (*shape, self.n_hash_cols)
            or token_mask.shape != shape
        ):
            raise ValueError("Engram hidden states, hashes and mask shapes disagree")
        if token_mask.dtype != torch.bool:
            raise TypeError("Engram token mask must be bool")
        if hidden_states.numel() == 0:
            return hidden_states
        embeddings = self.embed(hash_ids).flatten(-2)
        kv, _ = self.wkv(embeddings, block_scale=None, output_dtype=None)
        # Fused CUDA gate has no QuaRot undo; match vLLM and take the torch path
        # whenever the checkpoint declares a rotation block.
        if hidden_states.is_cuda and self.rotation_block is None:
            from tokenspeed_kernel.ops.residual import normalized_dot_gate

            return normalized_dot_gate(
                hidden_states, kv, self.q_weight, self.k_weight, token_mask, self.eps
            )
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        h = hidden_states.float()
        original = h
        if self.rotation_block is not None:
            block = self.rotation_block.to(device=h.device, dtype=torch.float32)
            original = (
                original.unflatten(-1, (-1, block.shape[0])) @ block.T
            ).flatten(-2)
        weight = self.q_weight.float() * self.k_weight.float()
        rstd = torch.rsqrt(original.square().mean(-1) + self.eps) * torch.rsqrt(
            key.square().mean(-1) + self.eps
        )
        dot = (original * weight * key).sum(-1) * rstd * self.dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(
            hidden_states.dtype
        )
