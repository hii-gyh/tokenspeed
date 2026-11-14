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

"""Host smoke for the Ascend FlatKV torch indexer."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_ROOT = (
    Path(__file__).resolve().parents[2]
    / "tokenspeed-kernel"
    / "python"
    / "tokenspeed_kernel"
    / "ops"
    / "attention"
    / "dsv41"
)


def _load(name: str):
    path = _ROOT / name
    spec = importlib.util.spec_from_file_location(f"dsv41_{name}_host", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # cache_codec is imported by indexer_torch; load it under its expected name.
    if name == "indexer_torch.py":
        codec_spec = importlib.util.spec_from_file_location(
            "tokenspeed_kernel.ops.attention.dsv41.cache_codec_torch",
            _ROOT / "cache_codec_torch.py",
        )
        codec = importlib.util.module_from_spec(codec_spec)
        assert codec_spec.loader is not None
        import sys

        sys.modules[codec_spec.name] = codec
        codec_spec.loader.exec_module(codec)
    spec.loader.exec_module(mod)
    return mod


def test_index_topk_picks_matching_key():
    codec = _load("cache_codec_torch.py")
    indexer = _load("indexer_torch.py")
    cache = torch.zeros(2, 64, 68, dtype=torch.uint8)
    # Put a distinctive key at logical row 3 → page0 row3.
    key = torch.zeros(1, 128, dtype=torch.bfloat16)
    key[0, 0] = 8.0
    codec.cache_scatter_torch(key, cache, torch.tensor([3], dtype=torch.int64), "index")
    # Query aligned with that key; one head.
    q = key.view(1, 1, 128).expand(1, 2, 128).contiguous()
    weights = torch.ones(1, 2, dtype=torch.bfloat16)
    table = torch.tensor([[1, 0]], dtype=torch.int32)  # page id 1 is unused; page 0 → need page id > 0
    # Page table stores page ids; slot 3 lives in page 0. Use page id 1 for column 0
    # by placing the key at slot 64+3 on page 1 instead.
    cache2 = torch.zeros(3, 64, 68, dtype=torch.uint8)
    codec.cache_scatter_torch(key, cache2, torch.tensor([64 + 3], dtype=torch.int64), "index")
    table = torch.tensor([[1]], dtype=torch.int32)
    visible = torch.tensor([8], dtype=torch.int32)
    rows, lens, blocks, block_lens = indexer.index_topk_torch(
        q, weights, cache2, table, visible, None, 4, 0, 8, None
    )
    assert int(lens[0]) >= 1
    assert 3 in rows[0, : int(lens[0])].tolist()
    assert blocks.shape == (1, 0)


def _reference_blocks(rows, lengths, visible, candidate_topk, block_size):
    tokens = rows.shape[0]
    blocks = torch.full((tokens, candidate_topk), -1, dtype=torch.int32)
    block_lens = torch.zeros(tokens, dtype=torch.int32)
    if candidate_topk == 0:
        return blocks, block_lens
    for token in range(tokens):
        seen = []
        for index in range(int(lengths[token])):
            row = int(rows[token, index])
            if row < 0:
                continue
            block = row // block_size
            if block not in seen:
                seen.append(block)
        count = int(visible[token])
        newest = max(0, (count - 1) // block_size)
        if newest not in seen and count > 0:
            seen.append(newest)
        chosen = sorted(seen)[:candidate_topk]
        block_lens[token] = len(chosen)
        if chosen:
            blocks[token, : len(chosen)] = torch.tensor(chosen, dtype=torch.int32)
    return blocks, block_lens


def test_candidate_blocks_match_the_host_loop():
    indexer = _load("indexer_torch.py")
    rows = torch.tensor(
        [
            [3, 11, 3, -1, 19],
            [8, -1, 8, 24, 1],
            [-1, -1, 4, 5, 6],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.int32,
    )
    lengths = torch.tensor([4, 5, 1, 0], dtype=torch.int32)
    visible = torch.tensor([20, 8, 0, 1], dtype=torch.int32)
    got_blocks, got_lens = indexer._candidate_blocks_from_rows(rows, lengths, visible, 2, 8)
    exp_blocks, exp_lens = _reference_blocks(rows, lengths, visible, 2, 8)
    assert torch.equal(got_blocks, exp_blocks)
    assert torch.equal(got_lens, exp_lens)


def test_candidate_mask_covers_only_live_blocks():
    indexer = _load("indexer_torch.py")
    blocks = torch.tensor([[0, -1, 2], [1, 1, -1]], dtype=torch.int32)
    mask = indexer._candidate_mask(blocks, 24, 8)
    expect = torch.zeros(2, 24, dtype=torch.bool)
    expect[0, 0:8] = True
    expect[0, 16:24] = True
    expect[1, 8:16] = True
    assert torch.equal(mask, expect)
