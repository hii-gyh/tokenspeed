# DeepSeek V4.1 indexer scores

The native Blackwell DeepGEMM indexer writes BF16 logits and passes BF16 rows
through masking, alignment, candidate gathering, and DeepSelect. Head weights
retain their existing precision. Paged scoring uses the caller's query tile;
replicated-cache decode supplies the complete local query batch, including
speculative verify rows. Owner-local history retains its per-request windows.
This design avoids splitting a decode batch solely because its configured
history capacity is large. Each score tensor uses 2 bytes per query-by-history
entry; paged native logits and their cleaned copy can coexist, so peak storage
includes the score matrices, alignment, and selection scratch.
Canonical replicated-cache prefill scores densely: the runtime batches request
windows, and splits a request whose own logits exceed the budget, so one call's
BF16 logits stay within `TOKENSPEED_PREFILL_INDEX_LOGITS_MB` (10 GiB by default,
`index_logit_rows`); scoring only checks that bound. DeepSelect reads the MQA
view's [0, end) using the padded row stride; dense Full, including
candidate-source Full, does not copy a cleaned matrix and sets invalid pages
inside [0, end) to -inf in place (`mask_invalid_pages`). The sparse candidate
scorer already produces BF16 logits.
Kernel warmup uses the same output dtype as execution and compiles both
independent and request-grouped paged scoring before capture.

Invocation-owned index metadata can also carry per-query request IDs. Paged
Full passes the same IDs to schedule construction and scoring, allowing the
native kernel to reuse history across consecutive queries of one request.
Such rows share a page table and have nondecreasing visible lengths. Empty
rows get distinct scheduling labels so they cannot truncate an active request's
history bound. The runtime groups canonical decode rows; dense prefill ignores
request IDs. Free decode windows use independent rows so reordering queries
cannot truncate the history scored for an earlier row.
Request IDs, tables, and lengths remain unchanged while sharing metadata;
graph replay refreshes their captured producers.

The paged selection cleaner writes its final aligned score layout directly.
It masks invalid pages and initializes the last live tile's prefetch padding;
wholly future tiles are unspecified. Native token selection consumes this
buffer with exclusive row ends, avoiding another alignment copy. Candidate
block reduction similarly initializes only live blocks and their final aligned
segment, and native selection uses the valid block counts. The full-width
cleaner and generic score-preparation helpers retain their existing contracts.

Candidate block maxima remain FP32, with the existing newest-block pinning and
logical-position mapping. Converting BF16 scores to FP32 for this reduction
does not recover precision lost when the producer wrote BF16. Equal BF16
scores may select different valid boundary ties from an FP32-score Top-K.
Correctness checks therefore compare selected score multisets against the
same BF16 input, while model-quality checks must assess the precision change
separately. The Hopper score producer retains FP32 output.

## Ascend FlatKV torch codec

The torch scatter packs value and scale columns into one selected-byte transfer.
Duplicate destinations retain the last active row; inactive rows retain their
read-back payload. Batches of at most 64 rows resolve that winner with pairwise
key comparisons instead of integer `argsort`, which otherwise runs on AiCpu.
Larger batches keep the sort-based implementation to bound comparison scratch.

On Ascend, selected-byte reads use `index_select`: advanced-index reads from a
multi-GiB strided cache can fail during NPU Graph replay. Writes use the native
ND scatter in `_ascend_codec.py`; advanced-index and `index_copy_` writes were
observed not to update the external cache on replay. CANN 9.2.1 silently
corrupted ND updates larger than 4096 bytes, so the helper splits larger
payloads into bounded launches. Decode-sized writes remain a single launch.
Inactive writes use slot zero as a read-modify-write destination and share that
destination's winner key, so they cannot overwrite a valid slot-zero update.
The hardware probe covers SWA, global and index layouts with duplicate and
inactive slots on the real 4.38 GB strided arena. The regression
`test/npu/ops/test_dsv41_ascend_codec.py` checks payloads beyond 4096 bytes in
both eager and graph replay; the generic CPU behavior tests retain the
portable path.
The BF16 direct-attention mirror uses a distinct three-dimensional ND update:
indices select `[page,row]` and each update is one 512-channel BF16 row. Unlike
the flat uint8 byte path, this operator matched advanced-index writes through
2048 rows on CANN 9.2.1 and does not use the 4096-byte splitting workaround.
The same read-back and winner logic preserves inactive and duplicate slots;
strided-arena Graph tests cover 55/512/2048 rows and updated-input replay.
Runtime quantization extracts float bit fields with floor-divide and multiply
rather than integer shifts, whose NPU Graph replay produced different exponents
and packed bytes from eager.
