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

from test.ci.npu.run_v41_benchmark import (
    benchmark_command,
    final_json,
    parse_args,
)

import pytest


def test_benchmark_command_preserves_measurement_identity_and_load():
    args = parse_args(
        [
            "--host",
            "192.0.2.10",
            "--engine",
            "vllm-ascend",
            "--protocol",
            "stream",
            "--topology",
            "two-node test topology",
            "--model",
            "model with spaces",
            "--repeat",
            "3",
            "--source-tree",
            "/root/source tree",
            "--model-path",
            "/mnt/weights/v41",
            "--weight-revision",
            "test-revision",
        ]
    )

    command = benchmark_command(args, "test-run")

    assert "--engine vllm-ascend" in command
    assert "--protocol stream" in command
    assert "--topology 'two-node test topology'" in command
    assert "--prompt-tag test-run" in command
    assert "--model 'model with spaces'" in command
    assert "--repeat 3" in command
    assert "--concurrency 8 --requests 16 --max-tokens 128" in command
    assert "--source-tree '/root/source tree'" in command
    assert "--model-path /mnt/weights/v41" in command
    assert "--weight-revision test-revision" in command
    assert "--server-info-path /get_server_info" in command


def test_final_json_uses_the_complete_result_after_progress_output():
    result = final_json(
        'progress\n{"ok": true, "summary": {"output_tok_per_s_mean": 12.5}}\n'
    )

    assert result["ok"] is True
    assert result["summary"]["output_tok_per_s_mean"] == 12.5


def test_final_json_rejects_missing_or_non_object_results():
    with pytest.raises(RuntimeError, match="no JSON"):
        final_json("request failed\n")
    with pytest.raises(RuntimeError, match="not an object"):
        final_json('[{"ok": true}]\n')
