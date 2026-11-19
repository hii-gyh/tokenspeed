#!/usr/bin/env bash
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

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python"
elif [[ -x "${ROOT}/.venv-wsl/bin/python" ]]; then
  PYTHON="${ROOT}/.venv-wsl/bin/python"
elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
else
  python3 -m venv "${ROOT}/.venv-wsl"
  PYTHON="${ROOT}/.venv-wsl/bin/python"
fi

if ! "${PYTHON}" -c 'import pytest, paramiko' >/dev/null 2>&1; then
  "${PYTHON}" -m pip install -U pip
  "${PYTHON}" -m pip install pytest paramiko
fi

if [[ -z "${TS_SSH_PW:-}" ]] && command -v powershell.exe >/dev/null 2>&1; then
  TS_SSH_PW="$(powershell.exe -NoProfile -Command '[Environment]::GetEnvironmentVariable("TS_SSH_PW", "User")' 2>/dev/null | tr -d '\r' || true)"
  export TS_SSH_PW
fi

if [[ "${1:-}" == "--bootstrap-ssh" ]]; then
  shift
  "${PYTHON}" "${ROOT}/test/ci/npu/bootstrap_v41_ssh.py" "$@"
  exec "${PYTHON}" "${ROOT}/test/ci/npu/run_v41_stage1.py" "$@"
fi

if [[ "${1:-}" == "--cluster2" ]]; then
  shift
  exec "${PYTHON}" "${ROOT}/test/ci/npu/run_v41_cluster.py" "$@"
fi

if [[ "${1:-}" == "--stage2" ]]; then
  shift
  exec "${PYTHON}" "${ROOT}/test/ci/npu/run_v41_stage2.py" "$@"
fi

if [[ "${1:-}" == "--stage1" ]]; then
  shift
  exec "${PYTHON}" "${ROOT}/test/ci/npu/run_v41_stage1.py" "$@"
fi

exec "${PYTHON}" "${ROOT}/test/ci/npu/run_v41_debug.py" "$@"
