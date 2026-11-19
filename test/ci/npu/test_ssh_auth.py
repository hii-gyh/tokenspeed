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

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from bootstrap_v41_ssh import KEY_COMMENT, install_command  # noqa: E402
from ssh_auth import authentication_kwargs  # noqa: E402


def test_password_is_explicitly_preserved_when_available(tmp_path):
    key_path = tmp_path / "key"
    key_path.write_text("unused", encoding="utf-8")
    assert authentication_kwargs("secret", key_path) == {"password": "secret"}


def test_dedicated_key_is_used_without_password(tmp_path):
    key_path = tmp_path / "key"
    key_path.write_text("unused", encoding="utf-8")
    assert authentication_kwargs(None, key_path) == {"key_filename": str(key_path)}


def test_missing_noninteractive_credentials_fail_before_connecting(tmp_path):
    with pytest.raises(RuntimeError, match="non-interactive SSH credentials"):
        authentication_kwargs(None, tmp_path / "missing")


def test_public_key_install_is_idempotent(tmp_path):
    public_key = f"ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITest {KEY_COMMENT}"
    command = install_command(public_key)
    env = dict(os.environ)
    env["HOME"] = str(tmp_path)
    for _ in range(2):
        result = subprocess.run(
            ["bash", "-c", command],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
    authorized = (tmp_path / ".ssh" / "authorized_keys").read_text(encoding="utf-8")
    assert authorized.splitlines() == [public_key]
