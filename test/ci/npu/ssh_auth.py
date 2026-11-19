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

"""Non-interactive SSH authentication shared by the V4.1 debug commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import paramiko

DEFAULT_KEY_PATH = Path.home() / ".ssh" / "tokenspeed_v41_ed25519"


def authentication_kwargs(
    password: str | None, key_path: Path = DEFAULT_KEY_PATH
) -> dict[str, Any]:
    """Select the password when supplied, otherwise require the dedicated key."""
    if password:
        return {"password": password}
    if key_path.is_file():
        return {"key_filename": str(key_path)}
    raise RuntimeError(
        "non-interactive SSH credentials are unavailable: set TS_SSH_PW or run "
        "the one-time SSH key bootstrap"
    )


def connect_client(
    host: str,
    *,
    user: str,
    password: str | None,
    key_path: Path = DEFAULT_KEY_PATH,
    port: int = 22,
    accept_new_host_key: bool,
) -> paramiko.SSHClient:
    """Connect without interactive prompts or ambient agent credentials."""
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(
        paramiko.AutoAddPolicy() if accept_new_host_key else paramiko.RejectPolicy()
    )
    try:
        client.connect(
            host,
            port=port,
            username=user,
            look_for_keys=False,
            allow_agent=False,
            timeout=20,
            banner_timeout=30,
            auth_timeout=20,
            **authentication_kwargs(password, key_path),
        )
    except Exception:
        client.close()
        raise
    return client
