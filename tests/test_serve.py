"""`awembed serve`: the CLI split and the bind address.

Two defects this pins, both measured 2026-10-27:

* the first appliance design wrote ``ENTRYPOINT ["awembed","serve"]`` plus
  ``command: ["serve", ...]``, so the process ran ``awembed serve serve`` and argparse
  exited 2 with every container reporting "restarting";
* ``serve`` bound 127.0.0.1 unconditionally, so another CONTAINER dialing
  ``http://awembed:8310`` was refused even with the port published.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from awembed import cli  # noqa: E402

needs_serve_extra = pytest.mark.skipif(importlib.util.find_spec("fastapi") is None,
                                       reason="awembed[serve] is not installed")


@needs_serve_extra
def test_the_stage_token_is_consumed_once(monkeypatch):
    """`awembed serve --model-dir X` hands `--model-dir X` to serve.main -- never the
    stage token twice."""
    import awembed.serve as serve

    seen = {}

    def fake_main(rest):
        seen["rest"] = rest
        return 0

    monkeypatch.setattr(serve, "main", fake_main)
    assert cli.main(["serve", "--model-dir", "/m"]) == 0
    assert seen["rest"] == ["--model-dir", "/m"]


@needs_serve_extra
def test_host_is_loopback_unless_asked(tmp_path):
    import awembed.serve as serve

    ap = serve.build_parser()
    assert ap.parse_args(["--model-dir", str(tmp_path)]).host == "127.0.0.1"
    assert ap.parse_args(["--model-dir", str(tmp_path), "--host", "0.0.0.0"]).host == \
        "0.0.0.0"
