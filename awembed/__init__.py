"""awembed -- Aither World Embed.

Train a small embedding model that knows your corpus: capture a large teacher's
judgement, distill it into a student together with your own labels, quantize with
a fidelity gate, and evaluate on a split that holds out whole directories. Every
stage is a standalone module with a `--self-test` and a gate that refuses an
artifact that is not fit to hand on.
"""

# From the INSTALLED distribution, never a second literal: the wheel said 0.1.4 while
# this file still said 0.1.1, so `awembed --version` lied about the artifact it was
# running (measured 2026-10-27 in a clean venv). The fallback only fires in a source
# checkout with no metadata, and is the value a release must carry.
try:
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("awembed")
except Exception:  # noqa: BLE001 - not installed (a source tree); the literal stands
    __version__ = "0.1.4"

__all__ = ["__version__"]
