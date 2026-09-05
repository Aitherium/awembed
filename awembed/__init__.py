"""awembed -- Aither World Embed.

Train a small embedding model that knows your corpus: capture a large teacher's
judgement, distill it into a student together with your own labels, quantize with
a fidelity gate, and evaluate on a split that holds out whole directories. Every
stage is a standalone module with a `--self-test` and a gate that refuses an
artifact that is not fit to hand on.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
