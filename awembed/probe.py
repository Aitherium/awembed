#!/usr/bin/env python3
"""Probe the NV-Embed-v2 teacher on THIS box: does it load, at what footprint, and
does one forward at the capture envelope succeed. Prints measured `R:` lines.

Exit 0 = loaded and one forward produced finite embeddings; 1 = it did not; 2 =
could not judge (no CUDA, no weights, an import error before the load started).

Why this exists (2026-09-02): on a rented A10 the teacher died at load three
relaunches in a row for three different reasons, and the SAME hand-typed ssh probe
was re-derived eight times to find them -- missing remote-code deps, a backbone
that loads fp32 whatever dtype is requested (27.87 GiB for a 14.62 GiB checkpoint),
and a transformers Cache API break that only shows on the first forward. Every one
of those is invisible to a static read; this is the one command that asks the box.

Run it through the driver: `embed_train.py --corpus <c> --attach <id> --probe`, or on
the box with the teacher venv's python: `python k3_teacher_probe.py`.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import struct
import sys
import time

MODEL_ID = "nvidia/NV-Embed-v2"
REVISION = os.environ.get("NV_EMBED_REVISION", "3fa59658547db50a1e8e3346cf057fd0c77ed6ef")
DTYPE_NAME = os.environ.get("NV_EMBED_DTYPE", "float16")
MAX_LEN = int(os.environ.get("NV_EMBED_MAX_LEN", "1024"))
BATCH = int(os.environ.get("NV_EMBED_BATCH", "2"))
INSTRUCTION = "Instruct: Given a question, retrieve passages that answer the question\nQuery: "


def _r(msg: str) -> None:
    print(f"R: {msg}", flush=True)


def checkpoint_dtypes() -> dict[str, float]:
    """GiB per stored dtype, read from the safetensors headers (no GPU, no load)."""
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    pat = os.path.join(hf_home, "hub", "models--nvidia--NV-Embed-v2", "snapshots", "*",
                       "*.safetensors")
    size = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8}
    gib: collections.Counter = collections.Counter()
    for f in glob.glob(pat):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            numel = 1
            for d in v["shape"]:
                numel *= d
            gib[v["dtype"]] += numel * size.get(v["dtype"], 4) / 2**30
    return {k: round(v, 2) for k, v in gib.items()}


def _self_test() -> int:
    """Prove the header reader (the part that needs no GPU) still measures: a synthetic
    safetensors header with one F16 and one F32 tensor must come back as the right GiB
    per dtype, and an empty cache must read as 'not cached', never as zero bytes."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        hub = os.path.join(td, "hub", "models--nvidia--NV-Embed-v2", "snapshots", "abc")
        os.makedirs(hub)
        header = {"a": {"dtype": "F16", "shape": [1024, 1024], "data_offsets": [0, 0]},
                  "b": {"dtype": "F32", "shape": [512, 512], "data_offsets": [0, 0]}}
        blob = json.dumps(header).encode()
        with open(os.path.join(hub, "x.safetensors"), "wb") as fh:
            fh.write(struct.pack("<Q", len(blob)) + blob)
        os.environ["HF_HOME"] = td
        got = checkpoint_dtypes()
        want_f16 = round(1024 * 1024 * 2 / 2**30, 2)
        want_f32 = round(512 * 512 * 4 / 2**30, 2)
        assert got == {"F16": want_f16, "F32": want_f32}, got
        os.environ["HF_HOME"] = os.path.join(td, "empty")
        assert checkpoint_dtypes() == {}, "an absent cache must read as absent"
    print("[ok] probe: safetensors header reader measures per-dtype GiB; absent cache is absent")
    return 0


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    if ap.parse_args().self_test:
        return _self_test()
    try:
        import torch
        import transformers
        from transformers import AutoModel
    except Exception as exc:  # noqa: BLE001 -- cannot judge without the stack
        _r(f"UNJUDGED: import failed: {exc!r}")
        return 2
    _r(f"transformers {transformers.__version__} torch {torch.__version__} "
       f"python {sys.version.split()[0]}")
    if not torch.cuda.is_available():
        _r("UNJUDGED: no CUDA device")
        return 2
    _r(f"gpu {torch.cuda.get_device_name(0)} "
       f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
    dtype = getattr(torch, DTYPE_NAME)

    ck = checkpoint_dtypes()
    if ck:
        _r(f"checkpoint GiB by stored dtype: {ck} total {sum(ck.values()):.2f}")
    else:
        _r("checkpoint not cached yet (will download)")

    t0 = time.time()
    try:
        model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, revision=REVISION,
                                          torch_dtype=dtype, low_cpu_mem_usage=True)
    except Exception as exc:  # noqa: BLE001 -- the failure IS the finding
        _r(f"FAIL: CPU load raised {type(exc).__name__}: {str(exc)[:300]}")
        return 1
    built = collections.Counter(str(p.dtype) for p in model.parameters())
    gib = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30
    _r(f"built on CPU in {time.time() - t0:.0f}s: param dtypes {dict(built)} = {gib:.2f} GiB "
       f"(requested {DTYPE_NAME})")
    if any(k != f"torch.{DTYPE_NAME}" for k in built):
        _r("NOTE: the remote code ignored the requested dtype for part of the model; "
           "the explicit cast below is load-bearing")

    try:
        model = model.to(dtype)
        t1 = time.time()
        model = model.to("cuda:0").eval()
    except Exception as exc:  # noqa: BLE001
        _r(f"FAIL: to(cuda) raised {type(exc).__name__}: {str(exc)[:300]}")
        return 1
    _r(f"on GPU in {time.time() - t1:.0f}s: allocated {torch.cuda.memory_allocated() / 2**30:.2f} "
       f"GiB, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    text = " ".join(["def f(x): return x"] * (MAX_LEN // 8))
    try:
        with torch.no_grad():
            emb = model.encode([text] * BATCH, instruction=INSTRUCTION, max_length=MAX_LEN)
    except Exception as exc:  # noqa: BLE001 -- e.g. the Cache API break shows ONLY here
        _r(f"FAIL: forward raised {type(exc).__name__}: {str(exc)[:300]}")
        return 1
    finite = bool(torch.isfinite(emb).all())
    _r(f"forward batch={BATCH} max_len={MAX_LEN}: shape {tuple(emb.shape)} finite={finite}; "
       f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    if not finite:
        _r("FAIL: non-finite embeddings")
        return 1
    _r("OK: teacher loads and answers on this box")
    return 0


if __name__ == "__main__":
    sys.exit(main())
