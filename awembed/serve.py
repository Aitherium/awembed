"""`awembed serve` — run a trained embedder as an OpenAI-compatible endpoint.

The serving half of the pipeline: capture -> distill -> quantize -> export -> SERVE. It
loads the ONNX export with onnxruntime, so the same artifact runs on a laptop CPU, in a
container, or on a phone through onnxruntime-android, and in a browser through
transformers.js -- one file, four surfaces.

Pooling is the MODEL's, never this module's choice: a mean-pooled BERT student compared
against last-token vectors scored 0.9618 against ITSELF (2026-10-27), which reads as a
broken export when nothing is wrong. `--pooling` defaults to the value recorded by the
ONNX stage's sidecar when present.

Optional dependencies: `pip install "awembed[serve]"`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Module scope, not inside the factory: FastAPI resolves handler annotations through the
# module globals, and this file uses `from __future__ import annotations`. A `Request`
# imported inside the function then answers 422 "missing query param request" on every
# call while /health stays green (measured 2026-10-27, twice).
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


def _resolve(model_dir: Path) -> tuple[Path, str]:
    """(onnx file, pooling) from a run dir or an export dir, honouring its sidecar."""
    pooling = os.environ.get("AITHER_EMBED_POOLING", "")
    for side in (model_dir / "onnx_sidecar.json", model_dir.parent / "onnx_sidecar.json"):
        if side.exists():
            try:
                import json

                data = json.loads(side.read_text(encoding="utf-8"))
                pooling = pooling or str(data.get("pooling") or "")
            except (OSError, ValueError) as exc:
                # Not silent: an unreadable sidecar means the pooling below is a guess,
                # and a guessed pooling measures a different model without saying so.
                print(f"[serve] sidecar {side} unreadable ({exc}); pooling falls back to "
                      f"{pooling or 'mean'}", file=sys.stderr)
    for cand in (model_dir / "onnx" / "model_quantized.onnx",
                 model_dir / "onnx" / "model.onnx",
                 model_dir / "model_quantized.onnx"):
        if cand.exists():
            return cand, (pooling or "mean")
    raise FileNotFoundError(
        f"no ONNX model under {model_dir} — export it first "
        f"(k3_embed_onnx.py --out <run>) or point --model-dir at the export"
    )


def build_app(model_dir: Path, max_len: int, pooling: Optional[str], device: str) -> Any:
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    onnx_path, resolved = _resolve(model_dir)
    pooling = (pooling or resolved).lower()
    providers = ["CPUExecutionProvider"]
    if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    names = {i.name for i in sess.get_inputs()}
    tok = AutoTokenizer.from_pretrained(str(onnx_path.parent.parent))
    state: Dict[str, Any] = {"dim": None}

    def encode(texts: List[str]) -> List[List[float]]:
        enc = tok(texts, padding=True, truncation=True, max_length=max_len,
                  return_tensors="np")
        feed = {n: enc[n].astype(np.int64) for n in names if n in enc}
        hidden = sess.run(["last_hidden_state"], feed)[0]
        mask = enc["attention_mask"][..., None].astype(hidden.dtype)
        if pooling == "mean":
            vec = (hidden * mask).sum(1) / np.clip(mask.sum(1), 1.0, None)
        else:
            idx = enc["attention_mask"].sum(1) - 1
            vec = hidden[np.arange(hidden.shape[0]), idx]
        vec = vec / np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12, None)
        state["dim"] = int(vec.shape[1])
        return vec.tolist()

    app = FastAPI()

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "healthy", "service": "awembed-serve",
                "model": onnx_path.name, "providers": sess.get_providers(),
                "pooling": pooling, "dim": state["dim"]}

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> JSONResponse:
        body = await request.json()
        raw = body.get("input")
        texts = [raw] if isinstance(raw, str) else list(raw or [])
        if not texts:
            raise HTTPException(status_code=400, detail="input is required")
        vecs = encode([str(t) for t in texts])
        return JSONResponse({
            "object": "list",
            "model": body.get("model") or model_dir.name,
            "data": [{"object": "embedding", "index": i, "embedding": v}
                     for i, v in enumerate(vecs)],
        })

    return app


def build_parser() -> argparse.ArgumentParser:
    """The stage's flags, separate from main() so a test can pin them without loading a
    model (and without the serve extra installed)."""
    ap = argparse.ArgumentParser(prog="awembed serve",
                                 description="serve a trained embedder over /v1/embeddings")
    ap.add_argument("--model-dir", required=True, help="run dir or ONNX export dir")
    ap.add_argument("--port", type=int, default=8310)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address; 0.0.0.0 is required for another CONTAINER to "
                         "reach it (a published port alone is not enough -- measured "
                         "2026-10-27 on the adk appliance compose)")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--pooling", choices=("mean", "last"), default=None,
                    help="default: the ONNX sidecar's value, else mean")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import uvicorn  # noqa: F401
        app = build_app(Path(args.model_dir), args.max_len, args.pooling, args.device)
    except ImportError as exc:
        print(f'awembed serve needs the serve extra: pip install "awembed[serve]" ({exc})',
              file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - a lane that cannot load is exit 2, not a crash
        print(f"could not load {args.model_dir}: {exc}", file=sys.stderr)
        return 2
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0
