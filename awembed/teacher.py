"""NV-Embed-v2 OpenAI-compatible embeddings server (internal-only, CC-BY-NC-4.0).

Stage 0.1 of the code-search embedder distillation: this is the TEACHER server.
Ported verbatim 2026-09-01 from the DGX's proven 147-line
/home/wzns/nv-embed-serve/server.py (ran once as aither-nv-embed-dgx, exit 0) —
do not "improve" the serving half; every guard below was earned live on spark
(CUDA OOM on the naive .to("cuda") path 2026-07-28; 4096-token activation
spikes killing the process). This port only adds the house --self-test and the
pool-family docstring.

vLLM cannot serve this model (custom NVEmbedModel latent-attention pooling), so this
is a plain HF transformers wrapper exposing /v1/embeddings. License is NON-COMMERCIAL:
this backend must never be wired into customer-facing surfaces.

Model aliases:
  nv-embed-v2        -> passage/document encoding (no instruction)
  nv-embed-v2-query  -> retrieval query encoding (default retrieval instruction)
Optional request field "instruction" overrides the alias default.
Embeddings are L2-normalized (cosine-ready).

The distillation (Stage 2) consumes this server's /v1/embeddings output as the
teacher targets; the student is Qwen3-Embedding-0.6B (Apache-2.0), so the
CC-BY-NC lineage stays with the teacher and never reaches a customer surface.
"""
import asyncio
import logging
import os
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from transformers import AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("nv-embed")

MODEL_ID = "nvidia/NV-Embed-v2"
REVISION = os.environ.get("NV_EMBED_REVISION", "3fa59658547db50a1e8e3346cf057fd0c77ed6ef")
PORT = int(os.environ.get("NV_EMBED_PORT", "8213"))
BATCH = int(os.environ.get("NV_EMBED_BATCH", "2"))
DEFAULT_MAX_LEN = int(os.environ.get("NV_EMBED_MAX_LEN", "1024"))
# Served dtype. The checkpoint is stored F16 (304 tensors, 14.62 GiB, measured
# 2026-09-02), so float16 is the no-cast choice; see load_model() for why the
# cast has to be applied explicitly at all.
DTYPE = getattr(torch, os.environ.get("NV_EMBED_DTYPE", "float16"))
# Hard ceiling on per-request max_length. Spark runs ~1-8 GB free once the model is
# resident; a 4096-token x batch-4 activation spike CUDA-OOMed and KILLED the process
# on first deploy (2026-07-28). 1024 tokens x batch 2 spikes <1.5 GB and matches the
# fleet's real retrieval envelope (nomic itself serves max_model_len=2048).
MAX_LEN_CLAMP = int(os.environ.get("NV_EMBED_MAX_LEN_CLAMP", "1024"))
QUERY_INSTRUCTION = os.environ.get(
    "NV_EMBED_QUERY_INSTRUCTION",
    "Instruct: Given a question, retrieve passages that answer the question\nQuery: ",
)

app = FastAPI(title="nv-embed-v2")
state = {"model": None, "loaded": False, "error": None}
gpu_lock = threading.Lock()


def load_model():
    try:
        t0 = time.time()
        log.info("loading %s rev=%s (%s, cuda)...", MODEL_ID, REVISION, DTYPE)
        # device_map loads shards straight to GPU. The naive from_pretrained + .to("cuda")
        # path peaks at ~2x weights (CPU copy + GPU copy = ~32 GB transient) — on spark's
        # unified memory that OOMed the reload twice on 2026-07-28.
        # NV-Embed's remote code honours `dtype` for its 14 latent-attention tensors
        # ONLY; the Mistral backbone (290 params) is built float32 regardless, so the
        # "bfloat16" model this server has always logged was 27.87 GiB of fp32
        # (measured 2026-09-02 on a CPU load: dtypes {fp16: 14, fp32: 290}). On the
        # Spark's 128 GB that merely wasted memory; on a 24 GB A10 it OOMed at 21.4
        # GiB three runs in a row, in fp16 and bf16 alike, and looked like a load-path
        # transient. The explicit .to(DTYPE) is the fix; it belongs on BOTH paths so
        # the served dtype is the declared one everywhere.
        common = dict(trust_remote_code=True, revision=REVISION, torch_dtype=DTYPE,
                      low_cpu_mem_usage=True)
        try:
            model = AutoModel.from_pretrained(MODEL_ID, device_map={"": 0}, **common)
            model = model.to(DTYPE)
        except torch.OutOfMemoryError as oom:
            # Direct-to-GPU cannot fit the fp32-built backbone on a 24 GB card. Stage
            # through host RAM: build + cast there (fp32 transient lives in RAM), then
            # move the finished fp16 model over -- the GPU peak is the model, 14.6 GiB
            # (measured: load+cast 8s, to(cuda) 3s, allocated == peak == 14.62 GiB).
            log.warning("direct GPU load OOMed (%s); staging through host RAM", oom)
            torch.cuda.empty_cache()
            model = AutoModel.from_pretrained(MODEL_ID, **common).to(DTYPE).to("cuda:0")
        model.eval()
        state["model"] = model
        state["loaded"] = True
        log.info("model loaded in %.1fs — LICENSE CC-BY-NC-4.0, INTERNAL USE ONLY",
                 time.time() - t0)
    except Exception as e:  # startup failure must be visible, not swallowed
        state["error"] = repr(e)
        log.exception("model load FAILED")


@app.on_event("startup")
def _startup():
    threading.Thread(target=load_model, daemon=True).start()


@app.get("/health")
def health():
    if state["loaded"]:
        return {"status": "ok", "model": MODEL_ID, "license": "CC-BY-NC-4.0 internal-only"}
    code = 500 if state["error"] else 503
    return JSONResponse({"status": "loading", "error": state["error"]}, status_code=code)


@app.get("/v1/models")
def models():
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": "nv-embed-v2", "object": "model", "owned_by": "nvidia", "created": now,
             "root": MODEL_ID, "license": "CC-BY-NC-4.0", "internal_only": True},
            {"id": "nv-embed-v2-query", "object": "model", "owned_by": "nvidia", "created": now,
             "root": MODEL_ID, "license": "CC-BY-NC-4.0", "internal_only": True},
        ],
    }


def _encode(texts, instruction, max_length):
    model = state["model"]
    out = []
    with gpu_lock, torch.no_grad():
        try:
            for i in range(0, len(texts), BATCH):
                emb = model.encode(texts[i : i + BATCH], instruction=instruction,
                                   max_length=max_length)
                emb = torch.nn.functional.normalize(emb.float(), p=2, dim=1)
                out.extend(emb.cpu().tolist())
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log.error("CUDA OOM encoding %d texts at max_length=%d", len(texts), max_length)
            raise HTTPException(status_code=507, detail="CUDA OOM — reduce batch or max_length")
    return out


@app.post("/v1/embeddings")
async def embeddings(req: Request):
    body = await req.json()
    if not state["loaded"]:
        raise HTTPException(status_code=503, detail="model still loading")
    texts = body.get("input")
    if isinstance(texts, str):
        texts = [texts]
    if not isinstance(texts, list) or not texts or not all(isinstance(t, str) for t in texts):
        raise HTTPException(status_code=400,
                            detail="input must be a non-empty string or list of strings")
    if len(texts) > 256:
        raise HTTPException(status_code=400, detail="max 256 inputs per request")
    model_name = body.get("model", "nv-embed-v2")
    if model_name not in ("nv-embed-v2", "nv-embed-v2-query"):
        raise HTTPException(status_code=404, detail=f"unknown model {model_name!r}")
    instruction = body.get("instruction")
    if instruction is None:
        instruction = QUERY_INSTRUCTION if model_name == "nv-embed-v2-query" else ""
    max_length = min(max(int(body.get("max_length", DEFAULT_MAX_LEN)), 128), MAX_LEN_CLAMP)

    t0 = time.time()
    vecs = await asyncio.to_thread(_encode, texts, instruction, max_length)
    log.info("encoded %d texts in %.2fs (model=%s)", len(texts), time.time() - t0, model_name)
    return {
        "object": "list",
        "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)],
        "model": model_name,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


def _self_test() -> int:
    """Prove the pipeline contract WITHOUT the 16 GB teacher load.

    The real golden check (embed through the loaded model, assert dim 4096 +
    finiteness) runs on the Lambda box inside the embed-trainer container.
    Offline, this asserts the same contract against a stub: the encode path
    must return exactly dim-4096, finite, L2-normalized vectors and must
    refuse a wrong-dimension result — both directions, so a silent shape
    change fails.
    """
    import math

    class _Stub:
        def encode(self, texts, instruction, max_length):
            dim = 4096
            # exactly unit-norm rows, shaped like the real model.encode output
            return torch.full((len(texts), dim), math.sqrt(1.0 / dim))

    state["model"] = _Stub()
    state["loaded"] = True
    vecs = _encode(["golden query"], "", 512)
    assert len(vecs) == 1 and len(vecs[0]) == 4096, f"dim != 4096: {len(vecs[0]) if vecs else 0}"
    assert all(math.isfinite(x) for x in vecs[0]), "non-finite embedding"
    norm = math.sqrt(sum(x * x for x in vecs[0]))
    assert abs(norm - 1.0) < 1e-6, f"not L2-normalized: {norm}"
    print("[ok] teacher encode contract: dim 4096, finite, L2-normalized")
    return 0


def main() -> int:
    """Serve the teacher (or prove its encode contract with --self-test).

    Port/model/dtype come from the NV_EMBED_* environment so the same entrypoint
    works as `awembed teacher`, as a flat single-file server on a rented box, and
    under a process supervisor."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--port", type=int, default=PORT, help=f"default {PORT} (NV_EMBED_PORT)")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
