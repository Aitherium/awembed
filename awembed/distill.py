#!/usr/bin/env python3
r"""Stage 2 of the code-search embedder distillation: train the STUDENT.

Reads <out>/pairs.jsonl (Stage 1) and fine-tunes Qwen3-Embedding-0.6B
(Apache-2.0, dim 1024) against the NV-Embed-v2 teacher's scores. Writes
<out>/student/ (save_pretrained + tokenizer) and train_sidecar.json.

The loss is dim-AGNOSTIC on purpose — teacher is 4096-d, student 1024-d:
  * InfoNCE over the positive, every hard negative in the batch, and the
    other rows' positives (in-batch negatives), temperature 0.05;
  * Margin-MSE (Hofstätter et al. 2020): MSE between the student's
    (s_pos - s_neg_k) margins and the teacher's — the canonical dense-retrieval
    distillation, and the only part that is actually "distilling".

Gate (exit 1): the mean loss over the last 10 steps must be below the mean
over the first 10, and no step may go non-finite. A run that "completed" with
no learning is the failure this family exists to catch
([[a-lora-can-train-export-and-do-nothing]]).

fp32 master weights + bf16 autocast on CUDA (0.6B + AdamW fits an A10
comfortably); --self-test runs the identical path on a thumbnail Qwen3, CPU.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as nnf

STUDENT_ID = "Qwen/Qwen3-Embedding-0.6B"


def _log(msg: str) -> None:
    print(f"[distill] {msg}", flush=True)


def load_pairs(out: Path, split: str) -> list[dict]:
    p = out / "pairs.jsonl"
    if not p.exists():
        raise SystemExit(f"[FAIL] {p} missing — run the capture stage first")
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    rows = [r for r in rows if r["split"] == split]
    if not rows:
        raise SystemExit(f"[FAIL] no {split!r} rows in {p}")
    k = len(rows[0]["negatives"])
    if k < 1 or any(len(r["negatives"]) != k or len(r["t_negs"]) != k for r in rows):
        raise SystemExit("[FAIL] every pairs row must carry the same K negatives + teacher scores")
    return rows


def batches(rows: list[dict], batch: int, seed: int, epoch: int):
    idx = list(range(len(rows)))
    random.Random(seed * 1000 + epoch).shuffle(idx)
    for i in range(0, len(idx), batch):
        yield [rows[j] for j in idx[i:i + batch]]


def distill_loss(q, pos, negs, t_pos, t_negs, temp: float, margin_w: float):
    """q [B,d], pos [B,d], negs [B,K,d], all L2-normalized; t_* teacher cosines."""
    n_b, n_k, d = negs.shape
    docs = torch.cat([pos, negs.reshape(n_b * n_k, d)], 0)   # [B + B*K, d]
    logits = (q @ docs.T) / temp
    nce = nnf.cross_entropy(logits, torch.arange(n_b, device=q.device))
    s_pos = (q * pos).sum(-1)                                    # [B]
    s_neg = torch.einsum("bd,bkd->bk", q, negs)                  # [B,K]
    mse = nnf.mse_loss(s_pos[:, None] - s_neg, t_pos[:, None] - t_negs)
    return nce + margin_w * mse, nce.detach(), mse.detach()


def _embed_grad(tok, model, texts, max_len, device, prefix=""):
    enc = tok([prefix + t for t in texts], padding=True, truncation=True,
              max_length=max_len, return_tensors="pt").to(device)
    h = model(**enc).last_hidden_state
    return nnf.normalize(pool_last_token(h, enc["attention_mask"]).float(), dim=-1)


def train(out: Path, student: str, seed: int, epochs: int, batch: int, lr: float,
          max_len: int, temp: float, margin_w: float, device: str,
          max_steps: int = 0, log_every: int = 10):
    torch.manual_seed(seed)
    random.seed(seed)
    rows = load_pairs(out, "train")
    tok, model = load_student(student, device, torch.float32)
    model.train()
    if device.startswith("cuda"):
        # Each step embeds batch x (1 query + 1 positive + K negatives) sequences WITH
        # grad -- 80 sequences at batch 16, K=3. Without checkpointing the per-layer
        # activations of that forward dominate: measured 2026-09-02 on a 24 GB A10,
        # step 0 OOMed at 19.4 GiB allocated (fp32 master + AdamW is ~9.6 GiB of it).
        # Recomputing activations in backward trades ~30% step time for fitting the
        # declared batch, so the seed/batch/schedule stay the reproducible record.
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = math.ceil(len(rows) / batch) * epochs
    if max_steps:
        total = min(total, max_steps)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    use_amp = device.startswith("cuda")
    history: list[dict] = []
    step, t0 = 0, time.time()
    _log(f"{len(rows)} train rows, K={len(rows[0]['negatives'])}, {total} steps, "
         f"batch {batch}, lr {lr}, device {device}, amp={use_amp}")
    for epoch in range(epochs):
        for b in batches(rows, batch, seed, epoch):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                q = _embed_grad(tok, model, [r["query"] for r in b], max_len, device, QUERY_PREFIX)
                pos = _embed_grad(tok, model, [r["positive"] for r in b], max_len, device)
                flat = [n for r in b for n in r["negatives"]]
                negs = _embed_grad(tok, model, flat, max_len, device).view(len(b), -1, q.shape[-1])
            t_pos = torch.tensor([r["t_pos"] for r in b], device=device)
            t_negs = torch.tensor([r["t_negs"] for r in b], device=device)
            loss, nce, mse = distill_loss(q, pos, negs, t_pos, t_negs, temp, margin_w)
            if not torch.isfinite(loss):
                raise SystemExit(f"[FAIL] non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            history.append({"step": step, "loss": loss.item(), "nce": nce.item(),
                            "mse": mse.item()})
            if step % log_every == 0:
                _log(f"step {step}/{total} loss={loss.item():.4f} nce={nce.item():.4f} "
                     f"mse={mse.item():.5f} lr={sched.get_last_lr()[0]:.2e} "
                     f"{time.time() - t0:.0f}s")
            step += 1
            if step >= total:
                break
        if step >= total:
            break
    return tok, model, history


# --- shared student helpers ------------------------------------------------------
# Kept BYTE-IDENTICAL across the distill, quantize and evaluate modules (each ships
# to the box as a single file, so there is no shared module). The stage-script parity
# test asserts the three copies match.

QUERY_PREFIX = ("Instruct: Given a code search question, retrieve the directory "
                "summary that answers it\nQuery: ")


def load_student(path, device, dtype):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(path), padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(str(path), torch_dtype=dtype).to(device)
    return tok, model


def pool_last_token(hidden, attention_mask):
    """Qwen3-Embedding pools the LAST token. With left padding that is column -1
    for every row; with right padding fall back to each row's last attended
    index, so the contract holds under either tokenizer setting."""
    if bool(attention_mask[:, -1].all()):
        return hidden[:, -1]
    idx = attention_mask.sum(1) - 1
    return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]


def encode_texts(tok, model, texts, max_len, batch, device, prefix=""):
    """No-grad, L2-normalized embeddings on CPU, in input order."""
    out = []
    model.eval()
    for i in range(0, len(texts), batch):
        enc = tok([prefix + t for t in texts[i:i + batch]], padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt").to(device)
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        out.append(nnf.normalize(pool_last_token(h, enc["attention_mask"]).float(), dim=-1).cpu())
    return torch.cat(out) if out else torch.empty(0)


def tiny_student(tmp):
    """A REAL tokenizer + a 2-layer Qwen3 the size of a thumbnail, saved to
    `tmp` in the exact layout save_pretrained produces — so --self-test
    exercises the same load/pool/encode path as the 0.6B model, offline."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model
    vocab = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2}
    for w in ("where is thing summary of neg the code live handled instruct given "
              "a search question retrieve directory that answers it query").split():
        vocab.setdefault(w, len(vocab))
    for k in range(64):
        vocab.setdefault(str(k), len(vocab))
    t = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="[PAD]", unk_token="[UNK]",
                                  eos_token="[EOS]", padding_side="left")
    cfg = Qwen3Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=8, max_position_embeddings=128)
    torch.manual_seed(0)
    model = Qwen3Model(cfg)
    tok.save_pretrained(str(tmp))
    model.save_pretrained(str(tmp))
    return tmp


# --- save + gate ----------------------------------------------------------------------

def save_and_gate(out: Path, tok, model, history: list[dict], cfg: dict) -> int:
    sdir = out / "student"
    sdir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(sdir), safe_serialization=True)
    tok.save_pretrained(str(sdir))
    first = sum(h["loss"] for h in history[:10]) / max(1, len(history[:10]))
    last = sum(h["loss"] for h in history[-10:]) / max(1, len(history[-10:]))
    side = dict(cfg, stage="distill", steps=len(history), loss_first10=first, loss_last10=last,
                history=history,
                created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (sdir / "train_sidecar.json").write_bytes(json.dumps(side, indent=1).encode("utf-8"))
    _log(f"saved {sdir}; loss first10={first:.4f} last10={last:.4f}")
    if not last < first:
        raise SystemExit(f"[FAIL] did not learn: last10 {last:.4f} >= first10 {first:.4f}")
    return 0


def _self_test() -> int:
    """The identical train/save/gate path on a thumbnail Qwen3, CPU, offline."""
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        tiny = tiny_student(tdp / "tiny")
        out = tdp / "out"
        out.mkdir()
        rows = []
        for k in range(12):
            rows.append({"i": k, "query": f"where is thing {k}",
                         "positive": f"summary of thing {k}",
                         "positive_dir": f"d/{k}", "negatives": [f"neg {k} {j}" for j in range(3)],
                         "split": "train", "kind": "synthetic",
                         "t_pos": 0.8, "t_negs": [0.2, 0.1, 0.3]})
        (out / "pairs.jsonl").write_bytes(b"".join(json.dumps(r).encode() + b"\n" for r in rows))
        tok, model, hist = train(out, str(tiny), seed=1, epochs=8, batch=4, lr=1e-2, max_len=16,
                                 temp=0.05, margin_w=1.0, device="cpu", log_every=1000)
        assert len(hist) == 24, len(hist)
        assert save_and_gate(out, tok, model, hist, {"student": "tiny"}) == 0
        # pooling contract: left padding must not change a text's pooled vector
        tok2, m2 = load_student(out / "student", "cpu", torch.float32)
        a = encode_texts(tok2, m2, ["where is thing 3"], 16, 8, "cpu")
        b = encode_texts(tok2, m2, ["where is thing 3", "summary of thing 0 neg 1 2 3 4 5"],
                         16, 8, "cpu")
        cos = float((a[0] * b[0]).sum())
        assert cos > 0.99, f"padding changed the pooled vector: cos={cos:.4f}"
        # gate direction: a flat loss curve must be refused
        flat = [{"step": i, "loss": 1.0, "nce": 1.0, "mse": 0.0} for i in range(20)]
        try:
            save_and_gate(tdp / "out2", tok, model, flat, {})
        except SystemExit as e:
            assert "did not learn" in str(e), e
        else:
            raise AssertionError("a run that learned nothing passed the gate")
    print("[ok] distill: train/save/gate path, padding-invariant pooling, gate both directions")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", help="accepted for the stage contract; pairs.jsonl is the input")
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--out", help="stage output root (shared by all stages)")
    ap.add_argument("--student", default=STUDENT_ID)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--margin-weight", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = full schedule")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    if not args.out:
        ap.error("--out is required")
    out = Path(args.out)
    tok, model, hist = train(out, args.student, args.seed, args.epochs, args.batch, args.lr,
                             args.max_len, args.temp, args.margin_weight, args.device,
                             max_steps=args.max_steps)
    cfg = {k: v for k, v in vars(args).items() if k != "self_test"}
    return save_and_gate(out, tok, model, hist, cfg)


if __name__ == "__main__":
    sys.exit(main())
