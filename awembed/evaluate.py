#!/usr/bin/env python3
r"""Stage 4 of the code-search embedder distillation: EVALUATE and GATE.

On the eval split of <out>/pairs.jsonl (dirs never seen in training — the
corpus splits at dir level), scores four systems:

    teacher    NV-Embed-v2 from the Stage 1 vectors      (the ceiling)
    baseline   Qwen3-Embedding-0.6B, untrained          (what we started from)
    student    <out>/student, bf16 on GPU / fp32 on CPU  (Stage 2)
    int8       <out>/student_int8, CPU                    (Stage 3)

Metrics:
    p@1, mrr      rank the positive against its own 3 hard negatives
    recall@k      rank the positive against EVERY distinct eval-split document

Gates (exit 1 on any) — relative, because "trained" is not "improved":
    student.recall@10 > baseline.recall@10  AND  student.p@1 > baseline.p@1
    int8.recall@10 >= --int8-floor x student.recall@10   (default 0.98)

Writes <out>/eval_report.json. --self-test proves the metric arithmetic
against hand-computed values and exercises the gate in both directions.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as nnf

BASELINE_ID = "Qwen/Qwen3-Embedding-0.6B"


def _log(msg: str) -> None:
    print(f"[eval] {msg}", flush=True)


# --- pure metrics (no model) ----------------------------------------------------

def hard_negative_metrics(s_pos: torch.Tensor, s_neg: torch.Tensor) -> dict:
    """s_pos [N], s_neg [N,K]. p@1: positive strictly above all its negatives.
    mrr: 1 / (1 + number of negatives scoring >= the positive)."""
    beaten_by = (s_neg >= s_pos[:, None]).sum(1).float()
    return {"p@1": (beaten_by == 0).float().mean().item(),
            "mrr": (1.0 / (1.0 + beaten_by)).mean().item()}


def recall_at_k(q_vec: torch.Tensor, d_vec: torch.Tensor, gold: torch.Tensor,
                ks=(1, 5, 10)) -> dict:
    """q_vec [N,d], d_vec [M,d] (normalized), gold [N] index into d_vec. Ties beat."""
    sims = q_vec @ d_vec.T                                      # [N,M]
    gold_sim = sims.gather(1, gold[:, None])                    # [N,1]
    rank = (sims >= gold_sim).sum(1)                            # 1 = top
    return {f"recall@{k}": (rank <= k).float().mean().item() for k in ks}


def evaluate(name: str, q_vec: torch.Tensor, d_vec: torch.Tensor, doc_index: dict,
             pairs: list[dict]) -> dict:
    """q_vec [N,d] aligned with pairs; d_vec [M,d] aligned with doc_index."""
    gold = torch.tensor([doc_index[p["positive"]] for p in pairs])
    negs = torch.tensor([[doc_index[n] for n in p["negatives"]] for p in pairs])
    s_pos = (q_vec * d_vec[gold]).sum(-1)
    s_neg = torch.einsum("nd,nkd->nk", q_vec, d_vec[negs])
    m = {"system": name, "n_queries": len(pairs), "n_docs": d_vec.shape[0]}
    m.update(hard_negative_metrics(s_pos, s_neg))
    m.update(recall_at_k(q_vec, d_vec, gold))
    _log(f"{name:9s} p@1={m['p@1']:.3f} mrr={m['mrr']:.3f} r@1={m['recall@1']:.3f} "
         f"r@5={m['recall@5']:.3f} r@10={m['recall@10']:.3f}")
    return m


def gate(report: dict, int8_floor: float) -> list[str]:
    fails = []
    b, s, q = report.get("baseline"), report.get("student"), report.get("int8")
    if b and s:
        if not s["recall@10"] > b["recall@10"]:
            fails.append(f"student recall@10 {s['recall@10']:.3f} <= baseline {b['recall@10']:.3f}")
        if not s["p@1"] > b["p@1"]:
            fails.append(f"student p@1 {s['p@1']:.3f} <= baseline {b['p@1']:.3f}")
    if s and q and q["recall@10"] < int8_floor * s["recall@10"]:
        fails.append(f"int8 recall@10 {q['recall@10']:.3f} < {int8_floor} x student "
                     f"{s['recall@10']:.3f}")
    return fails


def eval_pairs(out: Path) -> tuple[list[dict], list[str], dict]:
    p = out / "pairs.jsonl"
    if not p.exists():
        raise SystemExit(f"[FAIL] {p} missing — run the capture stage first")
    pairs = [r for r in map(json.loads, p.read_text(encoding="utf-8").splitlines())
             if r["split"] == "eval"]
    if not pairs:
        raise SystemExit("[FAIL] no eval rows in pairs.jsonl")
    docs = list(dict.fromkeys([r["positive"] for r in pairs]
                              + [n for r in pairs for n in r["negatives"]]))
    return pairs, docs, {t: i for i, t in enumerate(docs)}


def teacher_vectors(out: Path, queries: list[str], docs: list[str]):
    """Reassemble Stage 1's sharded fp16 vectors for exactly these texts."""
    from safetensors.numpy import load_file
    tdir = out / "teacher"

    def gather(sub: str, index_name: str, wanted: list[str]):
        if not (tdir / index_name).exists():
            return None  # Stage 1 never ran here (e.g. --self-test): no ceiling, not a crash
        index = json.loads((tdir / index_name).read_text(encoding="utf-8"))
        pos = {t: i for i, t in enumerate(index)}
        shards = sorted((tdir / sub).glob("shard_*.safetensors"))
        if not shards:
            return None
        import numpy as np
        allv = np.concatenate([load_file(str(s))["emb"] for s in shards]).astype("float32")
        try:
            return torch.from_numpy(allv[[pos[t] for t in wanted]])
        except KeyError:
            return None

    return gather("queries", "queries_index.json", queries), gather("docs", "docs_index.json", docs)


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


def load_int8(int8_dir, base_dir=None):
    """Rebuild the weight-only-int8 student from <int8_dir>/model_int8.pt.

    The export holds every nn.Linear weight as int8 plus a per-output-channel
    fp32 scale (`<name>.weight_scale`) and every other tensor verbatim; the
    architecture comes from the saved config. Dequantize (int8 * scale) into an
    ordinary fp32 model -- the artifact is a size deliverable, the runtime is
    plain fp32. Kept identical in the quantize and evaluate modules
    (asserted by test_embed_stage_scripts.py)."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    int8_dir = Path(int8_dir)
    tok = AutoTokenizer.from_pretrained(str(int8_dir), padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(str(int8_dir))
    model = AutoModel.from_config(cfg)
    blob = torch.load(int8_dir / "model_int8.pt", map_location="cpu", weights_only=True)
    if blob.get("scheme") != "int8_weight_only_per_channel":
        raise SystemExit(f"[FAIL] {int8_dir}: unknown int8 scheme {blob.get('scheme')!r}")
    tensors = blob["tensors"]
    state = {}
    for k, v in tensors.items():
        if k.endswith(".weight_scale"):
            continue
        if v.dtype == torch.int8:
            state[k] = v.float() * tensors[k[:-len(".weight")] + ".weight_scale"]
        else:
            state[k] = v
    model.load_state_dict(state)
    model.eval()
    return tok, model


# --- systems ------------------------------------------------------------------------------

def system_vectors(tok, model, device, queries, docs, max_len, batch):
    q = encode_texts(tok, model, queries, max_len, batch, device, QUERY_PREFIX)
    d = encode_texts(tok, model, docs, max_len, batch, device)
    return q, d


def run(out: Path, baseline: str, student_dir: Path, int8_dir: Path, device: str,
        max_len: int, batch: int, int8_floor: float) -> int:
    pairs, docs, doc_index = eval_pairs(out)
    queries = [p["query"] for p in pairs]
    _log(f"{len(pairs)} eval queries over {len(docs)} distinct eval docs")
    report: dict = {"n_eval_queries": len(pairs), "n_eval_docs": len(docs),
                    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    tq, td = teacher_vectors(out, queries, docs)
    if tq is not None and td is not None:
        report["teacher"] = evaluate("teacher", tq, td, doc_index, pairs)
    else:
        _log("teacher vectors absent for this split — no ceiling reported")

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    if baseline:
        tok, m = load_student(baseline, device, dtype)
        report["baseline"] = evaluate("baseline", *system_vectors(tok, m, device, queries, docs,
                                                                 max_len, batch), doc_index, pairs)
        del m
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if not (student_dir / "config.json").exists():
        raise SystemExit(f"[FAIL] no trained student at {student_dir} — run Stage 2 first")
    tok, m = load_student(student_dir, device, dtype)
    report["student"] = evaluate("student", *system_vectors(tok, m, device, queries, docs,
                                                           max_len, batch), doc_index, pairs)
    del m

    if (int8_dir / "model_int8.pt").exists():
        tok, q = load_int8(int8_dir)
        report["int8"] = evaluate("int8", *system_vectors(tok, q, "cpu", queries, docs,
                                                         max_len, batch), doc_index, pairs)
    else:
        _log(f"no int8 export at {int8_dir} — skipping (Stage 3 not run)")

    fails = gate(report, int8_floor)
    report["gate_failures"] = fails
    (out / "eval_report.json").write_bytes(json.dumps(report, indent=2).encode("utf-8"))
    _log(f"report -> {out / 'eval_report.json'}")
    if fails:
        for f in fails:
            _log(f"[FAIL] {f}")
        return 1
    _log("[ok] gates passed" if "baseline" in report else
         "[note] no baseline evaluated — improvement gate not exercised")
    return 0


def _self_test() -> int:
    """Metric arithmetic against hand-computed values, the gate in both
    directions, and a run() smoke on a thumbnail student (no baseline/int8)."""
    # hard-negative metrics: row0 beats all 3 -> p@1 hit, rr 1; row1 beaten by
    # two (0.6, 0.5) -> rr 1/3. p@1 = 0.5, mrr = (1 + 1/3) / 2.
    m = hard_negative_metrics(torch.tensor([0.9, 0.5]),
                              torch.tensor([[0.1, 0.2, 0.3], [0.6, 0.4, 0.5]]))
    assert abs(m["p@1"] - 0.5) < 1e-6 and abs(m["mrr"] - (1 + 1 / 3) / 2) < 1e-6, m
    # recall: q0 == D0 (rank 1); q1 leans toward D2 over its gold D1 (rank 2)
    eye = torch.eye(4)
    qs = torch.stack([eye[0], nnf.normalize(0.6 * eye[1] + 0.8 * eye[2], dim=0)])
    r = recall_at_k(qs, eye, torch.tensor([0, 1]), ks=(1, 2, 5))
    assert r == {"recall@1": 0.5, "recall@2": 1.0, "recall@5": 1.0}, r
    # gate directions
    ok = {"baseline": {"recall@10": .5, "p@1": .5}, "student": {"recall@10": .6, "p@1": .6},
          "int8": {"recall@10": .59}}
    assert gate(ok, 0.98) == [], gate(ok, 0.98)
    worse = dict(ok, student={"recall@10": .5, "p@1": .6})
    assert any("recall@10" in f for f in gate(worse, 0.98)), "a non-improving student passed"
    lossy = dict(ok, int8={"recall@10": .4})
    assert any("int8" in f for f in gate(lossy, 0.98)), "a lossy int8 passed"
    # run() smoke on a thumbnail student
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        tiny = tiny_student(tdp / "tiny")
        out = tdp / "out"
        out.mkdir()
        rows = [{"i": k, "query": f"where is thing {k}", "positive": f"summary of thing {k}",
                 "negatives": [f"neg {k} {j}" for j in range(3)], "split": "eval",
                 "kind": "synthetic", "t_pos": 0.8, "t_negs": [0.2, 0.1, 0.3]} for k in range(6)]
        (out / "pairs.jsonl").write_bytes(b"".join(json.dumps(x).encode() + b"\n" for x in rows))
        assert run(out, "", tiny, out / "student_int8", "cpu", 16, 8, 0.98) == 0
        rep = json.loads((out / "eval_report.json").read_text(encoding="utf-8"))
        assert rep["student"]["n_queries"] == 6 and rep["student"]["n_docs"] == 24, rep["student"]
        assert 0.0 <= rep["student"]["recall@10"] <= 1.0
    print("[ok] eval: metric arithmetic, gate both directions, run() smoke")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", help="accepted for the stage contract; pairs.jsonl is the input")
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--out", help="stage output root (shared by all stages)")
    ap.add_argument("--baseline", default=BASELINE_ID, help="'' to skip the untrained baseline")
    ap.add_argument("--student-dir", default="", help="default <out>/student")
    ap.add_argument("--int8-dir", default="", help="default <out>/student_int8")
    ap.add_argument("--int8-floor", type=float, default=0.98)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    if not args.out:
        ap.error("--out is required")
    out = Path(args.out)
    return run(out, args.baseline, Path(args.student_dir) if args.student_dir else out / "student",
               Path(args.int8_dir) if args.int8_dir else out / "student_int8", args.device,
               args.max_len, args.batch, args.int8_floor)


if __name__ == "__main__":
    sys.exit(main())
