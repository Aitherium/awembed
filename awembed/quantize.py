#!/usr/bin/env python3
r"""Stage 3 of the code-search embedder distillation: QUANTIZE the student.

Weight-only int8 on every nn.Linear of <out>/student (Stage 2): each weight
matrix is rounded to int8 with one symmetric scale per OUTPUT channel and the
scales are kept in fp32; everything else (embeddings, norms) stays fp32. Saved to
<out>/student_int8/ as model_int8.pt + config + tokenizer, with
quant_sidecar.json (fp32 bytes, int8 bytes, ratio, fidelity). load_int8()
dequantizes into an ordinary fp32 model, so the artifact is a SIZE deliverable.

Why weight-only and not dynamic int8 (the previous scheme): measured 2026-09-02 on
the trained 0.6B student over 64 eval docs -- dynamic int8 (activations quantized
per tensor at runtime) gave mean cosine 0.873 against the fp32 student, and the
damage was the MLP blocks (MLP-only 0.888, attention-only 0.987): Qwen3's MLP
activations carry outliers an 8-bit per-tensor range cannot hold. Per-channel
WEIGHT scales alone changed nothing (0.875), so the weights were never the
problem. Weight-only int8 per channel: mean cosine 0.9993, min 0.9991. In the eval
stage the dynamic export had cost recall@10 0.899 -> 0.872 and failed the floor.

Fidelity gate (exit 1): mean cosine between fp32 and int8 embeddings over a
sample of eval-split documents must be >= --min-cosine (default 0.98). A
quantized export that silently drifted is worse than none — the eval stage
would then bless a model that does not embed what the trained one embeds.

Dynamic int8 is a CPU-engine format (the edge/CPU deliverable); GPU serving
keeps the bf16 student. Stage 4 evaluates BOTH.
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


def _log(msg: str) -> None:
    print(f"[quantize] {msg}", flush=True)


def _param_bytes(model) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def sample_eval_docs(out: Path, n: int) -> list[str]:
    p = out / "pairs.jsonl"
    if not p.exists():
        raise SystemExit(f"[FAIL] {p} missing — run the capture stage first")
    docs: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["split"] == "eval":
            docs.append(r["positive"])
            docs.extend(r["negatives"])
    docs = list(dict.fromkeys(docs))
    if not docs:
        raise SystemExit("[FAIL] no eval docs in pairs.jsonl")
    return docs[:n]


INT8_SCHEME = "int8_weight_only_per_channel"


def int8_weight_only_state(model) -> dict:
    """The compact export: every nn.Linear weight as int8 + a per-output-channel fp32
    scale (`<name>.weight` int8, `<name>.weight_scale` [out, 1]); every other tensor
    of the state dict verbatim. Symmetric, absmax per row, 127 levels."""
    linear_weights = {f"{n}.weight" for n, m in model.named_modules()
                      if isinstance(m, torch.nn.Linear)}
    tensors: dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        if k in linear_weights:
            scale = v.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
            tensors[k] = torch.round(v / scale).clamp(-127, 127).to(torch.int8)
            tensors[k[:-len(".weight")] + ".weight_scale"] = scale.contiguous()
        else:
            tensors[k] = v.detach().clone()
    return {"scheme": INT8_SCHEME, "tensors": tensors}


def quantize(student_dir: Path, out_dir: Path, texts: list[str], min_cos: float,
             max_len: int, batch: int) -> dict:
    tok, model = load_student(student_dir, "cpu", torch.float32)
    ref = encode_texts(tok, model, texts, max_len, batch, "cpu")
    fp32_bytes = _param_bytes(model)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(int8_weight_only_state(model), out_dir / "model_int8.pt")
    tok.save_pretrained(str(out_dir))
    model.config.save_pretrained(str(out_dir))
    # Fidelity is measured on what the eval stage will actually LOAD -- the saved
    # file through load_int8 -- not on an in-memory twin, so the reload contract is
    # part of the number rather than a separate assertion.
    _, qmodel = load_int8(out_dir)
    q_emb = encode_texts(tok, qmodel, texts, max_len, batch, "cpu")
    cos = (ref * q_emb).sum(-1)
    mean_cos, min_row = cos.mean().item(), cos.min().item()
    int8_bytes = (out_dir / "model_int8.pt").stat().st_size
    side = {
        "stage": "quantize", "source": str(student_dir), "scheme": INT8_SCHEME,
        "fp32_param_bytes": fp32_bytes, "int8_file_bytes": int8_bytes,
        "ratio": round(fp32_bytes / max(1, int8_bytes), 3),
        "fidelity_mean_cosine": mean_cos, "fidelity_min_cosine": min_row,
        "fidelity_n_texts": len(texts), "min_cosine_gate": min_cos,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "quant_sidecar.json").write_bytes(json.dumps(side, indent=2).encode())
    _log(f"fp32 {fp32_bytes / 1e6:.0f} MB -> int8 {int8_bytes / 1e6:.0f} MB "
         f"({side['ratio']}x); fidelity mean cos {mean_cos:.4f} min {min_row:.4f}")
    if mean_cos < min_cos:
        raise SystemExit(f"[FAIL] int8 fidelity {mean_cos:.4f} < gate {min_cos}")
    return side


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


def _self_test() -> int:
    """Quantize a thumbnail Qwen3, prove the export reloads to IDENTICAL outputs,
    and prove the fidelity gate can refuse."""
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        tiny = tiny_student(tdp / "tiny")
        out = tdp / "out"
        out.mkdir()
        rows = [{"i": k, "query": f"where is thing {k}", "positive": f"summary of thing {k}",
                 "negatives": [f"neg {k} {j}" for j in range(3)], "split": "eval",
                 "kind": "synthetic", "t_pos": 0.8, "t_negs": [0.2, 0.1, 0.3]} for k in range(6)]
        (out / "pairs.jsonl").write_bytes(b"".join(json.dumps(r).encode() + b"\n" for r in rows))
        texts = sample_eval_docs(out, 128)
        assert len(texts) == 24, len(texts)
        side = quantize(tiny, out / "student_int8", texts, min_cos=0.5, max_len=16, batch=8)
        assert side["ratio"] > 1.5, f"int8 export did not shrink: {side['ratio']}x"
        assert side["fidelity_mean_cosine"] > 0.5
        # reload contract: the saved int8 model embeds exactly like a fresh quantization
        tok, q = load_int8(out / "student_int8")
        got = encode_texts(tok, q, texts, 16, 8, "cpu")
        tok0, m0 = load_student(tiny, "cpu", torch.float32)
        blob = int8_weight_only_state(m0)
        assert any(t.dtype == torch.int8 for t in blob["tensors"].values()), "nothing quantized"
        deq = {}
        for k, v in blob["tensors"].items():
            if k.endswith(".weight_scale"):
                continue
            scale = blob["tensors"].get(k[:-7] + ".weight_scale")
            deq[k] = v.float() * scale if v.dtype == torch.int8 else v
        m0.load_state_dict(deq)
        want = encode_texts(tok0, m0, texts, 16, 8, "cpu")
        assert torch.allclose(got, want, atol=1e-5), "reloaded int8 model drifted from export"
        # a foreign scheme must be refused, not silently loaded as garbage
        bad = out / "student_int8_bad"
        bad.mkdir()
        for f in (out / "student_int8").iterdir():
            if f.name != "model_int8.pt":
                (bad / f.name).write_bytes(f.read_bytes())
        torch.save({"scheme": "dynamic_int8_linear", "tensors": {}}, bad / "model_int8.pt")
        try:
            load_int8(bad)
        except SystemExit as e:
            assert "scheme" in str(e), e
        else:
            raise AssertionError("load_int8 accepted an unknown scheme")
        # gate direction: an impossible threshold must refuse
        try:
            quantize(tiny, out / "student_int8_b", texts, min_cos=1.01, max_len=16, batch=8)
        except SystemExit as e:
            assert "fidelity" in str(e), e
        else:
            raise AssertionError("the fidelity gate accepted an impossible threshold")
    print("[ok] quantize: shrink, byte-faithful reload, scheme refusal, "
          "fidelity gate both directions")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", help="accepted for the stage contract; pairs.jsonl is the input")
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--out", help="stage output root (shared by all stages)")
    ap.add_argument("--student-dir", default="", help="default <out>/student")
    ap.add_argument("--int8-dir", default="", help="default <out>/student_int8")
    ap.add_argument("--min-cosine", type=float, default=0.98)
    ap.add_argument("--n-texts", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    if not args.out:
        ap.error("--out is required")
    out = Path(args.out)
    student = Path(args.student_dir) if args.student_dir else out / "student"
    if not (student / "config.json").exists():
        raise SystemExit(f"[FAIL] no trained student at {student} — run Stage 2 first")
    quantize(student, Path(args.int8_dir) if args.int8_dir else out / "student_int8",
             sample_eval_docs(out, args.n_texts), args.min_cosine, args.max_len, args.batch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
