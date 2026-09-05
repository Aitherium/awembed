#!/usr/bin/env python3
r"""Stage 1 of the code-search embedder distillation: capture TEACHER targets.

Encodes every unique query and document in the corpus through the NV-Embed-v2
teacher (the teacher module, /v1/embeddings, dim 4096, L2-normalized) and
writes, into --out:

    teacher/queries/shard_NNNNN.safetensors   fp16 [n, 4096]   (+ .json sidecar)
    teacher/docs/shard_NNNNN.safetensors
    teacher/queries_index.json, teacher/docs_index.json   position -> text
    teacher_manifest.json                     dim, counts, corpus sha, model ids
    pairs.jsonl   one row per corpus row: query, positive, 3 negatives (TEXT),
                  split, kind, and the teacher's cosine scores t_pos / t_negs

Those scores are what Stage 2 distills (Margin-MSE on teacher score margins,
which is dim-agnostic — the student is dim 1024). The raw vectors are kept so
re-weighting a loss never re-captures, and so Stage 4 can report the
teacher's own retrieval numbers as the ceiling.

RESUMABLE, like k3_distill_capture.py: one checkpoint per shard, written
atomically; a shard whose sidecar sha matches its texts is skipped on rerun.

The teacher is CC-BY-NC-4.0 — its vectors and scores are INTERNAL training
targets and never ship. Only the Apache-2.0 student does.

Refuses (exit 1) a corpus without `hard_negative_summaries`: measured
2026-09-01, the shipped corpus had 2,265 negatives with no reading text, and
training against bare paths learns a shortcut instead of retrieval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

DIM = 4096
QUERY_MODEL = "nv-embed-v2-query"
DOC_MODEL = "nv-embed-v2"
HERE = Path(__file__).resolve().parent
# The teacher server lives beside this file: `teacher.py` in the awembed package,
# `k3_teacher_serve.py` in a flat single-file deployment. Same bytes either way.
DEFAULT_TEACHER_SCRIPT = next(
    (HERE / n for n in ("teacher.py", "k3_teacher_serve.py") if (HERE / n).exists()),
    HERE / "teacher.py",
)


def _log(msg: str) -> None:
    print(f"[capture] {msg}", flush=True)


def _sha(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- teachers -------------------------------------------------------------

class HttpTeacher:
    """The real thing: the teacher module's OpenAI-compatible endpoint."""

    def __init__(self, url: str, batch: int = 64):
        self.url = url.rstrip("/")
        self.batch = min(batch, 256)  # server caps a request at 256 inputs

    def embed(self, texts: list[str], model: str) -> np.ndarray:
        out = np.empty((len(texts), DIM), dtype=np.float32)
        for i in range(0, len(texts), self.batch):
            chunk = texts[i:i + self.batch]
            body = json.dumps({"model": model, "input": chunk}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.url}/v1/embeddings", data=body,
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read())["data"]
            for d in data:
                out[i + d["index"]] = np.asarray(d["embedding"], dtype=np.float32)
        return out


class FakeTeacher:
    """Deterministic stand-in for --self-test: sha256(text) seeds a unit vector.

    Same text -> same vector, so the resume proof below is real; different
    models get different vectors so a swapped alias is visible."""

    def embed(self, texts: list[str], model: str) -> np.ndarray:
        out = np.empty((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int(hashlib.sha256(f"{model}|{t}".encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
            out[i] = v / np.linalg.norm(v)
        return out


def make_teacher(url: str, batch: int):
    return FakeTeacher() if url == "fake://" else HttpTeacher(url, batch)


# --- sharded, resumable encode ------------------------------------------------

def encode_sharded(teacher, texts: list[str], model: str, out_dir: Path,
                   shard: int) -> np.ndarray:
    from safetensors.numpy import load_file, save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    vecs = np.empty((len(texts), DIM), dtype=np.float32)
    n_shards = math.ceil(len(texts) / shard) if texts else 0
    done = skipped = 0
    for si in range(n_shards):
        lo, hi = si * shard, min((si + 1) * shard, len(texts))
        chunk = texts[lo:hi]
        st = out_dir / f"shard_{si:05d}.safetensors"
        side = out_dir / f"shard_{si:05d}.json"
        want = {"lo": lo, "hi": hi, "sha": _sha(chunk), "model": model, "dim": DIM}
        if st.exists() and side.exists():
            try:
                if json.loads(side.read_text(encoding="utf-8")) == want:
                    vecs[lo:hi] = load_file(str(st))["emb"].astype(np.float32)
                    skipped += 1
                    continue
            except Exception:  # noqa: BLE001 — a checkpoint that cannot be READ is work not done
                # safetensors raises its own SafetensorError on a torn file
                # (found by --self-test); whatever the reason, recompute.
                _log(f"shard {si} unreadable — recomputing")
        emb = teacher.embed(chunk, model)
        if emb.shape != (hi - lo, DIM) or not np.isfinite(emb).all():
            raise SystemExit(f"[FAIL] teacher returned shape {emb.shape} / non-finite "
                             f"for shard {si} ({model})")
        # atomic: tmp -> replace, so a crash mid-write never leaves a torn shard
        # The fp16 SHARD is the source of truth: round-trip through it before
        # scoring, so a first run and a resumed run compute identical scores
        # (found by --self-test: fp32-then-fp16 differed in the 4th decimal).
        emb16 = emb.astype(np.float16)
        tmp = st.with_suffix(".tmp")
        save_file({"emb": emb16}, str(tmp))
        os.replace(tmp, st)
        side.write_bytes(json.dumps(want).encode("utf-8"))
        vecs[lo:hi] = emb16.astype(np.float32)
        done += 1
    _log(f"{model}: {len(texts)} texts in {n_shards} shards "
         f"({done} encoded, {skipped} resumed)")
    return vecs


# --- teacher lifecycle -----------------------------------------------------------

def _health(url: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=10) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as e:
        return 0, str(e)


def spawn_teacher(script: Path, port: int, timeout_s: int) -> subprocess.Popen:
    """Start the teacher server and block until /health is 200.

    500 = the model load FAILED (the server reports it rather than swallowing
    it) — fail fast with the body instead of waiting out the timeout."""
    if not script.exists():
        raise SystemExit(f"[FAIL] teacher script missing: {script}")
    env = dict(os.environ, NV_EMBED_PORT=str(port))
    # The teacher pins transformers==4.42.4 (NV-Embed's remote code breaks on
    # newer Cache APIs: 4.55 and 4.57 both raise DynamicCache.get_usable_length)
    # while the student needs >=4.51 for Qwen3 -- so the teacher may live in its
    # own venv. NV_EMBED_PYTHON names that interpreter; unset = this one.
    python = os.environ.get("NV_EMBED_PYTHON") or sys.executable
    proc = subprocess.Popen([python, str(script)], env=env)
    url = f"http://127.0.0.1:{port}"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise SystemExit(f"[FAIL] teacher exited early rc={proc.returncode}")
        code, body = _health(url)
        if code == 200:
            _log(f"teacher up at {url} after {time.time() - t0:.0f}s")
            return proc
        if code == 500:
            proc.terminate()
            raise SystemExit(f"[FAIL] teacher model load failed: {body[:400]}")
        time.sleep(5)
    proc.terminate()
    raise SystemExit(f"[FAIL] teacher not healthy after {timeout_s}s")


# --- the stage ------------------------------------------------------------------

REQUIRED = ("query", "positive_summary", "hard_negatives", "hard_negative_summaries",
            "split", "kind")


def load_corpus(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    bad = [i for i, r in enumerate(rows) if any(k not in r for k in REQUIRED)
           or len(r["hard_negative_summaries"]) != len(r["hard_negatives"])
           or not all(r["hard_negative_summaries"])]
    if bad:
        raise SystemExit(
            f"[FAIL] {len(bad)} corpus rows lack inline negative summaries (first: row "
            f"{bad[0]}). Regenerate with the corpus stage — training against "
            "path-only negatives learns a shortcut, not retrieval.")
    return rows


def run(corpus_path: Path, out: Path, seed: int, teacher, shard: int) -> int:
    rows = load_corpus(corpus_path)
    queries = list(dict.fromkeys(r["query"] for r in rows))
    docs = list(dict.fromkeys(
        [r["positive_summary"] for r in rows]
        + [s for r in rows for s in r["hard_negative_summaries"]]))
    qi = {t: i for i, t in enumerate(queries)}
    di = {t: i for i, t in enumerate(docs)}
    _log(f"{len(rows)} rows -> {len(queries)} unique queries, {len(docs)} unique docs")

    tdir = out / "teacher"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "queries_index.json").write_bytes(json.dumps(queries).encode("utf-8"))
    (tdir / "docs_index.json").write_bytes(json.dumps(docs).encode("utf-8"))
    q_vecs = encode_sharded(teacher, queries, QUERY_MODEL, tdir / "queries", shard)
    d_vecs = encode_sharded(teacher, docs, DOC_MODEL, tdir / "docs", shard)

    pairs = out / "pairs.jsonl"
    tmp = pairs.with_suffix(".tmp")
    n_bad = 0
    margins: list[float] = []
    with tmp.open("wb") as fh:
        for i, r in enumerate(rows):
            q = q_vecs[qi[r["query"]]]
            t_pos = float(q @ d_vecs[di[r["positive_summary"]]])
            t_negs = [float(q @ d_vecs[di[s]]) for s in r["hard_negative_summaries"]]
            if not all(math.isfinite(x) and -1.05 <= x <= 1.05 for x in [t_pos, *t_negs]):
                n_bad += 1
            margins.append(t_pos - sum(t_negs) / len(t_negs))
            fh.write(json.dumps({
                "i": i, "query": r["query"], "positive": r["positive_summary"],
                "positive_dir": r.get("positive_dir", ""),
                "negatives": r["hard_negative_summaries"],
                "split": r["split"], "kind": r["kind"],
                "t_pos": t_pos, "t_negs": t_negs,
            }).encode("utf-8") + b"\n")
    if n_bad:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"[FAIL] {n_bad} rows have non-finite / out-of-range teacher scores")
    os.replace(tmp, pairs)

    mean_margin = float(np.mean(margins))
    manifest = {
        "stage": "capture", "dim": DIM, "seed": seed,
        "n_rows": len(rows), "n_queries": len(queries), "n_docs": len(docs),
        "corpus": corpus_path.name, "corpus_sha256": _file_sha(corpus_path),
        "query_model": QUERY_MODEL, "doc_model": DOC_MODEL,
        "teacher": type(teacher).__name__,
        "teacher_mean_margin_pos_minus_neg": mean_margin,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "teacher_manifest.json").write_bytes(json.dumps(manifest, indent=2).encode())
    _log(f"pairs.jsonl: {len(rows)} rows; teacher mean margin (pos - mean neg) = "
         f"{mean_margin:+.4f}")
    if isinstance(teacher, HttpTeacher) and mean_margin <= 0:
        raise SystemExit("[FAIL] the teacher does not prefer positives over negatives on "
                         "average — its targets would teach the wrong thing")
    return 0


def _self_test() -> int:
    """Offline proof with the FakeTeacher: contract, atomic resume, the corpus gate."""
    rows = []
    for k in range(8):
        rows.append({"query": f"where is thing {k}?", "positive_dir": f"d/{k}",
                     "positive_summary": f"summary of thing {k}",
                     "hard_negatives": [f"d/n{k}{j}" for j in range(3)],
                     "hard_negative_summaries": [f"neg {k}-{j}" for j in range(3)],
                     "split": "train" if k < 6 else "eval", "kind": "synthetic"})
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        corpus = tdp / "c.jsonl"
        corpus.write_bytes(b"".join(json.dumps(r).encode() + b"\n" for r in rows))
        out = tdp / "out"
        assert run(corpus, out, 1, FakeTeacher(), shard=3) == 0
        pairs = [json.loads(x) for x in (out / "pairs.jsonl").read_text().splitlines()]
        assert len(pairs) == 8 and all(len(p["t_negs"]) == 3 for p in pairs)
        assert all(-1 <= p["t_pos"] <= 1 for p in pairs)
        # resume: damage ONE doc shard, rerun, and only that shard is rewritten
        shards = sorted((out / "teacher" / "docs").glob("shard_*.safetensors"))
        assert len(shards) >= 2, shards
        before = {s: s.stat().st_mtime_ns for s in shards}
        shards[1].write_bytes(b"torn")
        time.sleep(0.02)
        assert run(corpus, out, 1, FakeTeacher(), shard=3) == 0
        after = {s: s.stat().st_mtime_ns for s in shards}
        assert after[shards[1]] != before[shards[1]], "damaged shard was not recomputed"
        assert after[shards[0]] == before[shards[0]], "intact shard was needlessly rewritten"
        pairs2 = [json.loads(x) for x in (out / "pairs.jsonl").read_text().splitlines()]
        assert [p["t_pos"] for p in pairs] == [p["t_pos"] for p in pairs2], "not deterministic"
        # the corpus gate: a row without inline summaries must be refused
        bad = dict(rows[0])
        bad.pop("hard_negative_summaries")
        (tdp / "bad.jsonl").write_bytes(json.dumps(bad).encode() + b"\n")
        try:
            run(tdp / "bad.jsonl", tdp / "out2", 1, FakeTeacher(), shard=3)
        except SystemExit as e:
            assert "inline negative summaries" in str(e), e
        else:
            raise AssertionError("a path-only corpus was accepted")
    print("[ok] capture: contract, atomic resume, determinism, corpus gate — all proven")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", help="corpus .jsonl (needs hard_negative_summaries)")
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--out", help="stage output root (shared by all stages)")
    ap.add_argument("--teacher-url", default="",
                    help="running teacher; empty = spawn --teacher-script locally; "
                         "'fake://' = deterministic stub")
    ap.add_argument("--teacher-script", default=str(DEFAULT_TEACHER_SCRIPT))
    ap.add_argument("--port", type=int, default=8213)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--shard", type=int, default=256)
    ap.add_argument("--teacher-timeout", type=int, default=1800,
                    help="seconds to wait for the 16 GB teacher to download + load")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    if not args.corpus or not args.out:
        ap.error("--corpus and --out are required")

    proc = None
    url = args.teacher_url
    try:
        if not url:
            proc = spawn_teacher(Path(args.teacher_script), args.port, args.teacher_timeout)
            url = f"http://127.0.0.1:{args.port}"
        return run(Path(args.corpus), Path(args.out), args.seed,
                   make_teacher(url, args.batch), args.shard)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
            _log("teacher stopped — VRAM released for Stage 2")


if __name__ == "__main__":
    sys.exit(main())
