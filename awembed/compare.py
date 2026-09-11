"""awembed compare -- score ANY served embedding endpoints on YOUR corpus.

`awembed eval` answers "did my student beat the teacher it learned from". This
answers the question people actually ask before adopting one: **which of these
models should I run, on my documents?** It is black-box on purpose -- it talks to
openai-shaped `/v1/embeddings` endpoints, so the student, the teacher, and every
third-party model are measured by the same code on the same rows.

    awembed compare --corpus docs.jsonl --queries queries.jsonl \
        --endpoint student=http://127.0.0.1:18101 --endpoint nomic=http://127.0.0.1:18103 \
        --qprefix nomic="search_query: " --dprefix nomic="search_document: " \
        --out compare.json

docs.jsonl    {"id": "livingston", "text": "..."}          one row per CHUNK
queries.jsonl {"query": "...", "gold": "livingston"}       gold = a doc id above

Prefixes are per endpoint because the conventions differ and they change the
numbers: nomic wants `search_query: ` / `search_document: `, Qwen3 wants an
`Instruct: ...\\nQuery: ` preamble, and a distilled student wants whatever its own
training used. An endpoint with no prefix is sent the raw text.

Metrics are DOC-level, not chunk-level: a query counts as a hit when its gold
document is the top-ranked chunk's document, so a corpus split into many chunks
does not inflate or deflate a model by its chunker.

Gate (why this exits non-zero):
  * an endpoint that cannot be reached, or answers with the wrong dims, is exit 2
    -- "could not judge", never a 0.0 that reads like a bad model;
  * a query whose gold id is in NO document is reported and excluded, because a
    missing answer is a broken gold set, not a retrieval failure;
  * every endpoint must return the SAME number of vectors as it was sent.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path

_EMBED_BATCH = 16


def _log(msg: str) -> None:
    print(f"[awembed compare] {msg}", flush=True)


def l2(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def embed(url: str, texts: list[str], batch: int = _EMBED_BATCH) -> list[list[float]]:
    """POST to an OpenAI-shaped /v1/embeddings. Raises RuntimeError with the
    endpoint named -- a silent [] here would score 0.0 and read as a bad model."""
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        body = json.dumps({"input": texts[i:i + batch], "model": "x"}).encode()
        req = urllib.request.Request(url.rstrip("/") + "/v1/embeddings", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise RuntimeError(f"{url} unreachable: {exc}") from exc
        rows = sorted(data.get("data") or [], key=lambda r: r.get("index", 0))
        if len(rows) != min(batch, len(texts) - i):
            raise RuntimeError(f"{url} returned {len(rows)} vectors for "
                               f"{min(batch, len(texts) - i)} inputs")
        out.extend(r["embedding"] for r in rows)
    return out


def score(docs: list[dict], queries: list[dict],
          doc_vecs: list[list[float]], query_vecs: list[list[float]],
          dims: int = 0) -> tuple[dict, list[dict]]:
    """Pure ranking math. `dims` truncates before scoring (Matryoshka exports, or
    the 256-d form a volunteer plane stores) -- truncation is re-normalised, never
    left unnormalised."""
    if dims:
        doc_vecs = [l2(v[:dims]) for v in doc_vecs]
        query_vecs = [l2(v[:dims]) for v in query_vecs]
    doc_ids = [d["id"] for d in docs]
    known = set(doc_ids)
    rows, skipped = [], []
    for q in queries:
        gold = q["gold"]
        if gold not in known:
            skipped.append({"query": q["query"], "gold": gold})
            continue
        qv = query_vecs[len(rows) + len(skipped)]
        ranked = sorted(range(len(docs)), key=lambda j: cosine(qv, doc_vecs[j]), reverse=True)
        seen: list[str] = []
        for j in ranked:
            if doc_ids[j] not in seen:
                seen.append(doc_ids[j])
        rank = seen.index(gold) + 1
        rows.append({"query": q["query"], "gold": gold, "top": seen[0], "rank": rank})
    n = len(rows) or 1
    metrics = {
        "n": len(rows),
        "p@1": sum(r["rank"] == 1 for r in rows) / n,
        "doc@3": sum(r["rank"] <= 3 for r in rows) / n,
        "mrr": sum(1.0 / r["rank"] for r in rows) / n,
    }
    if skipped:
        metrics["skipped"] = len(skipped)
    return metrics, rows


def compare(endpoints: dict[str, str], docs: list[dict], queries: list[dict],
            qprefix: dict[str, str], dprefix: dict[str, str], dims: int = 0) -> dict:
    doc_texts = [d["text"] for d in docs]
    known = {d["id"] for d in docs}
    unmatched = {q["gold"] for q in queries if q["gold"] not in known}
    if len(unmatched) == len({q["gold"] for q in queries}):
        # EVERY query points at an id no document carries. Scoring this yields a
        # table of 0.000 that reads as "all four models are bad" when the truth is
        # "your gold column does not name your corpus". Measured 2026-09-10: the
        # first real run of this stage did exactly that and exited 0.
        sample = sorted(unmatched)[:3]
        raise RuntimeError(
            f"no query's gold id exists in the corpus (e.g. {sample}); "
            f"corpus ids look like {sorted(known)[:3]}")
    reports: dict[str, dict] = {}
    for name, url in endpoints.items():
        d_texts = [dprefix.get(name, "") + t for t in doc_texts]
        q_texts = [qprefix.get(name, "") + q["query"] for q in queries]
        dv = [l2(v) for v in embed(url, d_texts)]
        qv = [l2(v) for v in embed(url, q_texts)]
        # Validated HERE, at the consumer, not only inside embed(): a vector list
        # one short does not crash -- it silently pairs query i with document i+1
        # and every number below it is wrong. Measured 2026-09-10 (the short-batch
        # test): compare() raised IndexError from score() instead of a verdict.
        if len(dv) != len(d_texts) or len(qv) != len(q_texts):
            raise RuntimeError(f"{name}: {len(dv)} vector(s) for {len(d_texts)} documents, "
                               f"{len(qv)} for {len(q_texts)} queries")
        if len({len(v) for v in dv}) != 1:
            raise RuntimeError(f"{name}: inconsistent vector length")
        if dims and len(dv[0]) < dims:
            raise RuntimeError(f"{name}: {len(dv[0])}-d vectors cannot be cut to {dims}")
        m, rows = score(docs, queries, dv, qv, dims)
        if m["n"] == 0:
            raise RuntimeError(f"{name}: 0 of {len(queries)} queries were scorable")
        reports[name] = {"url": url, "dims": len(dv[0]), "metrics": m,
                         "per_query": rows if m["n"] <= 50 else rows[:50]}
    return reports


def render(reports: dict) -> str:
    width = max(len(n) for n in reports) + 2
    head = f"{'endpoint':{width}s} {'dims':>5s} {'n':>4s} {'p@1':>6s} {'doc@3':>6s} {'MRR':>6s}"
    lines = [head, "-" * len(head)]
    for name, rep in sorted(reports.items(), key=lambda kv: -kv[1]["metrics"]["p@1"]):
        m = rep["metrics"]
        lines.append(f"{name:{width}s} {rep['dims']:5d} {m['n']:4d} "
                     f"{m['p@1']:6.3f} {m['doc@3']:6.3f} {m['mrr']:6.3f}")
    return "\n".join(lines)


def _self_test() -> int:
    """The metric arithmetic, the truncation, the gold-miss rule, and the gate."""
    docs = [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}, {"id": "c", "text": "C"}]
    # q0 lands on a (rank 1); q1 lands on b while its gold is c (rank 2)
    query_vecs = [[1.0, 0.0], [0.0, 1.0]]
    doc_vecs = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    queries = [{"query": "qa", "gold": "a"}, {"query": "qc", "gold": "c"}]
    m, rows = score(docs, queries, doc_vecs, query_vecs)
    assert m["p@1"] == 0.5 and m["doc@3"] == 1.0, m
    assert abs(m["mrr"] - (1 + 1 / 2) / 2) < 1e-9, m
    assert [r["rank"] for r in rows] == [1, 2], rows

    # a gold id in no document is SKIPPED and counted, never scored as a miss
    m2, rows2 = score(docs, queries + [{"query": "qx", "gold": "zz"}], doc_vecs, query_vecs[:2] + [[1, 0]])
    assert m2["n"] == 2 and m2["skipped"] == 1, m2

    # truncation re-normalises: 2-d cut of a 4-d vector must stay unit length
    v = l2([3.0, 4.0, 0.0, 0.0])
    assert abs(math.sqrt(sum(x * x for x in l2(v[:2]))) - 1.0) < 1e-12

    # the gate: a batch answer shorter than the batch is refused, not scored
    class _Resp:
        def __init__(self, n): self.n = n
        def read(self): return json.dumps({"data": [{"index": i, "embedding": [1.0, 0.0]}
                                                    for i in range(self.n)]}).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False
    real = urllib.request.urlopen
    urllib.request.urlopen = lambda *a, **k: _Resp(1)          # one vector for two inputs
    try:
        try:
            embed("http://stub", ["x", "y"])
            raise AssertionError("a short batch was accepted")
        except RuntimeError as exc:
            assert "1 vectors for 2" in str(exc), exc
    finally:
        urllib.request.urlopen = real

    # a gold set that matches NOTHING must refuse, not print a table of 0.000
    try:
        compare({"m": "u1"}, docs, [{"query": "q", "gold": "not-a-doc"}], {}, {})
        raise AssertionError("an all-unmatched gold set was scored as 0.000")
    except RuntimeError as exc:
        assert "gold id exists" in str(exc), exc

    # per-endpoint prefixes are applied on both sides, and only where given.
    # compare() calls embed twice per endpoint (documents, then queries), so the
    # stub ACCUMULATES -- an overwrite would hide whichever call came first.
    seen: dict[str, list[str]] = {}
    def fake_embed(url, texts, batch=_EMBED_BATCH):
        seen.setdefault(url, []).extend(texts)
        return [[1.0, 0.0]] * len(texts)
    real_embed, sys.modules[__name__].embed = embed, fake_embed
    try:
        compare({"m": "u1"}, docs, queries, {"m": "Q: "}, {"m": "D: "})
        assert seen["u1"] == ["D: A", "D: B", "D: C", "Q: qa", "Q: qc"], seen
    finally:
        sys.modules[__name__].embed = real_embed

    print("[ok] compare: metrics, truncation, gold-miss, short-batch gate, prefixes")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", required=False, help="docs.jsonl: {id, text} per chunk")
    ap.add_argument("--queries", required=False, help="queries.jsonl: {query, gold}")
    ap.add_argument("--endpoint", action="append", default=[],
                    help="name=url, repeatable (e.g. student=http://127.0.0.1:18101)")
    ap.add_argument("--qprefix", action="append", default=[], help='name="text" per endpoint')
    ap.add_argument("--dprefix", action="append", default=[], help='name="text" per endpoint')
    ap.add_argument("--dims", type=int, default=0, help="truncate+renormalise before scoring")
    ap.add_argument("--out", help="write the full report (metrics + per-query) here")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    if not (args.corpus and args.queries and args.endpoint):
        ap.error("--corpus, --queries and at least one --endpoint are required")

    endpoints: dict[str, str] = {}
    for spec in args.endpoint:
        name, _, url = spec.partition("=")
        if not (name and url):
            ap.error(f"--endpoint must be name=url, got {spec!r}")
        endpoints[name] = url

    def _pairs(items: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for spec in items:
            name, _, text = spec.partition("=")
            out[name] = text
        return out

    docs = load_jsonl(Path(args.corpus))
    queries = load_jsonl(Path(args.queries))
    if not docs or not queries:
        _log("corpus and queries must both be non-empty")
        return 2
    try:
        reports = compare(endpoints, docs, queries, _pairs(args.qprefix),
                          _pairs(args.dprefix), args.dims)
    except RuntimeError as exc:
        _log(f"could not judge: {exc}")
        return 2
    print(render(reports))
    if args.out:
        Path(args.out).write_text(json.dumps(reports, indent=2), encoding="utf-8")
        _log(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
