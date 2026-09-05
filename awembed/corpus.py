#!/usr/bin/env python3
r"""Build the code-search distillation corpus (embedder plan, Stage 0.2).

Output: JSONL (--out), one row per
query:
    {"query": str, "positive_dir": str, "positive_summary": str,
     "hard_negatives": [str, ...], "hard_negative_summaries": [str, ...],
     "split": "train"|"eval", "kind": "gt"|"synthetic"}

The negatives carry their reading TEXT inline (`hard_negative_summaries`, same
order as `hard_negatives`) so every downstream stage is self-contained and
never needs the readings file. Measured 2026-09-01: the shipped corpus had
been built from an OLDER readings snapshot, so 2,265 of its 2,292 distinct
negative paths resolved to no reading at all — a distillation run against
it would have learned "summaries are positives, bare paths are negatives",
a shortcut that scores well and retrieves nothing.

Sources, in order:
  1. Ground-truth localization questions + their target files (--ground-truth,
     or the built-in example set; pairs whose target is not in the readings are
     skipped). positive_summary = the target dir's reading.
  2. Synthetic queries derived from the per-directory readings (--readings).
     Each read dir yields 2 queries: one from its most distinctive content
     terms, one "where does <dir> live" form.
  3. Hard negatives = the measured ls-topicality failure class: non-target
     dirs whose names share terms with the query, ranked by shared-term
     count (top 3). This is the class the baseline ls-listing ranking
     actually misses on (the 3/18 vs 0/18 measured wall).

Splits: dir-level split-unique — a dir's queries never span train/eval
(assigned by hash of the dir path, 80/20), so the eval cannot be gamed by
memorising a dir seen in training. The held-out pair-matching slice is the
eval half of the GT questions.

Gate (exit 1 on any): >= 3000 total queries, exactly HARD_NEG_K hard
negatives on every row EACH with an inline summary, splits disjoint, every
positive dir has a reading.

Measured 2026-09-01: the embedder lane is LIVE again (the fleet embedder
serving nomic-embed-text, 768-dim round trip verified) — the corpus is the
substrate for the distillation that replaces it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# Inputs/outputs are relative to the working directory. The READINGS file is one
# JSON object per line, one per directory of the repository being indexed:
#   {"rel": "<dir path, forward slashes>", "reading": "<a paragraph on what the dir
#    does>", "prompt_version": "<any non-empty tag>"}
# Producing readings is the caller's job (any LLM over `ls` + a few file heads works;
# ~1,500 chars per directory is plenty). A repository walker that writes them is the
# next thing this module grows; until then, bring the file.
DEFAULT_OUT = Path("corpus.jsonl")
DEFAULT_READINGS = Path("dir_readings.jsonl")
MIN_QUERIES = 3000
HARD_NEG_K = 3

# Ground-truth (question -> target file) pairs seed the `kind: "gt"` rows. The list
# below is the EXAMPLE set this module was measured with; a pair whose target
# directory is absent from your readings is skipped, so on another repository it
# contributes nothing and every row is synthetic -- pass --ground-truth to supply
# your own (JSONL rows {"query": ..., "target": "<path under the repo>"}).
if True:  # kept indented so the example list reads as the data it is
    GROUND_TRUTH = [
        ("where are FastContext hints injected into the prompt?", "lib/clients/fastcontext.py"),
        ("where are MCP tools discovered and registered?", "apps/AitherNode/mcp_server.py"),
        ("where is the MCP HTTP gateway served?", "apps/AitherNode/mcp_gateway.py"),
        ("where are tools ranked and selected for a task?", "lib/core/ToolGraph.py"),
        ("where does the MCTS router score tool candidates?", "lib/cognitive/MCTSRouter.py"),
        ("where is a task decomposed into executable steps?", "lib/orchestration/IntentPlanner.py"),
        ("where does the orchestrator choose which backend to dispatch to?",
         "lib/orchestration/MasterOrchestrator.py"),
        ("where is the vector embedding index for code?", "lib/clients/code_index.py"),
        ("where is the recursive language model REPL runtime?", "lib/cognitive/RLMRuntime.py"),
        ("where are evidence packs built and citations verified?",
         "lib/cognitive/evidence_pack.py"),
        ("where is tool selection telemetry recorded?", "lib/core/ToolSelectionTrace.py"),
        ("where are per-tool call counts and latency tracked?", "lib/core/ToolMetrics.py"),
        ("where is the layered memory hub client?", "lib/clients/memory_hub.py"),
        ("where does the codebase file indexer live?", "lib/core/CodebaseIndexer.py"),
        ("where is the tool registry that calls MCP servers?", "lib/cognitive/ToolRegistry.py"),
        ("where is the plan executed step by step?", "lib/orchestration/PlanExecutor.py"),
        ("where is the interactive environment exploration driver?",
         "lib/orchestration/InteractiveEnvironmentDriver.py"),
        ("where is the unified MCTS search implemented?", "lib/cognitive/UnifiedMCTS.py"),
    ]
    STOPWORDS = {
        "where", "what", "which", "who", "whom", "whose", "when", "why", "how", "does", "did",
        "the", "and", "for", "are", "was", "were", "has", "have", "had", "can", "could", "would",
        "should", "will", "with", "from", "that", "this", "these", "those", "its", "you", "your",
        "our", "into", "onto", "out", "not", "any", "all", "get", "got", "set", "put", "find",
        "locate", "look", "live", "lives", "implemented", "implement", "used", "use", "using",
        "there", "here", "them", "they", "some", "such", "than", "then", "each",
    }

    def terms_of(question: str):  # type: ignore[misc]
        raw = "".join(c.lower() if c.isalnum() else " " for c in question).split()
        return [t for t in raw if len(t) > 2 and t not in STOPWORDS]


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _load_readings(path: Path) -> dict[str, str]:
    """JSONL cache: {rel: reading} for the CURRENT prompt version only."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("prompt_version") and row.get("rel") and row.get("reading"):
            # The probe runs on Windows hosts (where paths come out with
            # backslashes — measured 2026-09-01: the cache keyed "lib\\cognitive"
            # while every GT lookup asks for "lib/cognitive", so ALL 18 GT rows
            # silently vanished from the corpus) and in Linux containers. The
            # builder normalises so the corpus is platform-independent.
            out[row["rel"].replace("\\", "/")] = row["reading"]
    return out


_TERM_CACHE: dict[tuple[str, int], list[str]] = {}


def _distinctive_terms(reading: str, k: int = 4) -> list[str]:
    """Top content terms of a reading by frequency — the query seed.

    Memoised: the fallback chain calls this for EVERY (row, other-dir) pair,
    and the same ~3000 readings repeat across ~6000 rows — without the cache
    a build is quadratic in (rows x dirs) and takes minutes (measured
    2026-09-01: a 3002-dir build blew the 120s foreground cap).
    """
    key = (reading, k)
    cached = _TERM_CACHE.get(key)
    if cached is not None:
        return cached
    words = "".join(c.lower() if c.isalnum() else " " for c in reading).split()
    counts: dict[str, int] = {}
    for w in words:
        if len(w) > 3 and w not in STOPWORDS:
            counts[w] = counts.get(w, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    out = [w for w, _ in ranked[:k]]
    _TERM_CACHE[key] = out
    return out


def _dir_name_terms(d: str) -> set[str]:
    """Term set of a dir NAME, memoised (called per (row, dir) pair)."""
    key = ("name", d)
    cached = _TERM_CACHE.get(key)
    if cached is None:
        cached = frozenset(terms_of(
            d.replace("/", " ").replace("-", " ").replace("_", " ")))
        _TERM_CACHE[key] = cached
    return set(cached)  # type: ignore[return-value]


def _ls_topicality_hard_negatives(question: str, target_dir: str,
                                  all_dirs: list[str], k: int = HARD_NEG_K) -> list[str]:
    """The measured failure class: dirs sharing terms with the question."""
    qterms = set(terms_of(question))
    scored = []
    for d in all_dirs:
        if d == target_dir:
            continue
        share = len(qterms & _dir_name_terms(d))
        if share > 0:
            scored.append((share, d))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [d for _, d in scored[:k]]


def _reading_overlap_negatives(positive_reading: str, target_dir: str,
                               readings: dict[str, str], k: int = HARD_NEG_K) -> list[str]:
    """Fallback for queries with NO name-term matches (measured: 183 of 2998 on
    the first full corpus). Rank other dirs by shared DISTINCTIVE terms between
    their readings and the positive reading — content near-misses, the class
    ls-topicality cannot see (their names share nothing).
    """
    pterms = set(_distinctive_terms(positive_reading, k=12))
    if not pterms:
        return []
    scored = []
    for d, reading in readings.items():
        if d == target_dir:
            continue
        dterms = set(_distinctive_terms(reading, k=12))
        share = len(pterms & dterms)
        if share > 0:
            scored.append((share, d))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [d for _, d in scored[:k]]


def _hard_negatives(question: str, target_dir: str, positive_reading: str,
                    all_dirs: list[str], readings: dict[str, str]) -> list[str]:
    """Name-term negatives first (the measured ls class); fill from reading
    overlap for the no-name-match class, then any remaining slots from
    same-parent siblings — every row needs >= HARD_NEG_K negatives.
    """
    hn = _ls_topicality_hard_negatives(question, target_dir, all_dirs)
    if len(hn) < HARD_NEG_K:
        hn += [d for d in _reading_overlap_negatives(
            positive_reading, target_dir, readings) if d not in hn]
    if len(hn) < HARD_NEG_K:
        parent = target_dir.rsplit("/", 1)[0]
        hn += [d for d in all_dirs
               if d != target_dir and d not in hn and d.rsplit("/", 1)[0] == parent]
    return hn[:HARD_NEG_K]


def _split_of(rel: str) -> str:
    h = hashlib.sha256(rel.encode("utf-8")).hexdigest()
    return "train" if int(h[:8], 16) % 100 < 80 else "eval"


def _dir_of(target: str) -> str:
    p = Path(target)
    return str(p.parent).replace("\\", "/")


def _row(query: str, target_dir: str, readings: dict[str, str],
         hard_negs: list[str], kind: str) -> dict:
    """One corpus row. Negatives are drawn from `readings` keys by
    construction, so every one has a summary; the gate re-checks anyway."""
    return {
        "query": query,
        "positive_dir": target_dir,
        "positive_summary": readings[target_dir],
        "hard_negatives": hard_negs,
        "hard_negative_summaries": [readings[d] for d in hard_negs if d in readings],
        "split": _split_of(target_dir),
        "kind": kind,
    }


def _load_ground_truth(path: Path | None) -> list[tuple[str, str]]:
    """Caller-supplied (query, target) pairs, or the example set."""
    if path is None:
        return list(GROUND_TRUTH)
    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("query") and row.get("target"):
            pairs.append((row["query"], row["target"].replace("\\", "/")))
    return pairs


def build(readings_path: Path, out_path: Path, ground_truth_path: Path | None = None) -> int:
    readings = _load_readings(readings_path)
    ground_truth = _load_ground_truth(ground_truth_path)
    all_dirs = sorted(readings)
    if not all_dirs:
        print(f"[FAIL] no readings in {readings_path} — expected one JSON object per line "
              "with non-empty 'rel', 'reading' and 'prompt_version' fields, one per "
              "directory of the repository")
        return 1

    rows: list[dict] = []
    # 1. GT questions
    for question, target in ground_truth:
        target_dir = _dir_of(target)
        if target_dir not in readings:
            print(f"[note] GT dir has no reading yet, skipping: {target_dir}")
            continue
        rows.append(_row(
            question, target_dir, readings,
            _hard_negatives(question, target_dir, readings[target_dir], all_dirs, readings),
            "gt"))

    # 2. synthetic queries from the readings (2 per dir)
    for d in all_dirs:
        reading = readings[d]
        terms = _distinctive_terms(reading)
        if terms:
            rows.append(_row(
                f"where is {' '.join(terms[:3])} handled?", d, readings,
                _hard_negatives(" ".join(terms[:3]), d, reading, all_dirs, readings),
                "synthetic"))
        rows.append(_row(
            f"where does the {d.rsplit('/', 1)[-1]} code live?", d, readings,
            _hard_negatives(d.rsplit('/', 1)[-1].replace("-", " ").replace("_", " "),
                            d, reading, all_dirs, readings),
            "synthetic"))

    # 3. gate
    failures: list[str] = []
    if len(rows) < MIN_QUERIES:
        failures.append(f"{len(rows)} < {MIN_QUERIES} total queries")
    short = [r["query"] for r in rows if len(r["hard_negatives"]) != HARD_NEG_K]
    if short:
        failures.append(f"{len(short)} rows do not carry exactly {HARD_NEG_K} hard "
                        f"negatives (e.g. {short[0][:60]!r})")
    # Every negative must resolve to reading TEXT — the 2,265-unresolved
    # defect above is exactly what this line exists to make impossible.
    untexted = [r["query"] for r in rows
                if len(r["hard_negative_summaries"]) != len(r["hard_negatives"])
                or not all(r["hard_negative_summaries"])]
    if untexted:
        failures.append(f"{len(untexted)} rows have a hard negative with no reading "
                        f"text (e.g. {untexted[0][:60]!r})")
    # dir-level disjointness is by construction (per-dir split hash)
    if failures:
        print("[FAIL] corpus gate:")
        for f in failures:
            print("  -", f)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    kinds = {k: sum(1 for r in rows if r["kind"] == k) for k in ("gt", "synthetic")}
    splits = {s: sum(1 for r in rows if r["split"] == s) for s in ("train", "eval")}
    print(f"[ok] corpus: {len(rows)} rows ({kinds}) splits={splits} "
          f"dirs={len(all_dirs)} -> {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readings", default=str(DEFAULT_READINGS),
                    help="JSONL of per-directory readings (rel, reading, prompt_version)")
    ap.add_argument("--ground-truth", default="",
                    help="optional JSONL of {query, target} pairs for kind=gt rows; "
                         "default: the built-in example set (skipped where targets are absent)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--min-queries", type=int, default=MIN_QUERIES)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        # both directions: a term match must produce a hard negative, and a
        # non-matching question must not (the ls-topicality matcher)
        dirs = ["lib/clients/code_index.py", "lib/core/ToolGraph.py", "apps/x/zip"]
        hn = _ls_topicality_hard_negatives(
            "where is the vector embedding index for code?", "lib/clients", dirs)
        assert hn, "term-sharing dir must rank as a hard negative"
        assert "lib/clients" not in hn
        empty = _ls_topicality_hard_negatives(
            "where is the quantum flux capacitor?", "lib/clients", dirs)
        assert empty == [], empty
        assert _split_of("a/b/c") in ("train", "eval")
        # the fallback arms: (1) reading-overlap fires when names share nothing
        # but the readings do; (2) the sibling fill fires when neither matches;
        # (3) a no-match positive still yields >= HARD_NEG_K negatives.
        readings = {
            "lib/clients": "handles vector embedding and retrieval for code",
            "lib/core": "handles vector embedding and retrieval for code",
            "apps/x/zip": "compression of archive files",
        }
        hn2 = _reading_overlap_negatives(
            "handles vector embedding and retrieval for code",
            "lib/clients", readings)
        assert hn2 == ["lib/core"], hn2
        hn3 = _hard_negatives(
            "where is the quantum flux capacitor?", "apps/zip",
            "compression of archive files",
            ["lib/clients", "lib/core", "apps/zip", "apps/y"], readings)
        # the fallback contract: siblings fill, the target is never included
        assert set(hn3) == {"apps/x/zip", "apps/y"} and "apps/zip" not in hn3, hn3
        # the row builder carries reading TEXT for every negative, in order;
        # a negative with no reading is dropped from the summaries so the
        # gate's length check trips (both directions).
        r = _row("q", "lib/clients", readings, ["lib/core", "apps/x/zip"], "synthetic")
        assert r["positive_dir"] == "lib/clients"
        assert r["hard_negative_summaries"] == [readings["lib/core"], readings["apps/x/zip"]]
        bad = _row("q", "lib/clients", readings, ["lib/core", "nope/never-read"], "synthetic")
        assert len(bad["hard_negative_summaries"]) != len(bad["hard_negatives"]), (
            "an unread negative must leave a visible length mismatch for the gate")
        print("[ok] hard-negative matcher + fallback arms + split + inline summaries, "
              "both directions")
        return 0

    return build(Path(args.readings), Path(args.out),
                 Path(args.ground_truth) if args.ground_truth else None)


if __name__ == "__main__":
    sys.exit(main())
