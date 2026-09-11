"""awembed compare: the black-box endpoint stage -- metrics, the gate, and the
refusal that matters. No torch and no network: the endpoint is stubbed."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from awembed import compare  # noqa: E402


DOCS = [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}, {"id": "c", "text": "C"}]


def _stub(monkeypatch, vectors):
    """One endpoint whose vectors are handed back in call order."""
    calls = {"n": 0}

    def fake(url, texts, batch=compare._EMBED_BATCH):
        calls["n"] += 1
        return vectors(calls["n"], texts)

    monkeypatch.setattr(compare, "embed", fake)
    return calls


def test_metrics_are_doc_level_not_chunk_level():
    # two chunks of the SAME document must not create two competing 'documents'
    docs = [{"id": "a", "text": "A1"}, {"id": "a", "text": "A2"}, {"id": "b", "text": "B1"}]
    dv = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
    qv = [[1.0, 0.0]]
    m, rows = compare.score(docs, [{"query": "q", "gold": "a"}], dv, qv)
    assert m["p@1"] == 1.0 and rows[0]["top"] == "a", (m, rows)


def test_a_gold_id_in_no_document_is_skipped_and_counted():
    m, _ = compare.score(DOCS, [{"query": "q", "gold": "zz"}], [[1.0, 0.0]] * 3, [[1.0, 0.0]])
    assert m["n"] == 0 and m["skipped"] == 1, m


def test_all_unmatched_gold_refuses_rather_than_scoring_zero(monkeypatch):
    _stub(monkeypatch, lambda n, texts: [[1.0, 0.0]] * len(texts))
    try:
        compare.compare({"m": "u"}, DOCS, [{"query": "q", "gold": "nope"}], {}, {})
        raise AssertionError("a gold set matching nothing was scored")
    except RuntimeError as exc:
        assert "gold id" in str(exc), exc


def test_a_short_batch_is_a_gate_not_a_score(monkeypatch):
    _stub(monkeypatch, lambda n, texts: [[1.0, 0.0]] * (len(texts) - 1))
    try:
        compare.compare({"m": "u"}, DOCS, [{"query": "q", "gold": "a"}], {}, {})
        raise AssertionError("an endpoint that dropped a vector was scored")
    except RuntimeError as exc:
        # the message names the counts, so a misalignment is diagnosable from the log
        assert "2 vector(s) for 3 documents" in str(exc), exc


def test_truncation_renormalises(monkeypatch):
    # a full-length vector truncated to 2-d must stay unit length, or the cosine
    # silently shrinks with the dimension count and every model looks equally bad
    _stub(monkeypatch, lambda n, texts: [[0.6, 0.8, 0.0, 0.0]] * len(texts))
    reports = compare.compare({"m": "u"}, DOCS, [{"query": "q", "gold": "a"}], {}, {}, dims=2)
    assert reports["m"]["dims"] == 4
    v = compare.l2([0.6, 0.8, 0.0, 0.0][:2])
    assert abs(sum(x * x for x in v) - 1.0) < 1e-12


def test_prefixes_apply_per_endpoint_and_only_where_given(monkeypatch):
    seen = {}

    def fake(url, texts, batch=compare._EMBED_BATCH):
        seen.setdefault(url, []).extend(texts)
        return [[1.0, 0.0]] * len(texts)

    monkeypatch.setattr(compare, "embed", fake)
    compare.compare({"p": "u1", "raw": "u2"}, DOCS, [{"query": "q", "gold": "a"}],
                    {"p": "Q: "}, {"p": "D: "})
    assert seen["u1"] == ["D: A", "D: B", "D: C", "Q: q"], seen
    assert seen["u2"] == ["A", "B", "C", "q"], seen


def test_render_sorts_by_p1_and_names_every_endpoint():
    reports = {"slow": {"url": "u", "dims": 8, "metrics": {"n": 3, "p@1": .3, "doc@3": .6, "mrr": .4},
                        "per_query": []},
               "best": {"url": "u", "dims": 8, "metrics": {"n": 3, "p@1": .9, "doc@3": 1.0, "mrr": .95},
                        "per_query": []}}
    out = compare.render(reports).splitlines()
    assert "best" in out[2] and "slow" in out[3], out


def test_cli_writes_the_report_and_a_missing_endpoint_is_exit_2(tmp_path, monkeypatch, capsys):
    (tmp_path / "docs.jsonl").write_text(json.dumps({"id": "a", "text": "A"}) + "\n", encoding="utf-8")
    (tmp_path / "queries.jsonl").write_text(json.dumps({"query": "q", "gold": "a"}) + "\n", encoding="utf-8")
    _stub(monkeypatch, lambda n, texts: [[1.0, 0.0]] * len(texts))
    out = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "awembed compare", "--corpus", str(tmp_path / "docs.jsonl"),
        "--queries", str(tmp_path / "queries.jsonl"), "--endpoint", "m=u", "--out", str(out)])
    assert compare.main() == 0
    assert json.loads(out.read_text(encoding="utf-8"))["m"]["metrics"]["p@1"] == 1.0

    monkeypatch.setattr(compare, "embed", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("u unreachable: refused")))
    monkeypatch.setattr(sys, "argv", [
        "awembed compare", "--corpus", str(tmp_path / "docs.jsonl"),
        "--queries", str(tmp_path / "queries.jsonl"), "--endpoint", "m=u"])
    assert compare.main() == 2
    assert "could not judge" in capsys.readouterr().out
