"""awembed -- Aither World Embed: train an embedding model that knows YOUR corpus.

    awembed corpus    build question -> right-answer -> hard-negative rows from a repo
    awembed teacher   serve a large teacher embedder as an OpenAI-compatible endpoint
    awembed probe     prove the teacher loads and answers on THIS machine (before you rent)
    awembed capture   score every row with the teacher (its margins are the target)
    awembed distill   train the small student against the teacher + the corpus labels
    awembed quantize  weight-only int8 export with a fidelity gate
    awembed eval      teacher vs baseline vs student vs int8 on the held-out split
    awembed run       capture -> distill -> quantize -> eval, in order, one output root

Every stage is a standalone module with its own `--self-test` and its own gate that
exits non-zero when the artifact it produced is not fit to hand on. A run that
"completed" without learning is the failure this tool is built to refuse.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from typing import Sequence

STAGES = {
    "corpus": ("awembed.corpus", "build the training corpus from a repository"),
    "teacher": ("awembed.teacher", "serve the teacher embedder (OpenAI-compatible)"),
    "probe": ("awembed.probe", "teacher load footprint + one forward on this machine"),
    "capture": ("awembed.capture", "score the corpus with the teacher -> pairs.jsonl"),
    "distill": ("awembed.distill", "train the student against teacher margins + labels"),
    "quantize": ("awembed.quantize", "weight-only int8 export, fidelity-gated"),
    "eval": ("awembed.evaluate", "teacher / baseline / student / int8 on the held-out split"),
}
PIPELINE = ("capture", "distill", "quantize", "eval")


def _dispatch(stage: str, rest: Sequence[str]) -> int:
    modname, _ = STAGES[stage]
    mod = importlib.import_module(modname)
    argv0 = sys.argv[0]
    sys.argv = [f"awembed {stage}", *rest]
    try:
        return int(mod.main() or 0)
    except SystemExit as exc:  # the stages exit with their gate verdict
        code = exc.code
        if isinstance(code, int):
            return code
        if code:
            print(code, file=sys.stderr)
            return 1
        return 0
    finally:
        sys.argv[0] = argv0


def _run(rest: Sequence[str]) -> int:
    """Run the four artifact stages in order against one --out root; stop at the
    first stage whose gate refuses. `rest` is passed to every stage (they share the
    --corpus/--seed/--out contract and ignore what they do not use)."""
    for stage in PIPELINE:
        print(f"[awembed] stage {stage}", flush=True)
        code = _dispatch(stage, rest)
        if code != 0:
            print(f"[awembed] stage {stage} refused (exit {code}); stopping", flush=True)
            return code
    print("[awembed] done: all stages passed their gates", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="awembed", description=__doc__.splitlines()[0],
                                 epilog="Use `awembed <stage> --help` for a stage's own flags.")
    ap.add_argument("--version", action="store_true", help="print the version and exit")
    sub = ap.add_subparsers(dest="stage")
    for name, (_, help_) in STAGES.items():
        sub.add_parser(name, help=help_, add_help=False)
    sub.add_parser("run", help="capture -> distill -> quantize -> eval", add_help=False)
    # Split at the stage name: everything after it belongs to the stage's own parser.
    head, rest = argv, []
    for i, a in enumerate(argv):
        if a in STAGES or a == "run":
            head, rest = argv[: i + 1], argv[i + 1:]
            break
    ns = ap.parse_args(head)
    if ns.version:
        from awembed import __version__
        print(__version__)
        return 0
    if not ns.stage:
        ap.print_help()
        return 2
    if ns.stage == "run":
        return _run(rest)
    return _dispatch(ns.stage, rest)


if __name__ == "__main__":
    sys.exit(main())
