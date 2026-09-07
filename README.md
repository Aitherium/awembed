# awembed — Aither World Embed

Train a small embedding model that knows **your** corpus, and prove it beats the big one.

A 0.6B student trained this way on one codebase retrieved the right directory on the
first try 80% of the time; the 7.85B general-purpose teacher it learned from managed 66%
on the same held-out questions. The student ships as 1.06 GB of int8 with 0.999 cosine
fidelity to its full-precision self. The whole run was one GPU for about two and a half
hours.

The reason a small model can win is not magic: the teacher never saw your corpus. The
student is trained on two signals at once — the teacher's *margins* between the right
answer and plausible wrong ones (distillation), and your corpus's own labels
(in-domain contrastive learning). For retrieval over something you own, in-domain
supervision is worth more than parameters.

## Install

```bash
pip install awembed              # the student side: torch + transformers
pip install "awembed[teacher]"   # also the teacher server (fastapi, einops, ...)
```

## The recipe

```bash
# 0. Build the corpus from per-directory READINGS of your repo: one JSON per line,
#    {"rel": "<dir>", "reading": "<a paragraph on what the dir does>", "prompt_version": "v1"}.
#    Any LLM over `ls` + a few file heads produces them (~1,500 chars each is plenty);
#    a walker that writes them for you is the next thing this module grows.
#    Output: one row per question, with the right directory and K hard negatives --
#    directories that look plausible and are wrong -- split by DIRECTORY, so
#    evaluation never sees a directory that training saw. Add your own
#    question -> target pairs with --ground-truth (JSONL {"query", "target"}).
awembed corpus --readings dir_readings.jsonl --out corpus.jsonl

# 1. Before renting anything: does the teacher load and answer HERE?
awembed probe                                  # exit 0 ok / 1 failed / 2 could not judge

# 2. Capture the teacher's judgement for every row (starts the teacher server itself)
awembed capture  --corpus corpus.jsonl --seed 1 --out artifacts/

# 3. Distill the student (fp32 master + bf16 autocast; gate: the loss must fall)
awembed distill  --corpus corpus.jsonl --seed 1 --out artifacts/

# 4. Weight-only int8 export (gate: >= 0.98 mean cosine to the fp32 student)
awembed quantize --corpus corpus.jsonl --seed 1 --out artifacts/

# 5. Teacher vs untrained baseline vs student vs int8, on the held-out directories
awembed eval     --corpus corpus.jsonl --seed 1 --out artifacts/

# ...or all four artifact stages in order, stopping at the first gate that refuses:
awembed run      --corpus corpus.jsonl --seed 1 --out artifacts/
```

Every stage writes a sidecar (`teacher_manifest.json`, `train_sidecar.json`,
`quant_sidecar.json`, `eval_report.json`) with what it measured, so the record of a run
is the run. Every stage has a `--self-test` that runs the same code path on a thumbnail
model, on CPU, offline.

## What the gates refuse

| stage | refuses when |
|---|---|
| capture | a corpus row lacks inline hard-negative text (path-only negatives teach a shortcut, not retrieval) |
| distill | the mean loss over the last 10 steps is not below the first 10, or any step is non-finite |
| quantize | the int8 export embeds at < 0.98 mean cosine to the fp32 student, measured through the loader the eval stage uses |
| eval | the student does not beat the baseline on both p@1 and recall@10, or int8 holds < 98% of the student's recall@10 |

A run that "completed" with no learning is exactly the thing a training pipeline is
best at hiding. The gates are the point.

## Defaults, and what to change

- **Teacher**: `nvidia/NV-Embed-v2` (7.85B, CC-BY-NC-4.0 — a teacher signal only; its
  weights and its captured targets never ship). Any OpenAI-compatible `/v1/embeddings`
  endpoint works: pass `--teacher-url` to `capture` instead of letting it start one.
- **Student**: `Qwen/Qwen3-Embedding-0.6B` (Apache-2.0, 1024-dim, last-token pooling).
  Pass `--student` to `distill` for another base.
- **Asymmetry**: queries carry an instruction prefix; documents embed plain. The prefix
  the student was trained with is the one you must use at query time. It is written in
  the sidecar.
- **Two environments**: the default teacher pins an older `transformers` than the
  student needs, so run `teacher`/`probe` in their own venv and point `capture` at it
  with `NV_EMBED_PYTHON=/path/to/venv/bin/python`.

## Using the student

The student is an ordinary Hugging Face directory (`artifacts/student/`) — serve it
with vLLM (`--task embed`), Text Embeddings Inference, or `transformers` directly. It is
a **new vector space** (1024-dim): index into a fresh collection, do not mix it with
vectors from the model you replaced.

It pairs with the rest of the Aither World family: `awgraph` (code graph search over
your repo), `awm` (agent memory), `awfind` (ranked answers), `awrecurse` (documents
larger than a context window), and `awdk` agents that consume any of those.

## Licence

Apache-2.0. Models you train carry their base model's licence.
