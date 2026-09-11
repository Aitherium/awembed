# awembed — Aither World Embed

<!-- aither-header:start GENERATED from the ecosystem registry. Edits here are overwritten; change the registry instead. -->

**[Docs](https://aitherium.github.io/awembed/)**  ·  [Source](https://github.com/Aitherium/awembed)  ·  `pip install awembed`  ·  [The Aither World](https://aitherium.github.io/)

> **The Aither World** is an operating system for agents — a Linux you can hand to one, the runtimes it works in, and the tools it works with. [awnix](https://github.com/Aitherium/awnix) is the Linux underneath it; **awembed** is one of its 46 bricks — each installs on its own, runs offline, and needs no account.
>
> **Start here:** Point it at one repo and get a 0.6B embedder that ranks your directories better than the 7B one it learned from, with the eval that proves it.

<!-- aither-header:end -->

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

## Prove it against the others

`awembed eval` scores your student against the teacher it learned from. `awembed compare`
scores **any served embedders** on **your** documents — black-box, over OpenAI-shaped
`/v1/embeddings`, so the student and every third-party model are measured by the same
code on the same rows:

```bash
awembed compare --corpus docs.jsonl --queries queries.jsonl \
  --endpoint student=http://127.0.0.1:18101 \
  --endpoint nomic=http://127.0.0.1:18103 \
  --qprefix nomic="search_query: " --dprefix nomic="search_document: " \
  --out compare.json
```

Prefixes are per endpoint, because the conventions differ and they move the numbers.

**Measured 2026-09-10** on 26 real chunks of a customer's engineering documents
(6 files), 28 queries — 7 real questions users had asked of that corpus, 21 written for
even coverage. Metrics are document-level (a hit is the top-ranked chunk's document), so
a chunker cannot flatter or punish a model:

| endpoint | dims | p@1 | doc@3 | MRR |
|---|---|---|---|---|
| student (this tool's output) | 1024 | **0.893** | 1.000 | **0.940** |
| Qwen3-Embedding-0.6B (raw) | 1024 | 0.821 | 0.964 | 0.900 |
| nomic-embed-text-v1.5 (`search_*` prefixes) | 768 | 0.821 | 1.000 | 0.899 |
| all-MiniLM-L6-v2 (the 384-d fallback that was serving that corpus) | 384 | 0.750 | 0.964 | 0.851 |
| student truncated to 256-d (the volunteer-compute form) | 256 | 0.821 | 1.000 | 0.905 |

Two findings worth carrying: a student distilled for **code** search transferred to
**prose** — the code-search query prefix changed nothing on documents either way — and
truncating to 256-d costs real questions (p@1 0.714 → 0.429), so keep the full width for
retrieval and use the narrow form only where it is verifying agreement, not ranking.

Caveat, stated: 28 queries over 6 documents makes a two-query gap directional, not
significant. Rerun it on your own corpus before you switch anything.

## Licence

Apache-2.0. Models you train carry their base model's licence.

<!-- aither-ecosystem:start GENERATED from the ecosystem registry. Edits here are overwritten; change the registry instead. -->

## The aw family

Standalone tools that share one idea: **replace something you would otherwise have to _trust_ with something you can _check_.**

Each installs on its own, works offline, and needs no account.

| | instead of trusting | you check |
|---|---|---|
| [awdk](https://github.com/Aitherium/awdk) | a framework's idea of how your agents should run | one loop you can read, pointed at a backend you already pay for |
| [awskills](https://github.com/Aitherium/awskills) | that an agent knows your procedure | the procedure written down, versioned, and loadable by any agent |
| [awpack](https://github.com/Aitherium/awpack) | that the pack you want shipped inside somebody's SDK, under whatever licence that SDK happens to carry | the pack as its own versioned artifact, with its own licence, that any agent runtime can install |
| [awm](https://github.com/Aitherium/awm) | that memory stayed in its lane | tenant:user:project scopes, so a write cannot cross a boundary |
| [awnode](https://github.com/Aitherium/awnode) | a vendor's cloud with every prompt | a local gateway routing to backends you chose |
| [awgraph](https://github.com/Aitherium/awgraph) | that grep found everything | an AST + tree-sitter call graph an agent can traverse |
| [awgit](https://github.com/Aitherium/awgit) | that no one else is editing this file | a lease, refused at commit time if you do not hold it |
| [awdelphi](https://github.com/Aitherium/awdelphi) | one agent's confident take on a decision | the round trace, the anonymity, and who dissents |
| [awtoll](https://github.com/Aitherium/awtoll) | that your tooling is saving you context | the measured token cost of each tool call, and what the alternative cost |
| [awseal](https://github.com/Aitherium/awseal) | that the artifact came from who you think | an Ed25519 seal — the key that verifies is not the key that forges |
| [awshare](https://github.com/Aitherium/awshare) | that the download is intact | content-addressed bundles, verified on fetch |
| [awnest](https://github.com/Aitherium/awnest) | that there is a person on the other end | a verdict with evidence, where "we could not tell" is not "yes" |
| [awnboard](https://github.com/Aitherium/awnboard) | a share link anyone who sees it can use | an invitation addressed to one person, for one gate, revocable |
| [awnix](https://github.com/Aitherium/awnix) | that the box is what you left it as | an immutable image you built, with atomic rollback |
| [awrecover](https://github.com/Aitherium/awrecover) | that the restore worked | a restore that fully lands or does not land at all |
| [awstorage](https://github.com/Aitherium/awstorage) | a du you ran last month, and a peers file that says 3 TB free | an inventory snapshot per node with a diff since the last one, and each tree classified re-fetchable or not |
| [awrelay](https://github.com/Aitherium/awrelay) | a SaaS in the middle of your agents | findings, alerts and coordination over your own transport |
| [awask](https://github.com/Aitherium/awask) | that anyone read the paragraph where you asked | the ask itself, with a button that steers the session that raised it |
| [awmail](https://github.com/Aitherium/awmail) | a mailbox somebody else can read | mail your agents send and receive over your own server |
| [awfind](https://github.com/Aitherium/awfind) | one vendor's idea of the web | results from whichever providers you configured |
| [awbrowse](https://github.com/Aitherium/awbrowse) | that the page said what you were told | the render, the DOM and the requests it made |
| [gawbbonet](https://github.com/Aitherium/gawbbonet) | the model to keep a 300-message campaign coherent by itself | campaign facts recalled from scoped memory you can list and edit |
| [aitherkvcache](https://github.com/Aitherium/aitherkvcache) | a vendor's quantisation defaults | sub-byte KV cache kernels you can benchmark yourself |
| [awrtifact](https://github.com/Aitherium/awrtifact) | a hand-rolled split script and a hand-edited worker manifest | byte-verified parts in a release, served with Range + CORS, sizes asserted by a live gate |
| [AitherZero](https://github.com/Aitherium/AitherZero) | a pile of scripts nobody has numbered | numbered, discoverable automation with declarative playbooks |
| [AitherConnect](https://github.com/Aitherium/AitherConnect) | what a page tells your browser to do | a federated search and desktop bridge you host |
| [awreason](https://github.com/Aitherium/awreason) | a confident paragraph | the phases it went through, and every tool call it made to get there |
| [awrecurse](https://github.com/Aitherium/awrecurse) | that everything you pasted in was actually read | which slices it opened, and what it concluded from each |
| [awprism](https://github.com/Aitherium/awprism) | the first explanation that fits | the ranked alternatives, and the observation that separates them |
| [awrepl](https://github.com/Aitherium/awrepl) | what the agent believes the value is | the value, printed from the live session |
| [awresearch](https://github.com/Aitherium/awresearch) | a summary of pages nobody opened | every claim against the source it came from |
| [awfocus](https://github.com/Aitherium/awfocus) | twelve terminal tabs and a bad memory | one command that names every session, finds any transcript, and opens or steers the one you want |
| [awgym](https://github.com/Aitherium/awgym) | that a world model learned anything from the games it saw | transitions captured from real play, fed back, and the retrodiction score falling on grids it never saw |
| [awpredict](https://github.com/Aitherium/awpredict) | a model because it trained without erroring | its prediction against a self-updating lookup, on the rows that are actually novel |
| [awsh](https://github.com/Aitherium/awsh) | that you already know the name of the command | what it decided your line meant, before it acts on it |
| [awkno](https://github.com/Aitherium/awkno) | that the docs site is up, or that you remember the family | the whole ecosystem in your terminal, with no network at all |
| **awembed** _(you are here)_ | a general-purpose embedder that has never seen your code | a held-out split of whole directories, scored teacher vs student vs int8 |
| [awtax](https://github.com/Aitherium/awtax) | a closed tax app's sealed file you can never read again | a plain, provider-neutral schema of every figure, with the page it came from |
| [awsettings](https://github.com/Aitherium/awsettings) | that you will remember to re-approve the same thing on every box you work from | one profile, unioned rather than overwritten, with the credentials left behind |

[**awnix**](https://github.com/Aitherium/awnix) is the ground floor — A Linux you can hand to an agent — immutable base, capabilities included.

## The Aitherium ecosystem

Every repository here is public. Each publishes an `aither-manifest.json` beside its page, so any surface can read every sibling's — the network is browsable from any node in it.

| repo | what it is | pages |
|---|---|---|
| [awdk](https://github.com/Aitherium/awdk) | Build AI agent fleets — 3 lines, any backend, local or cloud | [docs](https://aitherium.github.io/awdk/) |
| [awskills](https://github.com/Aitherium/awskills) | Portable agent skills — self-contained procedures an agent loads on demand | [docs](https://aitherium.github.io/awskills/) |
| [awpack](https://github.com/Aitherium/awpack) | First-party agent packs — the ones we build, versioned and installable on their own | [docs](https://aitherium.github.io/awpack/) |
| [awm](https://github.com/Aitherium/awm) | A portable, scoped agent memory | [docs](https://aitherium.github.io/awm/) |
| [awnode](https://github.com/Aitherium/awnode) | A lightweight local gateway — bridges your apps to the AI backends you chose | [docs](https://aitherium.github.io/awnode/) |
| [awrun](https://github.com/Aitherium/awrun) | A priority-aware queue and dispatcher for agentic runs and ad-hoc CI builds. It also judges whether the runner pool is big enough for the queue it is draining, and can ask a host to grow it -- reserving capacity is zero-sum, so a saturated pool needs more of it, not a different share of it | [docs](https://aitherium.github.io/awrun/) |
| [awgraph](https://github.com/Aitherium/awgraph) | A semantic code graph for agents — AST + tree-sitter, call graphs | [docs](https://aitherium.github.io/awgraph/) |
| [awgit](https://github.com/Aitherium/awgit) | Semantic version control on top of git — edit-ops and leases | [docs](https://aitherium.github.io/awgit/) |
| [awdelphi](https://github.com/Aitherium/awdelphi) | Anonymous multi-round expert panels — a converged answer with a trace | [docs](https://aitherium.github.io/awdelphi/) |
| [awtoll](https://github.com/Aitherium/awtoll) | What every tool call costs you in context, measured from your own transcripts | [docs](https://aitherium.github.io/awtoll/) |
| [awseal](https://github.com/Aitherium/awseal) | Sign an artifact so a stranger can verify it | [docs](https://aitherium.github.io/awseal/) |
| [awshare](https://github.com/Aitherium/awshare) | Publish an artifact and fetch it back verified | [docs](https://aitherium.github.io/awshare/) |
| [awdit](https://github.com/Aitherium/awdit) | An append-only audit trail whose gaps are DETECTABLE | [docs](https://aitherium.github.io/awdit/) |
| [awbac](https://github.com/Aitherium/awbac) | Role-based access control that fails closed and explains itself | [docs](https://aitherium.github.io/awbac/) |
| [awiam](https://github.com/Aitherium/awiam) | Who is this caller? A directory and session store that fails honestly | [docs](https://aitherium.github.io/awiam/) |
| [awtunnel](https://github.com/Aitherium/awtunnel) | Reach a service that has no public address | [docs](https://aitherium.github.io/awtunnel/) |
| [awnest](https://github.com/Aitherium/awnest) | Prove there is a human before you let them into the nest | [docs](https://aitherium.github.io/awnest/) |
| [awnboard](https://github.com/Aitherium/awnboard) | A front gate you can put in front of anything, and hand someone the key to | [docs](https://aitherium.github.io/awnboard/) |
| [awnix](https://github.com/Aitherium/awnix) | A Linux you can hand to an agent — immutable base, capabilities included | [docs](https://aitherium.github.io/awnix/) |
| [awrecover](https://github.com/Aitherium/awrecover) | Labelled snapshots with an all-or-nothing restore | [docs](https://aitherium.github.io/awrecover/) |
| [awstorage](https://github.com/Aitherium/awstorage) | Every drive on every node, indexed, classified and diffed -- so you can see what you own before you delete it | [docs](https://aitherium.github.io/awstorage/) |
| [awrelay](https://github.com/Aitherium/awrelay) | Portable agent messaging — findings, alerts, coordination | [docs](https://aitherium.github.io/awrelay/) |
| [awask](https://github.com/Aitherium/awask) | Your agent asks you a question — and acts on your answer | [docs](https://aitherium.github.io/awask/) |
| [awmail](https://github.com/Aitherium/awmail) | Give an agent an email address — send, and actually receive | [docs](https://aitherium.github.io/awmail/) |
| [awnet](https://github.com/Aitherium/awnet) | The agentic web — agents host a mesh, and agents join one | [docs](https://aitherium.github.io/awnet/) |
| [awfind](https://github.com/Aitherium/awfind) | A portable search client — query, results, ranking | [docs](https://aitherium.github.io/awfind/) |
| [awbrowse](https://github.com/Aitherium/awbrowse) | A portable browser client — navigate, console, network, DOM, screenshot | [docs](https://aitherium.github.io/awbrowse/) |
| [awknowledge](https://github.com/Aitherium/awknowledge) | How to run a coding agent so the result survives — the laws, with evidence | [docs](https://aitherium.github.io/awknowledge/) |
| [gawbbonet](https://github.com/Aitherium/gawbbonet) | GobboNet campaigns with a real agent brain — scoped memory, graph recall | [docs](https://aitherium.github.io/gawbbonet/) |
| [aitherkvcache](https://github.com/Aitherium/aitherkvcache) | Near-optimal KV cache quantization for LLM inference — sub-byte compression | [docs](https://aitherium.github.io/aitherkvcache/) |
| [awrtifact](https://github.com/Aitherium/awrtifact) | Deliberately chunk artifacts into GitHub release assets — the productized aitherkvcache mirror lane | [docs](https://aitherium.github.io/awrtifact/) |
| [AitherZero](https://github.com/Aitherium/AitherZero) | PowerShell 7+ automation framework — numbered, self-describing scripts | [docs](https://aitherium.github.io/AitherZero/) |
| [AitherConnect](https://github.com/Aitherium/AitherConnect) | Browser extension — federated AI search, page context, and the Living OS overlay | [docs](https://aitherium.github.io/AitherConnect/) |
| [awreason](https://github.com/Aitherium/awreason) | A portable reasoning client — sessions, phases, thoughts, and the chain that produced the answer | [docs](https://aitherium.github.io/awreason/) |
| [awrecurse](https://github.com/Aitherium/awrecurse) | Answer a question over a context far larger than the window — recursively, with the trace kept | [docs](https://aitherium.github.io/awrecurse/) |
| [awprism](https://github.com/Aitherium/awprism) | Turn a failure into ranked hypotheses — and say what would confirm each one | [docs](https://aitherium.github.io/awprism/) |
| [awrepl](https://github.com/Aitherium/awrepl) | A REPL an agent can actually use — state that survives between turns | [docs](https://aitherium.github.io/awrepl/) |
| [awresearch](https://github.com/Aitherium/awresearch) | Ask a research question, get a cited report you can check | [docs](https://aitherium.github.io/awresearch/) |
| [awfocus](https://github.com/Aitherium/awfocus) | See, search and steer every Claude session from one command | [docs](https://aitherium.github.io/awfocus/) |
| [awgym](https://github.com/Aitherium/awgym) | An ARC training gym — a game a world model can watch, and six roles that play through it | [docs](https://aitherium.github.io/awgym/) |
| [awpredict](https://github.com/Aitherium/awpredict) | Predict what your environment does next, and how surprised you were | [docs](https://aitherium.github.io/awpredict/) |
| [awsh](https://github.com/Aitherium/awsh) | Your terminal answers you -- type a question where a command would go | [docs](https://aitherium.github.io/awsh/) |
| [awkno](https://github.com/Aitherium/awkno) | The man page for the Aither World — every brick, stack and law, offline | [docs](https://aitherium.github.io/awkno/) |
| **awembed** _(you are here)_ | Train an embedding model that knows your corpus, and prove it beats the big one | [docs](https://aitherium.github.io/awembed/) |
| [awtax](https://github.com/Aitherium/awtax) | Turn any tax PDF -- returns, W-2, 1099, statements, even scans -- into structured data you can check | [docs](https://aitherium.github.io/awtax/) |
| [awsettings](https://github.com/Aitherium/awsettings) | Your agent's permissions and config, following you to the next machine | [docs](https://aitherium.github.io/awsettings/) |

<div id="aither-constellation" data-self="awembed"></div>
<script src="aither-constellation.js"></script>

<!-- aither-ecosystem:end -->
