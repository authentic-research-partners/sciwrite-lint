# Evaluation Framework

sciwrite-lint ships an evaluation framework for verifying detection quality and scoring accuracy. Run eval commands via `python -m evals`.

## Prerequisites

- **Synthetic + SciLint Score evals**: vLLM running locally
- **Real-world eval**: GROBID + vLLM for full pipeline; `--judge` additionally requires Claude CLI (Sonnet adjudication)
- **Calibration**: vLLM + calibration PDFs downloaded

## Evaluation tiers

| Tier | Command | What it answers |
|------|---------|-----------------|
| **Synthetic** | `python -m evals eval-synthetic` | Does each check detect what it should? (P/R/F1 per check) |
| **Real-world inject** | `python -m evals eval-real-world inject` | Does detection work on messy real papers? (recall on LaTeX + PDF) |
| **Real-world FPR** | `python -m evals eval-real-world fpr` | Do text rules produce false positives on clean real papers? (Sonnet-judged) |
| **Real-world pipeline** | `python -m evals eval-real-world pipeline` | Does the full pipeline survive papers it has never seen? |
| **Real-world pipeline + judge** | `...pipeline --judge` | Which findings are wrong, and why? (Sonnet TP/FP verdicts) |
| **SciLint Score eval** | `python -m evals eval-scilint-score` | Do the scoring components classify correctly? |
| **Calibration** | `python -m evals eval-calibration` | Does the tool rank known papers correctly? |

### When to run each

- **Synthetic**: every code change — fast (<1s deterministic, ~30s with LLM)
- **Real-world inject**: after check refactors — tests recall on messy real LaTeX and PDFs
- **Real-world FPR**: after text rule changes — fast FPR check (no vLLM/GROBID needed, just Sonnet)
- **Real-world pipeline**: after major changes — full end-to-end on unseen papers
- **Real-world pipeline + judge**: periodic — every FP verdict is a bug report to investigate
- **SciLint Score eval**: after prompt or taxonomy changes — validates classification accuracy
- **Calibration**: after prompt or weight changes — validates scoring via ordinal constraints

## Synthetic eval

Each case is a LaTeX document with known issues (or known clean). LLM-backed checks are warmed concurrently: the runner deduplicates cases by `(tex_content, figure_descriptions)` hash and fires `run_llm_checks_batched` for every unique paper inside a single `asyncio.run`, so vLLM sees one big burst of concurrent queries instead of a sequence of per-case calls. After warm-up, each per-case assertion reads from an in-memory cache. See `evals/synthetic.py::_warm_llm_cache`.

```bash
python -m evals eval-synthetic                         # all checks
python -m evals eval-synthetic --checks dangling-cite  # specific checks
```

### Adding cases

Case generators live in `evals/gen_*.py` modules. Each generator returns a list of `SyntheticCase` objects (defined in `evals/synthetic_types.py`):

```python
SyntheticCase(
    name="unique_case_name",
    check_id="dangling-cite",
    description="What this case tests",
    tex_content="...",  # complete LaTeX document
    expected=[ExpectedFinding(rule_id="dangling-cite", context="orphan_key")],
)
```

For realistic cases, use `build_realistic_paper()` from `evals/synthetic_templates.py` — it generates a full CS paper where you override individual sections to inject errors.

## Synthetic corpus (cross-format)

A small set of self-contained manuscripts, each demonstrating one issue the linter detects — a dangling citation, an unreferenced figure, a dangling cross-reference, a prose error, a percentage set that doesn't add up, a footnote URL, an uncited bibliography entry, numeric `[1]` citations, a multi-file bibliography — plus clean controls. Every scenario can be rendered to LaTeX, Markdown, and PDF from a single abstract spec, so the *same* content can be checked through each front-end (the LaTeX parser, pandoc, and GROBID). Nothing is committed; the corpus is generated on demand.

Materialize it to inspect or lint yourself (no services needed for `tex`/`md`; `pdf` needs `pdflatex`):

```bash
python -m evals synth-corpus --out corpus                  # tex + md
python -m evals synth-corpus --out corpus --format tex,md,pdf
sciwrite-lint check corpus/dangling_cite/dangling_cite.md  # lint one
```

It writes a `MANIFEST.md` listing every scenario and the checks it is built to trigger.

Two coverage evals run the checks on the rendered corpus and report what fires per format:

```bash
python -m evals eval-synth-corpus            # deterministic checks via PDF round-trip (needs pdflatex + GROBID)
python -m evals eval-synth-corpus --llm      # LLM-backed checks on tex/md (needs text vLLM)
```

PDF coverage is reported at the rule level: rendering to PDF loses the source's symbolic citation keys and figure labels, so a cite to a missing entry (which renders as `[?]`) cannot be recovered — that is reported as a known gap rather than a regression. LLM coverage is recall-level, since LLM-backed checks are non-deterministic; each format's output is checked for the expected issue, so the same prose should surface the same checks whether authored as LaTeX or Markdown.

## Real-world eval

Downloads real papers from arXiv (LaTeX) and bioRxiv (PDF), exercises the full stack against messy real data and live APIs. This is how integration bugs are found — synthetic eval and unit tests cannot catch them.

```bash
python -m evals eval-real-world corpus -n 100          # download papers
python -m evals eval-real-world inject --max-papers 10 # inject errors (LaTeX + PDF), measure detection
python -m evals eval-real-world fpr --max-papers 5     # text-rule FPR (Sonnet-judged, no vLLM/GROBID)
python -m evals eval-real-world pipeline --max-papers 5 # full pipeline on 5 papers
python -m evals eval-real-world pipeline --judge        # + Sonnet judges findings as TP/FP
python -m evals eval-real-world pipeline --concurrency 4 # max papers in concurrent stages (default: 2, validated up to 4)
python -m evals eval-real-world report                 # show results
```

The pipeline eval uses **batch-by-stage orchestration**: GPU-heavy stages (vision, embedding, cited vision) run in a single subprocess per batch — loading each model once for all papers. Non-GPU stages (vLLM checks, network fetch, GROBID parsing) run all papers concurrently. This avoids VRAM contention while maximizing GPU utilization.

The `--judge` workflow: run pipeline → Sonnet judges each finding → investigate every FP → trace to root cause → fix → rerun. Every FP verdict is a bug report.

## SciLint Score eval

Tests SciLint Score components on expert-labeled cases:

```bash
python -m evals eval-scilint-score                     # all axes (requires vLLM)
python -m evals eval-scilint-score --axes taxonomy     # claim taxonomy only
python -m evals eval-scilint-score --axes laudan       # Laudan only
```

- **Claim taxonomy** (22 cases): per-dimension classification accuracy
- **Laudan** (8 cases): problem-solving effectiveness scoring

## Calibration

Select papers with known quality. Define expected ordinal rankings. Iterate on prompts and weights until the system delivers those rankings. Runs the full pipeline (including claim verification, reference internal checks, and bibliography verification) via batch-by-stage orchestration — same GPU coordination as the real-world pipeline eval.

See [docs/calibration.md](calibration.md).

