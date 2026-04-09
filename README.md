# sciwrite-lint

A linter for scientific manuscripts. Checks that your references exist, your metadata is accurate, your citations support the claims you make, and your cited papers' own bibliographies are real. Works on LaTeX and PDF. Runs entirely on your machine. Produces a **SciLint Score**.

The only open-source tool that combines reference verification, claim checking, manuscript consistency, figure analysis, and bibliography verification in one pipeline — powered entirely by a local LLM on your own GPU.

## Why

AI writing tools produce text that *looks* like good science — fluent prose, correct formatting, plausible-sounding citations. But they don't verify whether the references are real, whether the cited papers actually say what you claim, or whether the numbers in your abstract match your results.

**sciwrite-lint does.** It checks your references against academic databases, downloads the cited papers, verifies that they actually say what you claim, and follows one level deeper to check your references' own bibliographies. Fully local — no manuscripts leave your machine.

## Features

**22 automated checks:**

### Reference verification
- **Do your references exist?** — checked against CrossRef, OpenAlex, Semantic Scholar, Open Library, and Library of Congress
- **Is the metadata accurate?** — title, authors, year, venue compared against canonical records
- **Are any retracted?** — every DOI cross-referenced against 60,000+ entries in the Retraction Watch database
- **Robust matching** — when references lack DOIs, a multi-signal matching engine scores candidates across title, author, year, and venue (handles the metadata errors that LLMs routinely introduce)

### Claim verification (local LLM)
- **Do cited papers support your claims?** — downloads full text from 8 open-access sources (arXiv, Semantic Scholar, OpenAlex, PubMed Central, Europe PMC, Unpaywall, bioRxiv/medRxiv, CORE), parses via GROBID, embeds sections, and verifies each claim against the actual source text
- **What role does each citation play?** — classifies citation purpose (evidence, contrast, method, attribution, context…) with graduated weights: an unsupported *evidence* citation is serious; an unsupported *context* citation barely matters
- **Are your references' own bibliographies real?** — batch-checks cited papers' reference lists for existence, metadata accuracy, and retraction. Papers built on fabricated evidence are flagged

### Manuscript consistency (local LLM)
- **Cross-section contradictions** — numbers, claims, and framing that drift between sections
- **Numbers vs. tables** — text claims that contradict the corresponding table or figure
- **Arithmetic and percentages** — stated totals that don't match components; percentages that don't sum to 100%
- **Sample size tracking** — N values that change across sections without explanation
- **Causal language** — unhedged causal claims in correlational studies
- **Abstract–body alignment** — abstract makes factual claims the body contradicts
- **Statistical reporting** — p-values vs. their verbal interpretation
- **Structure promises** — contributions promised in the introduction but never delivered

### Figure checks (vision model + LLM)
- **Caption vs. content** — does the caption match what the figure actually shows?
- **Text vs. figure** — does the text describe the figure accurately?
- **Axis labels** — units, labels, and scales consistent with the text
- **Figure–table agreement** — same data in a figure and table should agree

### Text checks (deterministic, no services needed)
- **Dangling citations** — `\cite{key}` with no matching bib entry
- **Dangling cross-references** — `\ref{X}` with no matching `\label{X}`
- **Unreferenced figures** — figures included but never referenced in the text

### Per-reference reliability score
All signals — metadata, retraction status, claim support, consistency, bibliography health — aggregate into a single reliability score per reference. When multiple independent checks flag the same reference, it is flagged as unreliable with specific reasons.

### SciLint Score

```
SciLint Score = Internal Consistency × Referencing Quality × Contribution
```

A single number combining:
- **Internal Consistency** — fraction of checks passed within the manuscript
- **Referencing Quality** — are references real, accurate, and do they support your claims? Each reference weighted by its reliability score and citation purpose
- **Contribution** *(experimental)* — five axes from philosophy of science (Popper, Lakatos, Kitcher, Laudan, Mayo): empirical content, progressiveness, unification, problem-solving effectiveness, test severity. Defaults to 1.0 until `contributions` runs

```bash
sciwrite-lint contributions paper.pdf --format json
```

### Privacy and security

- **Your manuscript never leaves your machine.** All parsing, LLM inference, and figure analysis run locally
- **Only citation metadata is sent externally** — DOIs, titles, author names for API verification. No paper content
- **Open-weights models pinned to specific versions** — results are reproducible forever, not dependent on a cloud provider's API updates
- **No API keys required** — all verification uses free public databases. Optional keys increase rate limits

### Two audiences

- **Humans** — colored terminal output with severity levels, locations, and explanations. Decide in seconds whether each finding is real
- **AI writing agents** — `--format json` output with structured fields (`level`, `rule_id`, `message`, `context`). Run sciwrite-lint in a write → check → fix → recheck loop. Configurable exit codes for CI integration

### Optimizations

Three models — a vision model (Qwen3-VL-2B), an embedding model (Arctic Embed), and an 8B reasoning LLM (Qwen3 via vLLM) — share a single consumer GPU. Models run in separate pipeline stages; on WSL2, CUDA memory virtualization pages idle allocations to system RAM, letting all three share physical VRAM. FP8 weights and KV cache (Ada Lovelace+) and per-paper SQLite caching with hash-based invalidation are baseline. On top of that:

- **Semantic section filtering** — embedding-based KNN retrieval sends only the ~5 most relevant sections per claim to the LLM, reducing LLM calls ~4x
- **Prefix-first prompt structure** — shared context placed before variable content in all prompts, maximizing vLLM's automatic prefix caching
- **Per-call-site thinking budgets** — each LLM call site has empirically tuned (max_tokens, thinking_preset) pairs, measured via grid search to maximize detection quality
- **Adaptive embedding batches** — token-aware batch sizing gives ~50x speedup over CPU while staying within VRAM limits
- **Batch-staged multi-paper pipeline** — when checking 2+ papers, GPU models load once per batch (vision/embedding/cited-vision) and vLLM/network stages run concurrently, giving a meaningful speedup over sequential per-paper runs. Tune via `--concurrency` (default 2, validated up to 4 on a single consumer GPU)
- **Phased API resolution** — citations flow through OpenAlex → Semantic Scholar → CrossRef → Open Library/LoC, each phase only processing what previous phases didn't resolve
- **8-source full-text cascade** — early exit on first successful download across arXiv, Semantic Scholar, OpenAlex, PubMed Central, Europe PMC, Unpaywall, bioRxiv, CORE
- **Live monitoring** *(advanced)* — `sciwrite-lint containers monitor` shows service health, VRAM usage, and KV cache utilization in a terminal dashboard

Full pipeline on a 50-reference paper: ~30 minutes initial (dominated by downloads and claim verification), minutes on cached reruns. On native Linux, embedding and vision default to CPU (tens of times slower) due to limited VRAM on consumer GPUs and no transparent paging of inactive vLLM container allocations; force GPU via config if VRAM permits.

## Install

**Assumed setup:** A workstation with an NVIDIA GPU (16+ GB VRAM). Developed and tested on Windows (WSL2). Native Linux is likely to work with GPU memory allocation tuning (see [docs/services.md](docs/services.md#embedding-device)). Not tested on macOS.

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/), a container runtime (podman or docker), CUDA drivers, and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
uv tool install sciwrite-lint --python 3.13
```

uv downloads Python 3.13 automatically (does not affect any Python you may already have) and installs sciwrite-lint as a globally available command.

## Setup

```bash
sciwrite-lint init                     # scaffold .sciwrite-lint.toml + references/ + local_pdfs/
sciwrite-lint config set-email you@example.com  # required for Unpaywall + Retraction Watch
sciwrite-lint containers start         # start GROBID + vLLM (needs GPU for vLLM)
sciwrite-lint containers monitor       # live dashboard: service health, VRAM, KV cache
```

![Monitor dashboard](docs/monitor.png)

`init` detects `.tex` files and their `.bib` references and generates a `.sciwrite-lint.toml` config. Review to confirm the right files were detected.

**Paywalled references:** drop PDFs into `local_pdfs/` with filenames matching the reference title. The tool fuzzy-matches filenames against your `.bib` titles and uses local copies instead of downloading.

**Optional API keys** increase rate limits for Semantic Scholar, NCBI, and CORE:

```bash
sciwrite-lint config show              # see what's configured
sciwrite-lint config set-key semantic-scholar YOUR_KEY  # dedicated rate limit
```

See [docs/services.md](docs/services.md) for GPU requirements, all external APIs, and API key details.

## Usage

```bash
sciwrite-lint check --paper my_paper            # full pipeline + SciLint Score
sciwrite-lint check --paper my_paper --fresh    # same, ignoring all caches
sciwrite-lint check                             # all papers (batch-staged when 2+, ~4-5x faster than sequential)
sciwrite-lint check --concurrency 4             # batch parallelism (default 2, validated up to 4)
sciwrite-lint check paper.tex                   # text + LLM rules on a .tex file
sciwrite-lint check paper.pdf                   # check a PDF (GROBID required)
sciwrite-lint contributions --paper my_paper    # add contribution axes to SciLint Score
sciwrite-lint contributions paper.pdf           # standalone file scoring
```

`check` runs the full pipeline in one command: text checks → figure analysis → LLM consistency → reference verification via APIs → download and parse cited papers → claim verification → consistency checks on cited papers → bibliography verification → aggregate reliability scores → SciLint Score. An initial run on a 50-reference paper takes up to 30 minutes (dominated by downloads and claim verification); subsequent cached runs complete in minutes.

Use `--fresh` to start from scratch (backs up the existing workspace before recreating it).

### Contribution axes (`sciwrite-lint contributions`)

`contributions` computes 5 contribution axes from philosophy of science (Popper, Lakatos, Kitcher, Laudan, Mayo) and updates the SciLint Score. Requires vLLM.

```bash
sciwrite-lint check --paper my_paper           # SciLint Score (contribution = 1.0)
sciwrite-lint contributions --paper my_paper   # add 5 contribution axes, update score
```

### Individual stages

For debugging or advanced workflows, each pipeline stage is also available as a standalone command:

| Command | What it does |
|---------|-------------|
| `verify --paper NAME` | API verification only (CrossRef, OpenAlex, Semantic Scholar, Open Library, Library of Congress) |
| `fetch --paper NAME` | Download full-text PDFs for verified references |
| `parse --paper NAME` | Parse PDFs via GROBID and compute embeddings |
| `verify-claims --paper NAME` | LLM reads cited sources, checks claim support |
| `ref-health --paper NAME` | Fast reference health check: cite/bib mismatches, ID coverage, local PDF matches (no API calls) |
| `contributions --paper NAME` | Add 5 contribution axes to SciLint Score (requires vLLM) |

## Output

Each finding has a severity level, a rule ID, and a message explaining the issue:

- **error** — a concrete manuscript problem (hallucinated reference, unsupported claim, retracted source)
- **warning** — needs human judgment (metadata mismatch, weak citation purpose, cross-section inconsistency)
- **info** — the tool could not complete a check (LLM error, API timeout, internal crash) or informational note

Findings also carry a `context` field with the reasoning behind the verdict: the LLM's explanation, which identifiers were searched, or which API provided the canonical data. This lets you distinguish "the tool found a problem" from "the tool couldn't check this."

Example terminal output:

```
 ERROR  reference-exists        johnson2024: Not found in any API
                                Searched with: title="Deep Learning for Climate", author="Johnson"

 ERROR  claim-support           smith2023: Cited paper does not support this claim
                                Claim: "transformers outperform RNNs by 15% on BLEU"
                                Verdict: paper reports 8% improvement, not 15%

 WARN   reference-accuracy      lee2022: Year mismatch (bib: 2022, canonical: 2021)
                                Source: OpenAlex (DOI: 10.1234/example)

 WARN   cross-section-consistency
                                Abstract claims "three novel contributions" but
                                Section 5 delivers two

 WARN   reference-unreliable    chen2019: Low reliability (0.35)
                                Metadata mismatch, 2 unsupported claims,
                                23% hallucinated bibliography entries

 INFO   caption-vs-content      Figure 3: could not extract figure from PDF

  SciLint Score: 0.41
    Internal Consistency:  0.85
    Referencing Quality:   0.48
  (Run 'sciwrite-lint contributions' to add contribution axes)
```

Output formats: terminal (default) or `--format json`.

## Documentation

- `sciwrite-lint checks` — list all checks
- `sciwrite-lint <command> --help` — detailed usage for any command
- [docs/services.md](docs/services.md) — GROBID, vLLM, external APIs, configuration

For contributors and advanced users:

- [docs/evals.md](docs/evals.md) — detection evaluation framework, adding test cases
- [docs/calibration.md](docs/calibration.md) — SciLint Score calibration against ground-truth papers

## License

MIT
