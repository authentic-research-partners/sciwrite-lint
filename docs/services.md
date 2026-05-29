# Services

sciwrite-lint requires two local services and uses several external APIs. Start both before running the linter.

**Assumed setup:** A workstation with an NVIDIA GPU (16+ GB VRAM) and at least 32 GB system RAM. Tested on WSL2 with NVIDIA driver 546.01+; native Linux uses the same GPU-sequencing code path with no extra configuration (see [Embedding device](#embedding-device) below), though it isn't actively tested. Not tested on macOS.

## Quick start

```bash
sciwrite-lint containers start         # start both GROBID + vLLM
sciwrite-lint containers status        # check if running, show logs
sciwrite-lint containers stop          # stop both
sciwrite-lint containers restart       # stop + start both
sciwrite-lint containers start --update  # pull latest images, then start
sciwrite-lint containers monitor       # live dashboard (KV cache, latency, throughput, RAM)
sciwrite-lint containers monitor -i 5  # refresh every 5s (default: 2s)
sciwrite-lint containers restart --recreate  # recreate containers (applies config changes)
```

## GROBID (PDF parsing)

[GROBID](https://github.com/kermitt2/grobid) extracts structured text from PDFs — sections, references, inline citations. Required for `sciwrite-lint check paper.pdf` and `sciwrite-lint parse`.

```bash
sciwrite-lint grobid start     # pulls image + starts container on port 8070
sciwrite-lint grobid status    # check if running
sciwrite-lint grobid stop      # stop container
```

Requires podman or docker. No GPU needed. Uses ~2 GB RAM at idle, up to 8 GB when parsing many PDFs concurrently.


## vLLM (local LLM)

A local language model for checks that require reading comprehension: `cross-section-consistency`, `structure-promises`, `cite-purpose`, `claim-support`, and the 11 full-paper consistency checks.

Default model: Qwen3 8B. Alternative: Gemma 3 12B. Vision model: Qwen3-VL 8B FP8.

```bash
# Start (downloads model on first run)
sciwrite-lint vllm start

# Start with a specific model
sciwrite-lint vllm start --model gemma3

# Start vision model (Qwen3-VL-8B-FP8 on port 5002)
sciwrite-lint vllm start --model qwen3-vl

# Check status
sciwrite-lint vllm status

# View logs
sciwrite-lint vllm logs

# Stop
sciwrite-lint vllm stop
```

Requires an NVIDIA GPU with at least 16 GB VRAM, CUDA drivers, and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Runs via podman/docker with `--device nvidia.com/gpu=all` (CDI).

### Concurrency: `[llm] max_concurrency` and `[vision] max_concurrency`

`sciwrite-lint check` issues many vLLM requests in parallel — full-paper consistency, ref-internal, and claim verification each batch dozens to hundreds of queries. The vision pipeline does the same with image requests against a separate vLLM container on port 5002. To keep either server from queueing the whole batch at once (which would cross the per-request timeout on the slowest-to-schedule request), the client caps in-flight requests per batch:

```toml
[llm]
max_concurrency = 12        # text-vLLM (port 5001), default 12

[vision]
max_concurrency = 64        # vision-vLLM (port 5002), default 64
request_timeout = 120       # seconds; vision responses are smaller than text
```

**Why a client-side cap matters.** The OpenAI-compatible API exposes no signal that distinguishes "your request is queued at the server" from "your request is being processed." The read timeout (300 s for text, 120 s for vision) starts ticking the moment the request leaves your machine and runs through queue wait, prefill, and decode together. If you flood vLLM with hundreds of large requests, the slowest one waits minutes for a scheduling slot, and the timeout fires *before* it ever runs. Capping concurrency in our process holds the surplus requests locally with no clock running, so the timeout only ever covers actual server-side work.

**Why the text and vision defaults differ.** Text full-paper queries can be ~30 K tokens each, so the text cap is set conservatively to keep KV-cache pressure manageable on a single mid-sized GPU; the heaviest in-codebase caller (reference-internal full-paper checks) drives this cap. Smaller-prompt callers (manuscript LLM checks, claim taxonomy, ref-internal pairwise) run at higher hardcoded caps internally — see "code-internal callers" below. Vision requests are uniformly small (~3.6 K tokens including the image embedding), so the vision cap can be much higher without saturating the multimodal KV cache.

**When to tune:**

- **Lower** (e.g. text → 4, vision → 32) if you see `APITimeoutError` mid-run, or if you run multiple papers concurrently (`--concurrency N` in `check`) on a single GPU — the cap is per-batch, so multi-paper mode multiplies it.
- **Raise** if you have a larger GPU and want shorter wall time. For text, 16–32 is reasonable on small-prompt workloads (LLM checks, claim taxonomy) — code-internal callers already pass per-batch hints so the global only binds the heavy full-paper case. Push the vision cap upward in similar increments and watch for saturation symptoms.

If you see a run abort with `vLLM at http://localhost:5001/v1 became unreachable mid-request: APITimeoutError`, the symptom is almost always saturation under heavy prompts, not a too-large single prompt. Lower `max_concurrency` (or `--concurrency` for paper-level batches) before raising `llm_timeout`.

## External APIs

Used by `sciwrite-lint verify` and `sciwrite-lint fetch`. Manage credentials with `sciwrite-lint config`.

### Polite email (required for full-text download)

Some services treat the email as **required** (the API will reject requests without it); others as an **optional polite-pool identifier** (works without, but you get slower/lower-priority service). `sciwrite-lint check` and `sciwrite-lint fetch` refuse to run without `polite_email` because their full-text stage depends on Unpaywall.

```bash
sciwrite-lint config set-email you@example.com
```

This writes `polite_email` into your project's `.sciwrite-lint.toml` under `[api]`.

- **Unpaywall**: **required** — API refuses requests without an email. No email → no open-access lookups at all.
- **Retraction Watch**: **required** — CSV download endpoint rejects requests without an email. No email → no retraction checks.
- **CrossRef**: **optional but recommended** — works without an email but puts you on a shared public pool (slower, 1 concurrent request). With email you join the polite pool (3 concurrent, priority queue).

### API keys (optional, except NASA ADS)

Most keys are *optional* — they raise rate limits but the source works without them. **NASA ADS is the exception:** the source is skipped entirely if no key is configured.

| Service | Status | Get a token at | Effect of setting it |
|---------|--------|----------------|----------------------|
| Semantic Scholar | Optional | [semanticscholar.org/product/api#api-key](https://www.semanticscholar.org/product/api#api-key) | Dedicated rate limit (faster) |
| NCBI (PubMed Central) | Optional | [ncbi.nlm.nih.gov/account/settings/](https://www.ncbi.nlm.nih.gov/account/settings/) | PMC rate 3 → 10 req/s |
| CORE | Optional | [core.ac.uk/services/api](https://core.ac.uk/services/api) | Full-text PDF downloads from institutional repos |
| NASA ADS | **Required for source** | [ui.adsabs.harvard.edu/user/settings/token](https://ui.adsabs.harvard.edu/user/settings/token) (sign in free, go to *Settings → API Token*) | Enables the NASA ADS source (5000 req/day cap) |

**Setup (two ways, either works):**

Via CLI (recommended):

```bash
sciwrite-lint config set-key semantic-scholar YOUR_KEY
sciwrite-lint config set-key ncbi YOUR_KEY
sciwrite-lint config set-key core YOUR_KEY
sciwrite-lint config set-key nasa-ads YOUR_TOKEN
```

By writing the file directly:

```bash
mkdir -p ~/.sciwrite-lint
echo "YOUR_TOKEN" > ~/.sciwrite-lint/nasa_ads_api_key
chmod 600 ~/.sciwrite-lint/nasa_ads_api_key
```

Check status or get signup links at any time:

```bash
sciwrite-lint config show          # lists configured/missing keys + signup URL for each
sciwrite-lint config remove-key <service>  # delete a key
```

Keys are stored in `~/.sciwrite-lint/` with 0600 permissions (owner-read only) and never committed — they live outside your project directory so they don't get swept up by `git add`.

### API reference

| API | Used for | Batching |
|-----|----------|----------|
| [CrossRef](https://www.crossref.org/) | Reference existence + metadata verification | No — one query per request. Strict concurrency limits: 1 concurrent (public), 3 concurrent (with `polite_email`) |
| [OpenAlex](https://openalex.org/) | Reference existence + metadata verification; batch DOI verification at depth-2 | Yes — 200 DOIs per request (pipe-separated filter) |
| [Semantic Scholar](https://www.semanticscholar.org/) | Reference existence + full-text URLs; batch arXiv/PMID verification at depth-2 | Yes — 500 IDs per request (`/paper/batch` endpoint) |
| [Open Library](https://openlibrary.org/) | Book/monograph existence verification; direct ISBN lookup when available | No — one query per request |
| [Library of Congress](https://www.loc.gov/) | Book/report existence verification; direct LCCN lookup when available (170M+ items, broadest catalog coverage) | No — one query per request |
| [PubMed Central](https://www.ncbi.nlm.nih.gov/pmc/) | Full-text PDF download | No — one DOI per request |
| [Europe PMC](https://europepmc.org/) | Full-text PDF download | No — one DOI per request |
| [Unpaywall](https://unpaywall.org/) | Open-access PDF URLs (requires `polite_email`) | No — one DOI per request |
| [NBER](https://www.nber.org/) | Economics working-paper title search (public JSON API) | No — one title query per request |
| [RePEc / IDEAS](https://ideas.repec.org/) | Economics working-paper title search (HTML scrape of search results + paper landing pages) | No — one title query per request |
| [HAL](https://hal.science/) | Title search for French national open archive (Solr JSON API; multi-discipline, ~4.5M docs) | No — one title query per request |
| [ERIC](https://eric.ed.gov/) | Title search for US Dept of Education research (JSON API; `ED*` documents only, ~660K OA docs) | No — one title query per request |
| [NASA ADS](https://ui.adsabs.harvard.edu/) | Title search for astronomy/astrophysics (JSON API, bearer token required — see API keys below; 5000 req/day cap) | No — one title query per request |
| [OSF Preprints](https://osf.io/preprints/) | Title search for cross-discipline preprints (SocArXiv, PsyArXiv, ChemRxiv, EarthArXiv, EngrXiv, MetaArXiv; ~190K preprints) | No — one title query per request |
| [CORE](https://core.ac.uk/) | Full-text PDF download (API key recommended) | No — one DOI per request |
| [Retraction Watch](https://www.crossref.org/labs/retraction-watch/) | Retraction and expression-of-concern detection (via CrossRef Labs CSV, requires `polite_email`) | N/A — full CSV cached locally; in-memory lookups |

At depth-2, bibliography entries from parsed references are batch-verified: DOIs via OpenAlex (200 per request), arXiv IDs and PMIDs via Semantic Scholar batch endpoint (500 per request), remaining entries via OpenAlex title search.

The Retraction Watch database (~60,000+ entries) is downloaded as a CSV and cached locally (`~/.sciwrite-lint/retraction-watch.csv`), refreshed every 24 hours.

## Security & Privacy

### What stays local

- **Your manuscript text** — never sent to any external service. Parsed locally, checked locally.
- **Full-text PDFs of cited papers** — downloaded to `references/{paper_name}/` (per-paper workspace) and processed by GROBID (local container). Never re-uploaded.
- **LLM inference** — runs locally via vLLM. No manuscript or reference text is sent externally.

### What leaves your machine

| Data | Sent to | Why |
|------|---------|-----|
| DOIs, arXiv IDs, PMIDs, ISBNs, LCCNs | CrossRef, OpenAlex, Semantic Scholar, Open Library, Library of Congress | Verify reference existence and metadata |
| Paper titles and author surnames | Same APIs (search queries) | Find references that lack structured identifiers |
| Paper titles and author surnames (first author only) | NBER, RePEc/IDEAS, HAL, ERIC, NASA ADS, OSF Preprints | Locate open-access versions of references in economics, French HSS/CS, US education research, astronomy, and cross-discipline preprint servers. First-author surname is used as a server-side filter (HAL, NASA ADS, ERIC) or free-text ranking hint (NBER, IDEAS, OSF) to disambiguate candidates when titles are short or generic. |
| Your polite email | CrossRef, Unpaywall, Retraction Watch | Required for API access / polite pool |
| API keys (if configured) | Semantic Scholar, CORE, NCBI, NASA ADS | Higher rate limits; NASA ADS requires a key to use the source at all |

All external API calls use HTTPS. Note: while no manuscript text is sent, the pattern of reference queries (titles, authors, DOIs) could allow an API provider or network observer to infer your research topic.

### Protections against malicious content

sciwrite-lint downloads untrusted content (PDFs, HTML, XML) from external sources. The pipeline includes:

- **XML parsing** uses `defusedxml` — blocks entity-expansion attacks (XXE, billion laughs) on GROBID TEI output
- **Download size limits** — 10 MB for HTML, 100 MB for PDFs, 200 MB for the Retraction Watch CSV
- **SSRF protection** — URL redirects (both HTTP 3xx and meta-refresh/JS) are validated against DNS-resolved IP addresses; requests to private/reserved IPs are blocked
- **Identifier validation** — DOIs, arXiv IDs, and other identifiers are format-validated before use in API URLs
- **No code execution** — fetched content is never passed to `eval`, `exec`, `pickle`, or similar
- **LLM output treated as data** — vLLM responses are JSON-parsed and read as strings/numbers; never passed to `eval`, `exec`, used in file paths, shell commands, or SQL queries. The LLM runs after all external API calls and downloads are complete — its output never feeds back into network requests
- **Prompt injection mitigations** — manuscript and cited paper text sent to vLLM is wrapped in XML delimiters (`<document>`, `<source_section>`) to separate trusted instructions from untrusted content. System prompts include anti-injection instructions directing the model to treat document content as data, not instructions. All LLM responses use strict JSON schema enforcement, constraining output to predefined structures
- **Container isolation** — GROBID and vLLM run in podman/docker containers, not as host processes

### API key storage

Keys are stored in `~/.sciwrite-lint/` with owner-read-only permissions. sciwrite-lint warns on startup if key files are readable by other users.

## Configuration

Container settings are optional in `.sciwrite-lint.toml`:

```toml
[containers]
grobid_memory = "8g"               # RAM limit (default: 8g)
vllm_memory = "4g"                 # RAM limit, GPU VRAM is separate (default: 4g)
# grobid_version = "0.8.2.1-crf"  # override GROBID version
# vllm_version = "v0.18.0"        # override vLLM version
```

Both container images are pinned to specific versions. Use `--update` to re-pull if you override the version in config.

### Embedding device

Embeddings are computed during `sciwrite-lint parse` and `sciwrite-lint check` (Stage 4). By default, sciwrite-lint uses the GPU if CUDA is available (`device = "auto"`).

```toml
[embeddings]
device = "auto"    # "auto" (GPU if available), "cpu", "cuda"
```

GPU embedding is ~50x faster than CPU (seconds vs. minutes for a typical paper).

**GPU memory sharing with vLLM:** The embedding model (~1.2 GB) and vLLM share a single GPU. The pipeline stops the text vLLM container before embedding runs and restarts it after, on **both WSL2 and native Linux** — neither platform tolerates concurrent active CUDA workloads on a 16–24 GB consumer GPU. Encode batch size adapts to chunk token length to stay within VRAM limits.

- **Windows (WSL2):** swap adds ~30 s per pipeline run for the embedding window. **Recommended setup:** in NVIDIA Control Panel → 3D Settings → Manage 3D Settings → *CUDA - Sysmem Fallback Policy*, choose **Prefer No Sysmem Fallback**. By default Windows silently spills GPU overflow into system RAM via PCIe (30–100× slower than VRAM); disabling the fallback makes WSL2 fail loudly on overflow instead of running degraded. Driver 546.01+.
- **Native Linux:** same swap path. CUDA OOM errors are loud by default — no setting to change.

To force CPU embedding on any platform, set `device = "cpu"`. This avoids any GPU memory interaction but is ~50x slower.

### Vision model (figure analysis)

Two backends for figure description:

| Backend | Model | How it runs | Accuracy |
|---------|-------|-------------|----------|
| `transformers` (default) | Qwen3-VL-2B-Instruct | In-process subprocess, no container | 85% caption mismatch |
| `vllm` | Qwen3-VL-8B-Instruct-FP8 | vLLM container on port 5002 | 100% caption mismatch |

The default (2B transformers) needs no extra setup. The 8B vLLM backend gives +15% detection on real-world caption mismatches but requires GPU time-sharing — the pipeline automatically swaps containers when enabled.

```toml
# .sciwrite-lint.toml
[vision]
backend = "vllm"    # or "transformers" (default)
```

```bash
# Or via CLI flag (overrides config)
sciwrite-lint check --paper my-paper --vision-backend vllm

# Standalone
sciwrite-lint vision --paper my-paper --backend vllm
```

**GPU memory sharing (transformers backend):** The 2B VL subprocess sequences with text vLLM the same way the embedder does — stop text vLLM, run vision in subprocess, restart text vLLM. Same on WSL2 and native Linux.

**Container swap (vllm backend):** The pipeline stops text vLLM, starts vision vLLM (~90-130s), runs all vision stages, then swaps back. Skips the swap if the vision container is already running. Pre-start with `sciwrite-lint containers start --vision` to avoid the wait during pipeline runs.

**GPU embedding swap (all platforms):** The pipeline automatically stops text vLLM before the embedding stage, runs embedding on GPU in a subprocess (~50× faster than CPU), then restarts text vLLM. Transparent — no configuration needed. WSL2 and native Linux behave the same way.

**Caching:** Figure descriptions are cached in workspace.db by SHA-256 of image bytes + caption text. Only changed or new images trigger re-inference. A typical manuscript (5-8 figures) takes ~35s on first run, 0s on subsequent runs.

**Requires:** `poppler-utils` installed system-side (for TikZ figure rendering from compiled PDFs). All Python dependencies are included in `pip install sciwrite-lint`.

Can also run standalone: `sciwrite-lint vision --paper my-paper`.
