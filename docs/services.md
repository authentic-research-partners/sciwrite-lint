# Services

sciwrite-lint requires two local services and uses several external APIs. Start both before running the linter.

**Assumed setup:** A workstation with an NVIDIA GPU (16+ GB VRAM). Developed and tested on WSL2 with 64 GB RAM and an RTX 4000 Ada (20 GB VRAM). Native Linux is likely to work but is untested and may require GPU memory allocation tuning (see [Embedding device](#embedding-device) below). Not tested on macOS.

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

Default model: Qwen3 8B. Alternative: Gemma 3 12B.

```bash
# Start (downloads model on first run)
sciwrite-lint vllm start

# Start with a specific model
sciwrite-lint vllm start --model gemma3

# Check status
sciwrite-lint vllm status

# View logs
sciwrite-lint vllm logs

# Stop
sciwrite-lint vllm stop
```

Requires an NVIDIA GPU with at least 16 GB VRAM, CUDA drivers, and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Runs via podman/docker with `--gpus all`.

## External APIs

Used by `sciwrite-lint verify` and `sciwrite-lint fetch`. Manage credentials with `sciwrite-lint config`.

### Polite email (required)

CrossRef, Unpaywall, and Retraction Watch require a polite email. Without it, `check` and `fetch` will refuse to run.

```bash
sciwrite-lint config set-email you@example.com
```

This writes `polite_email` into your project's `.sciwrite-lint.toml` under `[api]`. Benefits:

- **CrossRef**: polite pool (faster rate limits, 3 concurrent vs 1)
- **Unpaywall**: required for open-access PDF lookup (will not work without email)
- **Retraction Watch**: required for retraction database download (will not work without email)

### API keys (optional)

Optional keys increase rate limits. Check current status and get signup links:

```bash
sciwrite-lint config show
```

Set a key:

```bash
sciwrite-lint config set-key semantic-scholar YOUR_KEY  # dedicated rate limit
sciwrite-lint config set-key ncbi YOUR_KEY              # PMC: 3 → 10 req/s
sciwrite-lint config set-key core YOUR_KEY              # CORE: institutional repos
```

Keys are stored in `~/.sciwrite-lint/` with 0600 permissions (owner-read only). Remove with `sciwrite-lint config remove-key <service>`.

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
| [CORE](https://core.ac.uk/) | Full-text PDF download (API key recommended) | No — one DOI per request |
| [Retraction Watch](https://www.crossref.org/labs/retraction-watch/) | Retraction and expression-of-concern detection (via CrossRef Labs CSV, requires `polite_email`) | N/A — full CSV cached locally; in-memory lookups |

At depth-2, bibliography entries from parsed references are batch-verified: DOIs via OpenAlex (200 per request), arXiv IDs and PMIDs via Semantic Scholar batch endpoint (500 per request), remaining entries via OpenAlex title search.

The Retraction Watch database (~60,000+ entries) is downloaded as a CSV and cached locally (`~/.sciwrite-lint/retraction-watch.csv`), refreshed every 24 hours.

## Security & Privacy

### What stays local

- **Your manuscript text** — never sent to any external service. Parsed locally, checked locally.
- **Full-text PDFs of cited papers** — downloaded to `references/{paper_name}/` (per-paper workspace) and processed by GROBID (local container). Never re-uploaded.
- **LLM inference** — runs locally via vLLM. No text sent externally unless you explicitly use `--backend claude`.

### What leaves your machine

| Data | Sent to | Why |
|------|---------|-----|
| DOIs, arXiv IDs, PMIDs, ISBNs, LCCNs | CrossRef, OpenAlex, Semantic Scholar, Open Library, Library of Congress | Verify reference existence and metadata |
| Paper titles and author surnames | Same APIs (search queries) | Find references that lack structured identifiers |
| Your polite email | CrossRef, Unpaywall, Retraction Watch | Required for API access / polite pool |
| API keys (if configured) | Semantic Scholar, CORE, NCBI | Higher rate limits |

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

**GPU memory sharing with vLLM:** The embedding model (~1.2 GB) and vLLM share a single GPU. The pipeline runs embedding and vLLM inference in separate sequential stages, never at the same time. GPU memory is explicitly released between stages. Encode batch size adapts to chunk token length to stay within VRAM limits.

- **Windows (WSL2):** GPU memory is virtualized — the OS transparently pages vLLM's idle memory while embedding runs. This is the tested and recommended platform. No configuration needed.
- **Native Linux:** GPU memory is physical (no overcommit). `device = "auto"` defaults to CPU. To enable GPU embedding, set `device = "cuda"` and ensure enough free VRAM (lower vLLM's `--gpu-memory-utilization` from 0.9 if needed).

To force CPU embedding on any platform, set `device = "cpu"`. This avoids any GPU memory interaction but is ~50x slower.

### Vision model (figure analysis)

The vision model (Qwen3-VL-2B-Instruct, ~4 GB float16) extracts structured descriptions from manuscript figures. It runs in Stage 0.5, before LLM consistency checks, so figure descriptions are available for caption-vs-content, text-vs-figure, and other figure checks. Also runs on cited paper PDFs during Stage 4.2.

**GPU memory sharing:** Same pattern as embeddings — on WSL2, CUDA memory overcommit lets the VL model share VRAM with vLLM transparently. On native Linux, auto-resolves to CPU.

**Caching:** Figure descriptions are cached in workspace.db by SHA-256 of image bytes + caption text. Only changed or new images trigger re-inference. A typical manuscript (5-8 figures) takes ~35s on first run, 0s on subsequent runs.

**Requires:** `poppler-utils` installed system-side (for TikZ figure rendering from compiled PDFs). All Python dependencies are included in `pip install sciwrite-lint`.

Can also run standalone: `sciwrite-lint vision --paper paper_a`.
