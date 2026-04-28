# Changelog

## [0.3.0] — 2026-04-28

### Added
- **Public Python API for open-access acquisition.** `download_pdf`, `fetch_web`, and `search_by_title` are importable from `sciwrite_lint`, giving scripts and reference managers direct access to the OA chain without running the linter. See [docs/python-api.md](docs/python-api.md).
- **Six more open-access sources.** The full-text downloader now also searches NBER (US economics), RePEc/IDEAS (European and non-US economics), HAL (French national archive), ERIC (US education), NASA ADS (astronomy), and OSF Preprints (SocArXiv, PsyArXiv, ChemRxiv). Useful for papers whose OA copy lives in a discipline-specific repository rather than arXiv or the journal's OA tier.
- **NASA ADS API key.** Configure with `sciwrite-lint config set-key nasa-ads <TOKEN>` (free token at https://ui.adsabs.harvard.edu/user/settings/token). Without a key the source is skipped with a one-time notice.
- **`prose-quality` check.** Grammar and semantic word-choice review at sentence level — flags syntactic errors and wrong-word slips that style checkers miss (e.g. "comprise of" → "consist of"). Hedged language, passive voice, and stylistic preference are deliberately not flagged.
- **`sciwrite-lint check --checks ID[,ID...]` flag.** Run a comma-separated subset of checks instead of disabling every other one in `.sciwrite-lint.toml`. Unknown IDs fail loudly with exit code 2.
- **Local web captures and MHTML archives.** A new `local_web_dir` drop folder accepts `.md` summaries and `.mhtml` browser saves; ingest converts MHTML to markdown via trafilatura. Lets reference material from JavaScript-heavy pages participate in claim verification. See [docs/local-sources.md](docs/local-sources.md).
- **Footnote-URL claims are now verified.** Claims backed by `\footnote{\url{...}}` rather than `\cite{key}` are matched against `local_web_dir` archives and verified against the captured text — same SUPPORTS / NOT_SUPPORTED / CANNOT_DETERMINE verdict as cited references. URLs without a capture are skipped silently. LaTeX-only for now.
- **Hash-aware re-ingest of drop-folder sources.** Replacing a PDF or MHTML in `local_pdfs_dir` / `local_web_dir` under the same filename now triggers a fresh copy + GROBID re-parse on the next run. Previously you had to use `--fresh` or delete the workspace file.
- **Per-paper `local_pdfs_dir`.** Each paper can supply its own curated-PDF directory in its `[[papers]]` TOML block; otherwise auto-detected from `Sources/full_text/` next to `file_path`.
- **Pre-download candidate ranking for title-search OA sources.** NBER / RePEc / HAL / ERIC / NASA ADS / OSF now compare every plausible candidate to the bib evidence (title similarity + surname overlap + year) and pick the best — or reject all of them — before any bytes are downloaded.
- **Server-side author filtering** on HAL, NASA ADS, and ERIC. NBER, IDEAS, and OSF use the surname as a free-text ranking hint where no field filter is documented.
- **Self-consistency voting infrastructure for LLM checks.** `local-llm` checks can opt in via an `n_samples` attribute; the batch runner forwards it to vLLM as the OpenAI `n` parameter. `prose-quality` ships at `n_samples=1`; flipping to 3 turns voting on as a one-line change.
- **Per-check temperature for LLM checks.** Categorical judgments (`prose-quality`) run at `temperature=0.0` for reproducibility; voting regimes get `~0.3` for sample diversity. Other checks remain at the default 0.6.
- **Pandoc-based LaTeX cleaning at manuscript ingest.** Citations become `[CITE]`, cross-references `[REF]`, math renders as readable Unicode, and stranded punctuation from removed references is cleaned up before LLM checks see it. `pypandoc-binary` ships pandoc inside the wheel — no system-level pandoc install required.
- **`search_by_title` returns richer per-hit metadata.** The public API now populates `title`, `authors`, and `year` per hit (when the source exposes them) and returns every candidate per source rather than the first hit.
- **Typed exceptions for Python-API consumers.** `SciWriteLintError` and `LLMConnectionError` let agents catch specific error classes instead of matching `RuntimeError` strings.
- **Standard 0/1/2 exit codes.** Matching `ruff` / `mypy`: `0` clean, `1` findings at error level, `2` tool error. Previously a dead vLLM could exit `0` with empty findings.

### Improved
- **Uncited `.bib` entries are no longer verified or fetched.** The working set is now filtered to keys actually referenced from `\cite*` / `\nocite` (or `.aux` if compiled). Users with a shared lab-wide `references.bib` should see noticeably fewer API calls. `dangling-cite` still reports structural issues from the raw bib.
- **Vision-vLLM container no longer restarts on reruns when figures haven't changed.** Pre-flight probes now hash exactly the figures they would process, including TikZ rendered from the compiled PDF. Removes ~2–3 minutes of container-restart overhead per rerun.
- **Unavailable references no longer re-queried on every run.** Definitive negatives (every OA source reached and reported no match) are cached with a timestamp and skipped on subsequent runs. Cache lifetime configurable via `[api] fetch_retry_ttl_days` (default 30); transient failures still retry; `--fresh` bypasses.

### Fixed
- **Web-resource references are no longer reported as dead unless the server positively confirms the URL is gone.** Only HTTP 404/410 → "dead" (ERROR); 4xx refusals, 5xx, TLS / timeout / decoding errors → "blocked" (WARNING — manual verify). The verifier also sends browser-like headers to bypass mainstream WAF refusals. Run `sciwrite-lint check --paper <name> --fresh` once to re-classify existing workspaces.
- **Multi-signal PDF match validation replaces the single-title check.** Fetched PDFs are validated against title similarity + first-page surname / DOI / year + ERIC / CORE landing-page detection. Run `--fresh` once to re-validate cached PDFs.
- **Short generic titles no longer false-accept the wrong paper.** When the bib lists authors, the validator now requires at least one bib surname on the PDF's first two pages before accepting on title similarity alone.
- **Unicode / non-English titles no longer mangled in Solr queries.** Titles sent to HAL, ERIC, and NASA ADS now pass through a Solr-reserved-character escape that leaves every Unicode code point untouched. French, German, Chinese, Vietnamese, and Cyrillic titles are now queried as written.
- **Local PDFs can now be matched by citekey prefix.** Filenames starting with a known bib key (e.g. `smith2020_Paper_Title.pdf`) match by the citekey directly. Fixes a length-bias for short bib titles under the `{citekey}_{Title}.pdf` archival convention.
- **Manual tier overrides no longer get re-verified on every run.** `api_match="manual"` (set by `sciwrite-lint override`) and the new synthetic footnote-URL citations now short-circuit via the verify cache, matching the CLI help.
- **Book-length references no longer hit a hardcoded 300-second embedding timeout.** The timeout now scales with input size — short docs still get the 300s floor; long policy reviews and books get up to 4500s.
- **Full-paper consistency checks no longer stall on thinking-budget overruns.** `llm_query` now adds the active thinking budget on top of `max_tokens` instead of cannibalizing it; on `finish_reason=length` the retry steps thinking down (high → medium → low → off) instead of repeating the same doomed call.
- **TikZ-only manuscripts no longer crash the vision stage when `vision_backend = "vllm"`.** The pre-flight check now detects TikZ / vector figure environments and triggers the container swap.
- **Cited-reference vision stage no longer crashes with `asyncio.run() cannot be called from a running event loop`.** Cited vision now runs in a subprocess, isolating both `transformers` (CUDA) and vLLM (asyncio) cleanly.
- **Configured `vision_backend` is never silently downgraded.** The cited-vision subprocess path now honors the `vision_backend` setting end-to-end.
- **Downloads from servers that gzip binary content** (e.g. MPRA working-paper PDFs returning `Content-Encoding: gzip` for already-compressed files) now decode correctly.
- **`sciwrite-lint vision` exits `2` on tool failure** instead of `1`, matching the 0/1/2 convention.
- **Standalone `verify`, `status`, `fetch`, and `verify-claims`** now find local reference PDFs in the per-paper workspace; previously these commands scanned the top-level `references/` directory and missed already-downloaded PDFs. The full `check` pipeline was unaffected.
- **`sciwrite-lint verify-claims`** now computes claim-query embeddings on demand when run standalone — previously returned `CANNOT_DETERMINE` for every claim.
- **`python -m evals eval-real-world`** auto-discovers `.sciwrite-lint.toml` and preflight-checks the email config, instead of failing 10 minutes in.
- **Unparseable cited PDFs no longer abort the whole paper's fetch.** A bad PDF is now rejected individually; the OA chain tries the next source and the rest of the paper's references keep processing. The classified reason ("image-only PDF", "unsupported PDF producer", etc.) appears per reference in `verify` output and `check --format json`.

## [0.2.2] — 2026-04-14

### Added
- **Paper citation.** Accompanying arXiv preprint ([arXiv:2604.08501](https://arxiv.org/abs/2604.08501)) is now surfaced in the README, `CITATION.cff`, and PyPI project sidebar.
- **`requirements-pinned.txt`** for reproducing the exact versions the maintainer tests against. Regular installs are unchanged.

### Improved
- **Open-access references flagged for manual download.** When a reference is confirmed open access but the publisher blocks programmatic downloads, `sciwrite-lint verify` now lists it separately with a direct URL and instructions to save to `local_pdfs/`. Previously these were mixed in with truly unavailable references in the T2 summary.

## [0.2.1] — 2026-04-10

### Added
- **Higher-quality vision option.** Choose between the default Qwen3-VL-2B model and a larger Qwen3-VL-8B (FP8, via vLLM) for figure analysis. Select via `--vision-backend` or the `[vision]` section of `.sciwrite-lint.toml`.
- **Automatic container lifecycle.** The pipeline starts, swaps, and stops text and vision vLLM containers as needed, skipping swaps when all figures are already cached.
- **GPU embeddings on native Linux** *(preliminary)*: embeddings run on GPU automatically when available.

### Changed
- **Faster end-to-end runs.** Vision processing now runs concurrently, claim-verification query embeddings are pre-computed, and redundant container swaps are avoided.
- **Consistency checks return a curated shortlist.** `full-paper-consistency` reports up to 5 findings per check and `cross-section-consistency` up to 4, ranked by the model. Papers with few issues are unchanged; papers with many issues now get the most important ones instead of an unbounded list.
- **Figure readability issues are itemized.** Each problem with a figure is reported as its own bullet instead of one combined note, so downstream consistency checks see one issue at a time.

### Reliability
- **Long-paper consistency analysis is now robust under heavy finding load.** Results from consistency and full-paper checks are always well-formed, even when the model identifies many issues in a single pass.

## [0.2.0] — 2026-04-09

Initial public release (alpha), accompanying [arXiv:2603.17893](https://arxiv.org/abs/2603.17893).

See [README.md](README.md) for features and usage.
