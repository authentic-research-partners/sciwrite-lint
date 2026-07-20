# Python API

sciwrite-lint exposes its open-access acquisition capabilities as a small public Python API. Useful for scripts, reference-manager integrations, or any workflow that needs OA-source coverage without running the full linter.

Everything is async. The public surface is four Pydantic models and three functions, importable from the package root:

```python
from sciwrite_lint import (
    download_pdf, fetch_web, search_by_title,
    FetchConfig, DownloadResult, WebResult, SearchHit,
)
```

These are network operations — no vLLM is needed. By default `download_pdf` validates each fetched PDF against the bibliographic details you supply (title / authors / DOI / year); that match check uses the GROBID container, so GROBID must be running whenever you pass those details. Supplying only an identifier — or passing `require_title_match=False` (see below) — skips the match check, and the download then needs no local services: it accepts any well-formed PDF (`%PDF` magic bytes + a sane size). `fetch_web` and `search_by_title` never need local services.

## download_pdf — identifiers/title → PDF on disk

```python
import asyncio
from pathlib import Path
from sciwrite_lint import download_pdf, FetchConfig

async def main():
    result = await download_pdf(
        Path("/tmp/attention.pdf"),
        arxiv_id="1706.03762",
        title="Attention Is All You Need",
        email="you@example.com",
        config=FetchConfig(timeout=30.0),
    )
    if result.found:
        print(f"Got {result.source} → {result.out_path}")
    else:
        print("All sources exhausted:")
        for reason in result.failed_sources:
            print(f"  - {reason}")

asyncio.run(main())
```

Tries these sources in priority order, returning on the first accepted PDF (validated against the bib evidence unless `require_title_match=False`):

1. arXiv (requires `arxiv_id`)
2. Semantic Scholar openAccessPdf (requires `s2_pdf_url`)
3. OA URL from OpenAlex (requires `oa_url`)
4. PubMed Central (requires `pmcid` or `doi`)
5. Europe PMC (requires `doi` or `pmcid`)
6. Unpaywall (requires `doi`)
7. bioRxiv / medRxiv (requires `doi` starting with `10.1101/`)
8. NBER Working Papers (requires `title`)
9. RePEc / IDEAS (requires `title`)
10. HAL (requires `title`)
11. ERIC (requires `title`)
12. NASA ADS (requires `title` + `config.nasa_ads_key`)
13. OSF Preprints (requires `title`)
14. CORE (requires `doi`)

### Signature

```python
async def download_pdf(
    out_path: Path,
    *,
    doi: str | None = None,
    arxiv_id: str | None = None,
    pmcid: str | None = None,
    s2_pdf_url: str | None = None,
    oa_url: str | None = None,
    title: str | None = None,
    authors: list[str] | None = None,
    year: int | None = None,
    entry_type: str = "article",
    email: str,             # required (Unpaywall policy)
    require_title_match: bool = True,
    config: FetchConfig | None = None,
) -> DownloadResult
```

`title`, `authors`, `year`, and `entry_type` are used both to drive title-search sources (NBER, IDEAS, HAL, ERIC, NASA ADS, OSF) and to feed the multi-signal match validator that runs before any bytes are cached. The validator combines GROBID header title similarity, first-page surname / DOI / year signals, and template-pattern rejection for ERIC / CORE landing pages. When only an identifier is supplied (no title, no authors, no DOI, no year), the positive-signal gate is skipped and the caller gets whatever the identifier path produces — the template hard rejects still fire.

Pass `require_title_match=False` to skip the match validator entirely — no GROBID needed. Any well-formed PDF (`%PDF` magic bytes + a non-trivial size) from the first source that returns one is accepted, and `title_check_score` is left `None`. Use it when the identifier already comes from a trusted match (e.g. a resolved search hit), so re-validating the title would be redundant. The pre-download ranking of title-search candidates is unaffected — it runs locally regardless.

### `DownloadResult`

| field | type | meaning |
| --- | --- | --- |
| `found` | `bool` | whether a validated PDF was written |
| `source` | `str \| None` | which source produced the PDF (`"arxiv"`, `"pmc"`, `"unpaywall"`, …) |
| `out_path` | `Path \| None` | the path the PDF was written to |
| `url_used` | `str \| None` | the URL that actually served the bytes |
| `title_check_score` | `float \| None` | title-similarity score from the match validator (GROBID header title vs. the supplied title); low or `0.0` when accepted on a non-title signal such as a DOI match; `None` when no title comparison ran — the validator was skipped (`require_title_match=False`) or only an identifier was supplied |
| `failed_sources` | `list[str]` | one string per source that was tried and didn't produce a PDF |
| `is_oa` | `bool` | whether Unpaywall or OpenAlex confirmed OA status |
| `oa_url` | `str \| None` | suggested URL if a manual download is the only path |
| `abstract` | `str \| None` | CORE abstract if CORE was queried |

### Errors vs. not-found

- No PDF found across all sources → `DownloadResult(found=False, failed_sources=[...])`. **Not an exception.**
- Infrastructure failures (network unreachable, invalid inputs) → raise `sciwrite_lint.SciWriteLintError` or `ValueError` (e.g. `out_path` doesn't end in `.pdf`, `email=""`).

### Handling "OA but blocked" — the manual-download prompt

Many open-access articles (Taylor & Francis OA, Wiley OA, some Cell Press, NAP, College Board...) are legally OA but gated behind JavaScript interstitials, click-through confirmations, or bot-UA blocks. An HTTP client can't pass these; a real browser does transparently. The API surfaces this as a three-state outcome:

```python
result = await download_pdf(out_path, doi="10.1080/...", email="you@example.com")

if result.found:
    # PDF saved — done.
    use(result.out_path)

elif result.is_oa and result.oa_url:
    # Confirmed OA by Unpaywall or OpenAlex, but every automated download
    # path failed (bot-block / JS-gate / CAPTCHA).
    # Tell the user to fetch it in a browser and drop it somewhere your
    # pipeline can find on the next run.
    print(f"Open {result.oa_url} in a browser, save to {out_path}")

else:
    # No OA signal — paywalled, or OA at a source neither Unpaywall nor
    # OpenAlex indexes. Nothing we can point at.
    print(f"No OA copy found. Reasons: {'; '.join(result.failed_sources)}")
```

The `is_oa` flag fires only when Unpaywall or OpenAlex confirm — in practice that covers most hybrid-OA publishers but not everything. If your use case needs broader coverage, layer your own heuristic on top (e.g. "if `found=False` and `doi` was provided, always offer the user the Unpaywall landing URL as a next step").

## fetch_web — URL → markdown in memory

```python
from sciwrite_lint import fetch_web

result = await fetch_web("https://example.com/blog-post")
if result.url_alive and result.markdown:
    print(result.title)
    print(result.markdown[:500])
```

Follows JS/meta-redirects. Retries once with the www-toggled hostname variant only when the failure might be hostname-specific — a 404 response, or a DNS/connection error that didn't reach a live server. Refusals (401/403/429/451), server errors (5xx), TLS failures, timeouts, and decoding errors affect both hostname variants equally, so the www retry is skipped. Sends browser-like headers (realistic UA + `Accept`, `Accept-Language`, `Sec-Fetch-*`) so WAF-protected sites (Cloudflare, Akamai, etc.) distinguish the request from a naive bot probe. Does not touch the filesystem — callers write to disk if they want persistence.

### Signature

```python
async def fetch_web(
    url: str,
    *,
    config: FetchConfig | None = None,
    client: httpx.AsyncClient | None = None,
) -> WebResult
```

Pass `client` to inject a pre-built `httpx.AsyncClient` (e.g. for tests with `httpx.MockTransport`, or to share a connection pool).

### `WebResult`

| field | type | meaning |
| --- | --- | --- |
| `url_alive` | `bool` | True if a 2xx/3xx response was produced |
| `status_code` | `int \| None` | HTTP status of the final response |
| `content_type` | `str \| None` | from the `Content-Type` header |
| `markdown` | `str \| None` | extracted markdown (None if extraction failed) |
| `title` | `str \| None` | extracted page title |
| `final_url` | `str \| None` | URL after www-variant + redirect chasing |
| `error` | `str \| None` | populated on failure |
| `blocked` | `bool` | True when we could not verify but the URL may still be valid: 4xx refusals (401/403/429/451), 5xx server errors, TLS/SSL failures, timeouts, connection errors, decoding errors, oversized responses. False **only** for explicit HTTP 404/410 or unfixable URL forms — the two cases where we have positive evidence the URL is gone. |

Three-valued failure outcome:

- `url_alive=True` → content served.
- `url_alive=False, blocked=True` → unverifiable. Ask the user to verify the URL in a browser; don't treat it as dead.
- `url_alive=False, blocked=False` → server positively confirmed the URL is gone (HTTP 404/410), or the URL as written cannot be processed at all.

The conservative split means a transient outage, an expired certificate, a gzip quirk, or a Cloudflare challenge does not silently demote an otherwise-valid citation to "remove this." An API failure is not an empty result.

## search_by_title — title → candidate hits

```python
from sciwrite_lint import search_by_title, FetchConfig

hits = await search_by_title(
    "An Economics Paper with a Distinctive Title",
    authors=["Jane Doe"],                 # narrows sources that support author filtering
    email="you@example.com",
    config=FetchConfig(nasa_ads_key=None),  # omit NASA ADS
)
for hit in hits:
    print(hit.source, hit.title, hit.pdf_url)
```

Queries these sources in parallel:

- NBER Working Papers
- RePEc / IDEAS
- HAL
- ERIC
- OSF Preprints
- NASA ADS (only if `config.nasa_ads_key` is set)

No downloading. Use `download_pdf(title=...)` if you want the PDF end-to-end, or feed a returned `hit.pdf_url` into your own downloader.

Unpaywall and CORE are DOI-based and therefore not part of title search — use `download_pdf(doi=...)` to reach them.

The returned list contains **every candidate each source produced**, not only the top hit per source. Callers who want a single best match can rank the list themselves (title similarity, author overlap, year match), or use `download_pdf` which does this internally against a multi-signal validator.

When `authors` is supplied, sources that support server-side author filtering (HAL, NASA ADS, ERIC) narrow their result sets with the first author surname; sources without a native author filter (NBER, IDEAS, OSF) append it to their free-text query as a ranking hint.

### Signature

```python
async def search_by_title(
    title: str,
    authors: list[str] | None = None,
    *,
    email: str,
    config: FetchConfig | None = None,
) -> list[SearchHit]
```

### `SearchHit`

| field | type | meaning |
| --- | --- | --- |
| `source` | `str` | `"nber"`, `"ideas"`, `"hal"`, `"eric"`, `"osf"`, `"nasa_ads"` |
| `pdf_url` | `str \| None` | URL that `download_pdf` can consume |
| `title` | `str \| None` | Title the source returned for this hit, when exposed by the source |
| `authors` | `list[str]` | Authors returned for this hit (HAL, NASA ADS, NBER, ERIC); empty on sources that don't return them per-candidate (IDEAS, OSF) |
| `year` | `int \| None` | Publication year when the source exposes it |
| `doi`, `arxiv_id`, `landing_url`, `abstract`, `score` | various | Reserved for future enrichment; currently `None` |

## FetchConfig

Shared tunables for all three functions. All fields optional; defaults match the internal pipeline.

```python
class FetchConfig(BaseModel):
    timeout: float = 30.0
    user_agent: str = "sciwrite-lint (open-access-acquisition)"
    nasa_ads_key: str | None = None
    unpaywall_interval: float = 0.1   # rate-limit seconds between calls
    core_interval: float = 1.0
```

API keys are passed explicitly — the public API does **not** read from `~/.sciwrite-lint/`. (The internal linter still does, for CLI users.)

## Stability

This is a public API: breaking changes require a SemVer major bump. Private helpers (everything under `sciwrite_lint.oa._*`, or elsewhere in the package prefixed with `_`) are internal and change without notice. If you need something that isn't on the public surface, open an issue rather than importing it directly.

## Errors

All functions raise `sciwrite_lint.SciWriteLintError` (or `ValueError` for invalid inputs) for infrastructure-level problems. "Tried all sources and none produced content" is never an exception — it's communicated via the return value.

```python
from sciwrite_lint import SciWriteLintError, LLMConnectionError

try:
    result = await download_pdf(out, doi="10.1234/foo", email="you@example.com")
except ValueError as e:
    ...  # bad input
except SciWriteLintError as e:
    ...  # network / API failure
```
