# Local Reference Sources

sciwrite-lint reads reference material from two "drop" directories next to your paper. Files you put there skip the open-access waterfall — they become first-class sources for `verify`, `fetch`, and `verify-claims`.

Two directories, one role each:

| Directory | Holds | For |
|-----------|-------|-----|
| `local_pdfs_dir` (default `local_pdfs/`) | `.pdf` and `.md` summaries | Academic sources: journal articles, reports, book scans |
| `local_web_dir` (default `local_web/`) | `.md` and `.mhtml` / `.mht` browser saves | Web pages: organisation pages, FAQs, press releases |

Both directories are auto-detected per paper: if your paper sits at `papers/my_paper/paper.tex`, sciwrite-lint will look at `papers/my_paper/Sources/full_text/` for academic sources and `papers/my_paper/Sources/full_text_web/` for web captures before falling back to the project-wide directories.

The directory itself signals trust level — a `.pdf` in the web directory or an `.mhtml` in the academic directory is silently ignored, not miscategorised.

## File naming

Either form works for matching a file to its `.bib` citekey:

**Form A — citekey prefix** (recommended, authoritative)

```
local_pdfs/smith2020_Important_Paper.pdf
local_web/bcnyorg2024_About.md
```

The leading token up to the first `_`, `-`, `.`, or space is treated as the citekey (case-insensitive). Everything after is decoration — title, version, page range, anything. No fuzzy matching involved.

**Form B — title-only** (used when the filename does not start with a citekey)

```
local_pdfs/The Structure of Scientific Revolutions.pdf
```

The filename is fuzzy-matched against `.bib` titles. Threshold is `0.80`. Works for unique, distinctive titles; fragile for short or generic ones.

Form A wins when both forms apply to the same file.

## Footnote-URL sources: the `Source:` header convention

Papers often cite informational web pages inline as a footnote URL rather than through a formal `.bib` entry — organisation pages, program descriptions, press releases, FAQ pages. In LaTeX that is `\footnote{\url{https://...}}`; in markdown it is a `[^id]` or inline `^[…]` footnote carrying a `<https://...>` or `[text](https://...)` link. sciwrite-lint verifies these claims against archived captures in `local_web_dir`, but because there is no citekey to match, each archived `.md` file must declare the URL it documents in a header.

**The convention.** Within the first 20 lines of any `.md` in `local_web_dir`, include a line of the form:

```markdown
Source: https://example.com/about
```

or equivalently:

```markdown
Source URL: https://example.com/about
```

Case-insensitive. Optional leading whitespace. Tolerates a wrapping HTML comment:

```markdown
<!--
Citation key: example_about
Source URL: https://example.com/about
Fetched:     2026-04-20
-->

# Page Title

…body content…
```

That's it. Given the header, sciwrite-lint matches every footnote URL in your manuscript — `.tex` or `.md` — to the `.md` file whose header declares the same URL, and the claim in that footnote's sentence becomes verifiable.

**URL matching is normalised.** The RFC-correct base normalisation runs via `rfc3986` (the same URI library httpx uses), so scheme + host case, percent-encoding, and path dot-segments are handled to spec. On top of that sciwrite-lint applies:

- Strip default ports (`:80` on http, `:443` on https).
- Strip trailing slash on non-empty paths (`/page/` == `/page`), but preserve root-only `/` (`https://host` ≠ `https://host/` on some servers).
- Drop URL fragment (`#section-2` ignored).
- Drop tracking params: `utm_source`, `utm_medium`, `utm_campaign`, `utm_term`, `utm_content`, `fbclid`, `gclid`, `mc_cid`, `mc_eid`.
- Preserve the order of remaining query parameters (we deliberately do not sort, unlike `w3lib.canonicalize_url` — query order can matter for some endpoints).

### Where the header comes from automatically

Browser-saved MHTML archives (File → Save As → Webpage, Single File) already carry the URL in their MIME envelope headers. sciwrite-lint reads `Snapshot-Content-Location` / `Content-Location` directly from each `.mhtml` / `.mht` in `local_web_dir` when building its URL index, so you can drop raw browser saves in and they participate in footnote-URL matching without any hand-editing. On a match, the MHTML is converted to `.md` at ingest time (trafilatura), and the generated file carries a `Source:` header so downstream readers only ever see markdown.

For `.md` files you write yourself (hand summaries, copies extracted from elsewhere), you add the header.

### What happens if there's no header

A `.md` without a `Source:` header is simply not available as a footnote-URL source — sciwrite-lint cannot know what URL it documents. Footnote URLs whose URL is not declared by any file are logged at INFO with a pointer to `local_web_dir`, so the missing capture is surfaced without failing the run or emitting a false-positive "unsupported claim" finding.

### What happens if two files declare the same URL

First-scanned wins. The duplicate is logged at DEBUG and otherwise ignored. Keep one archived capture per URL.

## Re-ingest: unchanged files are skipped, updates propagate

sciwrite-lint records the SHA-256 of every matched source file in its workspace database at ingest time. On the next run, it re-hashes each file and:

- Skips the copy if the hash matches (cheap: one hash per file, no re-parse, no re-embed).
- Re-copies and re-parses if the hash changes (i.e. you replaced the file with a corrected version under the same name).

This means the drop directory is the source of truth, file by file. You don't need `--fresh` when you swap in a better scan — just drop it in and re-run.

## Examples

### Cited reference with a locally-archived PDF

```
papers/my_paper/paper.tex    contains \cite{smith2020}
papers/my_paper/paper.bib    contains @article{smith2020, ...}
papers/my_paper/Sources/full_text/smith2020_Title.pdf
```

sciwrite-lint matches by citekey prefix, GROBID-parses the PDF, and claim-verifies every claim citing `smith2020` against the text.

### Footnote-URL with a web capture

```
papers/my_paper/paper.tex    contains \footnote{\url{https://example.org/about}}
papers/my_paper/Sources/full_text_web/example_About.md
                             first 20 lines contain:  Source: https://example.org/about
```

sciwrite-lint synthesizes a `Citation` with key `fn_<hash>`, pre-registers it as a `T1` local source (no API calls, no download), and runs claim verification on the enclosing sentence.

### Footnote-URL with a browser MHTML save

```
papers/my_paper/paper.tex    contains \footnote{\url{https://example.org/dynamic-page}}
papers/my_paper/Sources/full_text_web/example_Dynamic.mhtml
                             (saved from the browser, no hand-editing)
```

At ingest, sciwrite-lint converts the `.mhtml` to `.md`, writes a `Source:` header from the MHTML envelope, and matches the footnote URL automatically.

## Current scope

Footnote-URL matching covers LaTeX (`\footnote{\url{}}`) and markdown (`[^id]` / `^[…]` footnotes carrying a `<url>` or `[text](url)` link). It does not yet cover PDF input: with GROBID the footnote → URL → sentence chain is recoverable from the TEI structure but uses a different code path; that's a later change.
