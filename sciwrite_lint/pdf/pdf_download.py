"""Smart PDF downloading with validation.

Downloads papers from OA URLs (Unpaywall, publisher sites). Handles:
- Direct PDF links (Content-Type: application/pdf)
- HTML pages with embedded/linked PDFs
- Validates downloaded PDF matches expected paper via
  :func:`sciwrite_lint.fulltext._validation.validate_pdf_match` (multi-signal
  — GROBID header title + pypdf first-page surname / DOI / year +
  template-pattern rejection)

No LLM — just HTTP, HTML parsing, GROBID title extraction,
and fuzzy string matching.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

import httpx

from sciwrite_lint._network import ResponseTooLarge, ssrf_safe_client, stream_with_limit

if TYPE_CHECKING:
    from sciwrite_lint.fulltext._validation import BibEvidence

_MIN_PDF_SIZE = 5_000
_MAX_PDF_SIZE = 100 * 1024 * 1024  # 100 MB — reject downloads larger than this
_DEFAULT_USER_AGENT = "sciwrite-lint/0.1 (citation-verification)"

_SKIP_PATTERNS = re.compile(
    r"(advertisement|banner|sidebar|nav|footer|header|cookie|consent|signup|login"
    r"|newsletter|subscription|purchase|buy|cart|checkout)",
    re.IGNORECASE,
)


async def download_and_validate(
    url: str,
    key: str,
    references_dir: Path,
    evidence: "BibEvidence | None" = None,
    user_agent: str = _DEFAULT_USER_AGENT,
) -> dict[str, Any]:
    """Download a PDF from a URL and validate it matches the expected paper.

    When *evidence* is provided, the downloaded PDF goes through
    :func:`sciwrite_lint.fulltext._validation.validate_pdf_match` — the same
    multi-signal gate used by the direct-download path. No *evidence* means
    bytes-level validation only (PDF magic, minimum size); the caller gets
    whatever the URL produced without content-match filtering.

    Returns dict with: found, local_path, source_url, pdf_title,
    match_score, reason.
    """
    result = {
        "found": False,
        "local_path": None,
        "source_url": url,
        "pdf_title": "",
        "match_score": 0.0,
        "reason": "",
    }

    try:
        async with ssrf_safe_client(
            timeout=30.0,
            headers={"User-Agent": user_agent},
        ) as client:
            resp = await stream_with_limit(client, url, _MAX_PDF_SIZE)
            if resp.status_code != 200:
                result["reason"] = f"HTTP {resp.status_code}"
                return result

            content_type = resp.headers.get("content-type", "").lower()

            if (
                "application/pdf" in content_type
                or "application/octet-stream" in content_type
                or url.lower().endswith(".pdf")
            ):
                return await _validate_and_save(
                    resp.content,
                    url,
                    key,
                    references_dir,
                    evidence,
                    result,
                )

            if "text/html" in content_type:
                pdf_url = _find_pdf_in_html(resp.text, url)
                if pdf_url:
                    pdf_resp = await stream_with_limit(client, pdf_url, _MAX_PDF_SIZE)
                    if pdf_resp.status_code == 200:
                        return await _validate_and_save(
                            pdf_resp.content,
                            pdf_url,
                            key,
                            references_dir,
                            evidence,
                            result,
                        )
                    else:
                        result["reason"] = (
                            f"PDF link returned HTTP {pdf_resp.status_code}"
                        )
                        return result
                else:
                    result["reason"] = "HTML page with no PDF links found"
                    return result

            result["reason"] = f"Unexpected content type: {content_type}"
            return result

    except ResponseTooLarge as e:
        result["reason"] = str(e)
        return result
    except httpx.TimeoutException:
        result["reason"] = "Timeout"
        return result
    except Exception as e:
        result["reason"] = str(e)[:200]
        return result


async def _validate_and_save(
    content: bytes,
    source_url: str,
    key: str,
    references_dir: Path,
    evidence: "BibEvidence | None",
    result: dict,
) -> dict:
    """Validate PDF content and save if it matches."""
    from sciwrite_lint.fulltext._validation import validate_pdf_match

    result["source_url"] = source_url

    if len(content) < _MIN_PDF_SIZE:
        result["reason"] = f"PDF too small ({len(content)} bytes) — likely error page"
        return result

    if not content[:5] == b"%PDF-":
        result["reason"] = "Not a valid PDF (bad magic bytes)"
        return result

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        if evidence is not None:
            vr = await validate_pdf_match(tmp_path, evidence)
            result["pdf_title"] = vr.signals.get("grobid_title", "")
            result["match_score"] = float(vr.signals.get("title_sim", 0.0) or 0.0)
            if not vr.accepted:
                result["reason"] = vr.reason
                return result

        references_dir.mkdir(parents=True, exist_ok=True)
        dest = references_dir / f"{key}.pdf"
        dest.write_bytes(content)
        result["found"] = True
        result["local_path"] = f"{key}.pdf"
        result["reason"] = (
            f"Downloaded and validated (score={result['match_score']:.2f})"
        )
        return result

    finally:
        tmp_path.unlink(missing_ok=True)


def _title_similarity(a: str, b: str) -> float:
    """Fuzzy title similarity with subtitle-variant handling.

    Books often have subtitles in one source but not the other (e.g.
    "Mind in Society" in bib vs "Mind in Society: The Development of
    Higher Psychological Processes" from the PDF). Tries all combinations
    of full title and main title (before colon) and returns the best score.
    """
    a_low = a.lower().strip()
    b_low = b.lower().strip()
    a_variants = [a_low]
    if ":" in a_low:
        a_variants.append(a_low.split(":")[0].strip())
    b_variants = [b_low]
    if ":" in b_low:
        b_variants.append(b_low.split(":")[0].strip())

    return max(_fuzzy_score(av, bv) for av in a_variants for bv in b_variants)


def _fuzzy_score(a: str, b: str) -> float:
    """Raw fuzzy similarity between two lowercased strings."""
    if not a or not b:
        return 0.0
    try:
        from rapidfuzz import fuzz

        return fuzz.ratio(a, b) / 100.0
    except ImportError:
        pass

    wa = set(a.split())
    wb = set(b.split())
    if not wa or not wb:
        return 0.0
    return (2 * len(wa & wb)) / (len(wa) + len(wb))


def _find_pdf_in_html(html: str, base_url: str) -> str | None:
    """Find the most likely PDF link in an HTML page."""
    candidates: list[tuple[int, str]] = []

    for pattern in [
        r'<embed[^>]+src=["\']([^"\']+\.pdf[^"\']*)["\']',
        r'<iframe[^>]+src=["\']([^"\']+\.pdf[^"\']*)["\']',
        r'<object[^>]+data=["\']([^"\']+\.pdf[^"\']*)["\']',
    ]:
        for match in re.finditer(pattern, html, re.IGNORECASE):
            url = match.group(1)
            if not _SKIP_PATTERNS.search(url):
                candidates.append((0, urljoin(base_url, url)))

    for match in re.finditer(
        r'<a[^>]+href=["\']([^"\']+\.pdf[^"\']*)["\'][^>]*>(.*?)</a>',
        html,
        re.IGNORECASE | re.DOTALL,
    ):
        url = match.group(1)
        link_text = match.group(2).lower()
        if _SKIP_PATTERNS.search(url) or _SKIP_PATTERNS.search(link_text):
            continue
        if any(kw in link_text for kw in ["download", "full text", "pdf", "full-text"]):
            candidates.append((1, urljoin(base_url, url)))
        else:
            candidates.append((2, urljoin(base_url, url)))

    meta = re.search(
        r'<meta[^>]+url=(["\']?)([^"\'\s>]+\.pdf[^"\'\s>]*)\1',
        html,
        re.IGNORECASE,
    )
    if meta:
        candidates.append((0, urljoin(base_url, meta.group(2))))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]
