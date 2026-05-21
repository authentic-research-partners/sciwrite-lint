"""RePEc / IDEAS — title search via two-hop HTML scrape.

IDEAS (ideas.repec.org) aggregates European and non-US working-paper series
(VATT, Etla, IZA Bonn, CEPR, ZEW, DIW, OECD) plus author self-archives.
There is no JSON API. The search flow is:

  1. POST ``/cgi-bin/htsearch2`` with form data ``q={title}``. Results are
     an HTML page with links of the form ``/p/{series}/{id}.html`` pointing
     at paper landing pages.
  2. GET each landing page. The download section contains an
     ``<INPUT type="radio" name="url" value="...pdf">`` element whose
     ``value`` attribute is the external file URL (hosted on the paper's
     original repository — VATT, IZA, Uni-Erlangen, etc.).

IDEAS returns no structured per-result metadata — titles/authors live in
the landing HTML we'd need to parse, which isn't worth the extra cost
since the downstream validator checks title/authors/year against the
actual PDF anyway. Candidates therefore arrive with URL only; the ranker
treats missing metadata as a neutral signal and lets them through.

When a bib surname is available we append it to the ``q`` form field so
IDEAS's text ranker biases toward papers the author appears on.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
from loguru import logger
from lxml import html as lxml_html

from sciwrite_lint._network import ssrf_safe_client
from sciwrite_lint.fulltext._common import (
    IDEAS_SEARCH,
    AcquisitionResult,
    _MAX_HTML_BYTES,
    _ideas_limiter,
    _polite_user_agent,
)
from sciwrite_lint.fulltext._download import _download_pdf
from sciwrite_lint.fulltext._search import SearchCandidate
from sciwrite_lint.fulltext._validation import BibEvidence, _extract_surname
from sciwrite_lint.rate_limiter import retry_on_transient

_IDEAS_LANDING_PATH_RE = re.compile(r"^/p/(?:[^/]+/){1,3}[^/]+\.html$")

# Max landing pages to fetch per search. Each is an extra request.
_IDEAS_MAX_LANDING_HOPS = 3


def _parse_ideas_landing_urls(html_text: str) -> list[str]:
    """Return all paper-landing URLs (``/p/...``) from IDEAS search HTML.

    Landing URLs look like ``https://ideas.repec.org/p/{series}/{id}.html``
    or the relative ``/p/{series}/{id}.html``; both forms are accepted and
    normalised to absolute URLs. Duplicate URLs are removed preserving
    first-seen order.
    """
    try:
        tree = lxml_html.fromstring(html_text)
    except (ValueError, lxml_html.etree.ParserError):
        logger.debug("IDEAS: could not parse search HTML")
        return []

    out: list[str] = []
    seen: set[str] = set()
    for a in tree.xpath("//a[@href]"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        if href.startswith("https://ideas.repec.org/p/"):
            path = href[len("https://ideas.repec.org") :]
            abs_url = href
        elif href.startswith("http://ideas.repec.org/p/"):
            path = href[len("http://ideas.repec.org") :]
            abs_url = "https://ideas.repec.org" + path
        elif href.startswith("/p/"):
            path = href
            abs_url = "https://ideas.repec.org" + path
        else:
            continue
        if not _IDEAS_LANDING_PATH_RE.match(path):
            continue
        if abs_url in seen:
            continue
        seen.add(abs_url)
        out.append(abs_url)
    return out


def _parse_ideas_file_url(html_text: str) -> str | None:
    """Return the first external PDF URL from an IDEAS landing page.

    Landing pages contain one or more
    ``<input type="radio" name="url" value="..." checked>`` elements whose
    ``value`` is the external file URL. We prefer the ``checked`` one if
    present, else the first ``url`` radio. Only absolute http(s) URLs ending
    in ``.pdf`` are returned; anything else is skipped so the caller never
    receives a landing page or scheme-spoofed value.
    """
    try:
        tree = lxml_html.fromstring(html_text)
    except (ValueError, lxml_html.etree.ParserError):
        logger.debug("IDEAS: could not parse landing HTML")
        return None

    candidates = tree.xpath('//input[@name="url" and @value]')

    def _valid(url: str) -> bool:
        lowered = url.lower()
        return lowered.startswith(("http://", "https://")) and lowered.endswith(".pdf")

    for inp in candidates:
        if inp.get("checked") is not None:
            value = (inp.get("value") or "").strip()
            if _valid(value):
                return value
    for inp in candidates:
        value = (inp.get("value") or "").strip()
        if _valid(value):
            return value
    return None


async def _fetch_ideas_landing(landing_url: str, user_agent: str) -> str | None:
    """Fetch an IDEAS landing page and return its HTML body, or None."""
    try:
        async with _ideas_limiter:
            async with ssrf_safe_client(timeout=15.0) as client:
                resp = await retry_on_transient(
                    lambda: client.get(landing_url, headers={"User-Agent": user_agent}),
                    label="IDEAS landing",
                )
    except httpx.HTTPError as e:
        logger.debug("IDEAS: landing fetch failed for {}: {}", landing_url, e)
        return None
    if resp.status_code != 200:
        logger.debug("IDEAS: landing {} returned {}", landing_url, resp.status_code)
        return None
    if len(resp.content) > _MAX_HTML_BYTES:
        logger.debug("IDEAS: landing too large ({} bytes)", len(resp.content))
        return None
    return resp.text


def _build_query(title: str, authors: list[str] | None) -> str:
    """Combine title + optional surname for IDEAS's free-text ``q`` field."""
    tokens = [title.strip()] if title.strip() else []
    if authors:
        surname = _extract_surname(authors[0]).strip()
        if surname:
            tokens.append(surname)
    return " ".join(tokens)


async def lookup_ideas_by_title(
    title: str,
    polite_email: str = "",
    *,
    authors: list[str] | None = None,
    year: int | None = None,  # noqa: ARG001  # IDEAS has no year filter
) -> list[SearchCandidate]:
    """Search RePEc/IDEAS by title. Return candidates with URL only.

    Runs the two-hop scrape (search → landing → PDF URL) for up to
    :data:`_IDEAS_MAX_LANDING_HOPS` landings. Candidates carry no title
    or author metadata because IDEAS does not expose them in a structured
    form; the ranker treats this as neutral and the validator checks the
    downloaded PDF directly.
    """
    if not title or not title.strip():
        return []
    user_agent = _polite_user_agent(polite_email)
    query = _build_query(title, authors)

    try:
        async with _ideas_limiter:
            async with ssrf_safe_client(timeout=15.0) as client:
                resp = await retry_on_transient(
                    lambda: client.post(
                        IDEAS_SEARCH,
                        data={"q": query},
                        headers={"User-Agent": user_agent},
                    ),
                    label="IDEAS search",
                )
    except httpx.HTTPError as e:
        logger.debug("IDEAS search failed for {!r}: {}", title[:60], e)
        return []

    if resp.status_code != 200:
        logger.debug("IDEAS search returned status {}", resp.status_code)
        return []
    if len(resp.content) > _MAX_HTML_BYTES:
        logger.debug("IDEAS: search response too large ({} bytes)", len(resp.content))
        return []

    landing_urls = _parse_ideas_landing_urls(resp.text)
    candidates: list[SearchCandidate] = []
    for landing in landing_urls[:_IDEAS_MAX_LANDING_HOPS]:
        html_body = await _fetch_ideas_landing(landing, user_agent)
        if not html_body:
            continue
        pdf_url = _parse_ideas_file_url(html_body)
        if pdf_url:
            candidates.append(SearchCandidate(url=pdf_url))
    return candidates


async def download_ideas_pdf(
    pdf_url: str,
    key: str,
    references_dir: Path,
    evidence: BibEvidence | None = None,
) -> AcquisitionResult:
    """Download a PDF from an IDEAS/RePEc result URL."""
    if not pdf_url:
        return AcquisitionResult(found=False)
    dest = references_dir / f"{key}_ideas.pdf"
    return await _download_pdf(
        pdf_url, dest, references_dir, "ideas", evidence=evidence
    )
