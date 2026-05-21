"""API clients for citation verification (CrossRef + OpenAlex + Semantic Scholar + Open Library + Library of Congress).

All APIs are free and require no API keys. Rate limiting is client-side
via MonotonicRateLimiter. Batch methods use httpx.AsyncClient for concurrent
lookups.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from sciwrite_lint._network import (
    clean_and_validate_doi,
    is_valid_arxiv_id,
    is_valid_isbn,
    is_valid_lccn,
    is_valid_pmcid,
    is_valid_pmid,
)
from sciwrite_lint.config import LintConfig
from sciwrite_lint.models import Citation
from sciwrite_lint.rate_limiter import (
    MonotonicRateLimiter,
    rate_limited_get,
    retry_on_transient,
)


_ARXIV_DOI_PREFIX = "10.48550/arXiv."  # DataCite DOI prefix for arXiv papers

# Module-level rate limiters (persist across calls)
_openalex_limiter = MonotonicRateLimiter(10, 1.0)  # 10 req/s
_s2_limiter = MonotonicRateLimiter(5, 1.0)  # ~5 req/s
_openlibrary_limiter = MonotonicRateLimiter(5, 1.0)  # ~5 req/s (be polite)
_loc_limiter = MonotonicRateLimiter(5, 1.0)  # ~5 req/s (be polite)


class CitationAPI:
    """Verify citations against CrossRef, OpenAlex, Semantic Scholar, Open Library, and Library of Congress."""

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        config: LintConfig | None = None,
    ):
        self._client = client or httpx.AsyncClient(timeout=15.0, follow_redirects=True)
        self._owns_client = client is None
        self._config = config or LintConfig()

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def lookup(self, citation: Citation) -> dict[str, Any] | None:
        """Try CrossRef → OpenAlex → Semantic Scholar → Open Library → Library of Congress.

        After each API returns a result, validates that the result plausibly
        matches the citation (title similarity ≥ 0.50). If an ID-based lookup
        returns a completely different paper (wrong DOI, hallucinated arXiv ID),
        the result is discarded and the next API is tried.
        """
        id_mismatch: dict[str, Any] | None = None
        for api_fn in (
            self._crossref_lookup,
            self._openalex_lookup,
            self._s2_lookup,
            self._openlibrary_lookup,
            self._loc_lookup,
        ):
            result = await api_fn(citation)
            if result and not result.get("error"):
                if _id_result_matches(citation, result):
                    if id_mismatch is not None:
                        # We found the correct paper via title search after
                        # an ID-based lookup returned the wrong paper.
                        # Tag the result so downstream checks can report it.
                        result["id_mismatch"] = id_mismatch
                    return result
                # Record the mismatch for downstream reporting
                if not id_mismatch:
                    id_mismatch = {
                        "source": result.get("source", ""),
                        "wrong_title": result.get("title", ""),
                        "wrong_doi": result.get("doi", ""),
                        "bib_doi": citation.doi,
                        "bib_arxiv_id": citation.arxiv_id,
                        "bib_pmid": citation.pmid,
                        "bib_isbn": citation.isbn,
                        "bib_lccn": citation.lccn,
                    }
        return None

    # ------------------------------------------------------------------
    # CrossRef (async httpx)
    # ------------------------------------------------------------------

    async def _crossref_lookup(self, citation: Citation) -> dict[str, Any] | None:
        try:
            from sciwrite_lint.references.crossref import crossref_lookup

            return await crossref_lookup(
                citation,
                polite_email=self._config.polite_email,
                client=self._client,
            )
        except Exception as e:
            # The waterfall caller distinguishes ``{"error": …}`` from
            # ``None`` (not-found): error dicts mean "lookup itself
            # failed" and stay visible, ``None`` means "no result for
            # this citation". All API lookups in this module return the
            # error shape so transient network failures don't masquerade
            # as misses.
            logger.debug("CrossRef lookup failed for {}: {}", citation.key, e)
            return {"error": str(e), "source": "crossref"}

    # ------------------------------------------------------------------
    # OpenAlex
    # ------------------------------------------------------------------

    async def _openalex_lookup(self, citation: Citation) -> dict[str, Any] | None:
        try:
            # Try DOI lookup
            clean_doi = clean_and_validate_doi(citation.doi) if citation.doi else None
            if clean_doi:
                resp = await rate_limited_get(
                    _openalex_limiter,
                    f"https://api.openalex.org/works/doi:{clean_doi}",
                    params={"select": _OA_FIELDS},
                    label="OpenAlex DOI",
                    client=self._client,
                    service="openalex",
                )
                if resp.status_code == 200:
                    return _parse_openalex(resp.json())

            # Try arXiv DOI (10.48550/arXiv.XXXX.XXXXX)
            if (
                citation.arxiv_id
                and not clean_doi
                and is_valid_arxiv_id(citation.arxiv_id)
            ):
                arxiv_doi = f"{_ARXIV_DOI_PREFIX}{citation.arxiv_id}"
                resp = await rate_limited_get(
                    _openalex_limiter,
                    f"https://api.openalex.org/works/doi:{arxiv_doi}",
                    params={"select": _OA_FIELDS},
                    label="OpenAlex arXiv DOI",
                    client=self._client,
                    service="openalex",
                )
                if resp.status_code == 200:
                    return _parse_openalex(resp.json())
                logger.debug(
                    "OpenAlex arXiv DOI lookup failed for '{}' (status {})",
                    citation.key,
                    resp.status_code,
                )

            # Try PMID lookup via filter
            if citation.pmid and is_valid_pmid(citation.pmid):
                resp = await rate_limited_get(
                    _openalex_limiter,
                    "https://api.openalex.org/works",
                    params={
                        "filter": f"ids.pmid:https://pubmed.ncbi.nlm.nih.gov/{citation.pmid}",
                        "select": _OA_FIELDS,
                    },
                    label="OpenAlex PMID",
                    client=self._client,
                    service="openalex",
                )
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    if results:
                        return _parse_openalex(results[0])

            query = _search_query(citation)
            if not query:
                return None
            resp = await rate_limited_get(
                _openalex_limiter,
                "https://api.openalex.org/works",
                params={
                    "filter": f"title.search:{query}",
                    "per_page": 10,
                    "select": _OA_FIELDS,
                },
                label="OpenAlex search",
                client=self._client,
                service="openalex",
            )
            if resp.status_code != 200:
                return None
            results = resp.json().get("results", [])
            parsed = [_parse_openalex(r) for r in results]
            return (
                _best_match(
                    citation.title or "",
                    parsed,
                    query_authors=citation.authors,
                    query_year=int(citation.year) if citation.year.isdigit() else None,
                    query_venue=citation.venue,
                )
                if parsed
                else None
            )

        except Exception as e:
            return {"error": str(e), "source": "openalex"}

    # ------------------------------------------------------------------
    # Semantic Scholar
    # ------------------------------------------------------------------

    async def _s2_lookup(self, citation: Citation) -> dict[str, Any] | None:
        try:
            # Try direct identifier lookups (DOI, arXiv, PMID, PMC)
            for prefix, value, label in self._s2_id_variants(citation):
                resp = await rate_limited_get(
                    _s2_limiter,
                    f"https://api.semanticscholar.org/graph/v1/paper/{prefix}:{value}",
                    params={"fields": _S2_FIELDS},
                    label=f"S2 {label}",
                    client=self._client,
                    service="semantic_scholar",
                )
                if resp.status_code == 200:
                    return _parse_s2(resp.json())
                logger.debug(
                    "S2 {} lookup failed for '{}' (status {})",
                    label,
                    citation.key,
                    resp.status_code,
                )

            # Identifier lookups failed — try title/author search
            query = _search_query(citation)
            if not query:
                return None
            resp = await rate_limited_get(
                _s2_limiter,
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": query, "limit": 10, "fields": _S2_FIELDS},
                label="S2 search",
                client=self._client,
                service="semantic_scholar",
            )
            if resp.status_code != 200:
                return None
            papers = resp.json().get("data", [])
            parsed = [_parse_s2(p) for p in papers]
            return (
                _best_match(
                    citation.title or "",
                    parsed,
                    query_authors=citation.authors,
                    query_year=int(citation.year) if citation.year.isdigit() else None,
                    query_venue=citation.venue,
                )
                if parsed
                else None
            )

        except Exception as e:
            return {"error": str(e), "source": "semantic_scholar"}

    @staticmethod
    def _s2_id_variants(
        citation: Citation,
    ) -> list[tuple[str, str, str]]:
        """Return (prefix, value, label) tuples for S2 identifier lookup."""
        variants: list[tuple[str, str, str]] = []
        clean_doi = clean_and_validate_doi(citation.doi) if citation.doi else None
        if clean_doi:
            variants.append(("DOI", clean_doi, "DOI"))
        if citation.arxiv_id and is_valid_arxiv_id(citation.arxiv_id):
            variants.append(("ArXiv", citation.arxiv_id, "arXiv"))
        if citation.pmid and is_valid_pmid(citation.pmid):
            variants.append(("PMID", citation.pmid, "PMID"))
        if citation.pmc_id and is_valid_pmcid(citation.pmc_id):
            variants.append(("PMCID", citation.pmc_id, "PMC"))
        if citation.isbn and is_valid_isbn(citation.isbn):
            variants.append(("ISBN", citation.isbn, "ISBN"))
        return variants

    # ------------------------------------------------------------------
    # Open Library (books, reports, monographs)
    # ------------------------------------------------------------------

    async def _openlibrary_lookup(self, citation: Citation) -> dict[str, Any] | None:
        try:
            # Direct ISBN lookup — high-confidence match, no title search needed
            if citation.isbn and is_valid_isbn(citation.isbn):
                isbn_result = await self._openlibrary_isbn_lookup(citation.isbn)
                if isbn_result:
                    return isbn_result

            query = _search_query(citation)
            if not query:
                return None
            params: dict[str, str | int] = {"q": query, "limit": 10}
            if citation.authors:
                params["author"] = citation.authors[0].split()[-1]
            resp = await rate_limited_get(
                _openlibrary_limiter,
                "https://openlibrary.org/search.json",
                params=params,
                label="Open Library search",
                client=self._client,
                service="openlibrary",
            )
            if resp.status_code != 200:
                return None
            docs = resp.json().get("docs", [])
            return _parse_openlibrary(docs[0]) if docs else None
        except Exception as e:
            return {"error": str(e), "source": "openlibrary"}

    async def _openlibrary_isbn_lookup(self, isbn: str) -> dict[str, Any] | None:
        """Direct ISBN lookup via Open Library ISBN API."""
        resp = await rate_limited_get(
            _openlibrary_limiter,
            f"https://openlibrary.org/isbn/{isbn}.json",
            label="Open Library ISBN lookup",
            client=self._client,
            service="openlibrary",
        )
        if resp.status_code != 200:
            logger.debug(
                "Open Library ISBN lookup failed for '{}': HTTP {}",
                isbn,
                resp.status_code,
            )
            return None
        data = resp.json()
        title = data.get("title", "")
        if not title:
            logger.debug("Open Library ISBN '{}': response has no title field", isbn)
            return None
        # OL stores subtitles separately — concatenate for matching
        subtitle = data.get("subtitle", "")
        if subtitle:
            title = f"{title}: {subtitle}"
        # Authors require a separate lookup (OL returns author keys, not names)
        # but having title + ISBN is enough for a high-confidence match
        return {
            "source": "openlibrary",
            "title": title,
            "authors": [],
            "year": _extract_ol_year(data),
            "doi": "",
            "venue": "",
            "citation_count": 0,
            "isbn": isbn,
        }

    # ------------------------------------------------------------------
    # Library of Congress (books, reports, government publications)
    # ------------------------------------------------------------------

    async def _loc_lookup(self, citation: Citation) -> dict[str, Any] | None:
        try:
            # Direct LCCN lookup — high-confidence match
            if citation.lccn and is_valid_lccn(citation.lccn):
                lccn_result = await self._loc_lccn_lookup(citation.lccn)
                if lccn_result:
                    return lccn_result

            query = _search_query(citation)
            if not query:
                return None
            resp = await rate_limited_get(
                _loc_limiter,
                "https://www.loc.gov/books/",
                params={"q": query, "fo": "json"},
                label="Library of Congress search",
                client=self._client,
                service="loc",
            )
            if resp.status_code != 200:
                return None
            results = resp.json().get("results", [])
            parsed = [_parse_loc(r) for r in results]
            return (
                _best_match(
                    citation.title or "",
                    parsed,
                    query_authors=citation.authors,
                    query_year=int(citation.year) if citation.year.isdigit() else None,
                    query_venue=citation.venue,
                )
                if parsed
                else None
            )
        except Exception as e:
            return {"error": str(e), "source": "loc"}

    async def _loc_lccn_lookup(self, lccn: str) -> dict[str, Any] | None:
        """Direct LCCN lookup via Library of Congress item API."""
        resp = await rate_limited_get(
            _loc_limiter,
            f"https://www.loc.gov/item/{lccn}/",
            params={"fo": "json"},
            label="Library of Congress LCCN lookup",
            client=self._client,
            service="loc",
        )
        if resp.status_code != 200:
            logger.debug(
                "LoC LCCN lookup failed for '{}': HTTP {}",
                lccn,
                resp.status_code,
            )
            return None
        data = resp.json()
        item = data.get("item", {})
        title = item.get("title", "")
        if not title:
            logger.debug("LoC LCCN '{}': response has no title field", lccn)
            return None
        contributors = item.get("contributor_names") or []
        date_str = item.get("date") or ""
        year = None
        if date_str:
            m = re.match(r"(\d{4})", date_str)
            if m:
                year = int(m.group(1))
        return {
            "source": "loc",
            "title": title,
            "authors": contributors,
            "year": year,
            "doi": "",
            "venue": "",
            "citation_count": 0,
            "lccn": lccn,
        }


# ---------------------------------------------------------------------------
# Batch async API lookups
# ---------------------------------------------------------------------------


async def batch_openalex(
    citations: list[Citation],
    config: LintConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Batch DOI lookup via OpenAlex filter. One request for all DOIs.

    Handles real DOIs and arXiv DOIs (10.48550/arXiv.XXXX.XXXXX) in a
    single batch request. Returns {citation_key: parsed_result}.
    """
    config = config or LintConfig()

    # Collect all DOIs: explicit DOIs + arXiv synthetic DOIs
    all_dois: list[tuple[str, str]] = []  # (key, doi)
    for c in citations:
        clean = clean_and_validate_doi(c.doi) if c.doi else None
        if clean:
            all_dois.append((c.key, clean))
        elif c.arxiv_id and is_valid_arxiv_id(c.arxiv_id):
            all_dois.append((c.key, f"{_ARXIV_DOI_PREFIX}{c.arxiv_id}"))

    if not all_dois:
        return {}

    # OpenAlex supports pipe-separated DOIs in filter
    doi_filter = "|".join(f"https://doi.org/{doi}" for _, doi in all_dois)
    doi_by_key = {doi.lower(): key for key, doi in all_dois}

    params: dict[str, str | int] = {
        "filter": f"doi:{doi_filter}",
        "per_page": 200,
        "select": _OA_FIELDS,
    }
    if config.polite_email:
        params["mailto"] = config.polite_email

    # Secondary index by arXiv ID: OpenAlex may return a different
    # canonical DOI than the query DOI (e.g. arXiv DOI → publisher DOI).
    keys_in_batch = {key for key, _ in all_dois}
    key_by_arxiv = {
        c.arxiv_id.lower(): c.key
        for c in citations
        if c.arxiv_id and c.key in keys_in_batch
    }

    results: dict[str, dict[str, Any]] = {}
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        from sciwrite_lint.usage import tracked

        async with tracked("openalex", dois=len(all_dois)):
            resp = await retry_on_transient(
                lambda: client.get("https://api.openalex.org/works", params=params),
                label="OpenAlex batch",
            )
            if resp.status_code != 200:
                # API failure ≠ "no matches". Log at WARNING so the operator
                # sees the API was unreachable; the verify-cascade still
                # falls through to S2/CrossRef per the pipeline contract.
                logger.warning(
                    "OpenAlex batch returned status {}; cascading to next provider",
                    resp.status_code,
                )
                return results
            works = resp.json().get("results", [])
            for work in works:
                parsed = _parse_openalex(work)
                doi = (parsed.get("doi") or "").lower()
                key = doi_by_key.get(doi)
                # Response DOI may differ from query DOI (arXiv → publisher).
                # Try arXiv ID from response as secondary match.
                if not key:
                    arxiv_id = (parsed.get("arxiv_id") or "").lower()
                    key = key_by_arxiv.get(arxiv_id)
                if key:
                    results[key] = parsed

    return results


async def batch_s2(
    citations: list[Citation],
    config: LintConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Batch lookup via Semantic Scholar POST endpoint.

    Accepts up to 500 IDs per request. Returns {citation_key: parsed_result}.
    """
    # Build ID list: DOI, arXiv, PMID — S2 batch supports all three.
    id_entries: list[tuple[str, str]] = []  # (key, s2_id_string)
    for c in citations:
        if c.doi:
            id_entries.append((c.key, f"DOI:{c.doi}"))
        elif c.arxiv_id and is_valid_arxiv_id(c.arxiv_id):
            id_entries.append((c.key, f"ARXIV:{c.arxiv_id}"))
        elif c.pmid and is_valid_pmid(c.pmid):
            id_entries.append((c.key, f"PMID:{c.pmid}"))

    if not id_entries:
        return {}

    results: dict[str, dict[str, Any]] = {}

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        from sciwrite_lint.usage import tracked

        # S2 batch: max 500 per request
        for i in range(0, len(id_entries), 500):
            batch = id_entries[i : i + 500]
            ids = [s2_id for _, s2_id in batch]
            async with tracked("semantic_scholar", ids=len(ids)):
                try:
                    resp = await retry_on_transient(
                        lambda ids=ids: client.post(
                            "https://api.semanticscholar.org/graph/v1/paper/batch",
                            json={"ids": ids},
                            params={"fields": _S2_FIELDS},
                        ),
                        label="S2 batch",
                    )
                    if resp.status_code != 200:
                        # API failure ≠ "no matches". Log at WARNING so the
                        # operator sees the API was unreachable.
                        logger.warning(
                            "S2 batch returned status {} for {} IDs; "
                            "cascading to next provider",
                            resp.status_code,
                            len(ids),
                        )
                        continue
                    papers = resp.json()
                    for paper, (key, _s2_id) in zip(papers, batch):
                        if paper:  # S2 returns null for not-found
                            results[key] = _parse_s2(paper)
                except httpx.HTTPError as e:
                    # Network error ≠ "no matches". WARNING so the operator
                    # can see API was unreachable; cascade continues.
                    logger.warning(
                        "S2 batch request failed ({}: {}); cascading to next provider",
                        type(e).__name__,
                        e,
                    )
                    continue

    return results


async def parallel_crossref(
    citations: list[Citation],
    config: LintConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Parallel CrossRef lookups with rate-limiting semaphore.

    Async httpx — one shared client, no threads, no per-call SSL overhead.
    """
    from sciwrite_lint.references.crossref import crossref_lookup

    config = config or LintConfig()
    results: dict[str, dict[str, Any]] = {}
    # CrossRef polite pool (with mailto): 10 req/s, 3 concurrent
    # Public pool (no mailto): 5 req/s, 1 concurrent
    if config.polite_email:
        crossref_limiter = MonotonicRateLimiter(10, 1.0)
        sem = asyncio.Semaphore(3)
    else:
        crossref_limiter = MonotonicRateLimiter(5, 1.0)
        sem = asyncio.Semaphore(1)

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:

        async def _lookup_one(c: Citation) -> None:
            async with sem, crossref_limiter:
                from sciwrite_lint.usage import tracked

                async with tracked("crossref"):
                    try:
                        result = await crossref_lookup(
                            c,
                            polite_email=config.polite_email,
                            client=client,
                        )
                        if result is None:
                            return  # not-found — fine, next provider handles
                        if result.get("error"):
                            # API failure ≠ "no match". Log at WARNING so
                            # the operator sees CrossRef was unreachable;
                            # don't propagate the error dict into ``results``
                            # because callers expect parsed-result shape.
                            logger.warning(
                                "CrossRef lookup failed for {}: {}",
                                c.key,
                                result["error"],
                            )
                            return
                        results[c.key] = result
                    except Exception as e:
                        # Network error ≠ "no match". WARNING so visible.
                        logger.warning(
                            "CrossRef lookup raised for {} ({}: {})",
                            c.key,
                            type(e).__name__,
                            e,
                        )

        await asyncio.gather(*[_lookup_one(c) for c in citations])
    return results


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_OA_FIELDS = (
    "id,ids,doi,title,authorships,publication_year,"
    "cited_by_count,primary_location,locations,open_access,abstract_inverted_index"
)

_S2_FIELDS = (
    "title,authors,year,venue,citationCount,externalIds,abstract,url,openAccessPdf"
)


def _decode_inverted_abstract(inverted: dict[str, list[int]]) -> str:
    word_positions = [
        (pos, word) for word, positions in inverted.items() for pos in positions
    ]
    word_positions.sort()
    return " ".join(w for _, w in word_positions)


def _parse_openalex(work: dict) -> dict[str, Any]:
    authors = [
        a.get("author", {}).get("display_name", "")
        for a in work.get("authorships", [])
        if a.get("author", {}).get("display_name")
    ]
    doi = work.get("doi", "") or ""
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/") :]
    abstract = None
    inverted = work.get("abstract_inverted_index")
    if inverted:
        abstract = _decode_inverted_abstract(inverted)

    # Extract arxiv_id from locations (ids.arxiv is often missing)
    arxiv_id = None
    for loc in work.get("locations") or []:
        landing = loc.get("landing_page_url") or ""
        if "arxiv.org/abs/" in landing:
            arxiv_id = landing.split("arxiv.org/abs/")[-1]
            break

    # Extract PMCID from OpenAlex ids (format: "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC...")
    pmcid = None
    oa_pmcid = (work.get("ids") or {}).get("pmcid") or ""
    if "PMC" in oa_pmcid:
        pmcid = oa_pmcid.split("/")[-1] if "/" in oa_pmcid else oa_pmcid

    # Venue from primary_location.source (canonical name from OpenAlex)
    venue = ""
    primary_loc = work.get("primary_location") or {}
    source = primary_loc.get("source") or {}
    venue = source.get("display_name") or ""

    return {
        "source": "openalex",
        "title": work.get("title", ""),
        "authors": authors,
        "year": work.get("publication_year"),
        "doi": doi,
        "venue": venue,
        "citation_count": work.get("cited_by_count", 0),
        "pdf_url": work.get("open_access", {}).get("oa_url"),
        "arxiv_id": arxiv_id,
        "pmcid": pmcid,
        "abstract": abstract,
        "retracted": work.get("is_retracted", False),
    }


def _parse_s2(paper: dict) -> dict[str, Any]:
    ext_ids = paper.get("externalIds") or {}
    oa_pdf = paper.get("openAccessPdf") or {}
    return {
        "source": "semantic_scholar",
        "title": paper.get("title", ""),
        "authors": [a.get("name", "") for a in (paper.get("authors") or [])],
        "year": paper.get("year"),
        "doi": ext_ids.get("DOI", ""),
        "venue": paper.get("venue", ""),
        "citation_count": paper.get("citationCount", 0),
        "arxiv_id": ext_ids.get("ArXiv"),
        "pmcid": ext_ids.get("PubMedCentral"),
        "s2_pdf_url": oa_pdf.get("url"),
        "abstract": paper.get("abstract"),
        "url": paper.get("url", ""),
    }


def _extract_ol_year(data: dict) -> int | None:
    """Extract publication year from Open Library ISBN API response."""
    pd = data.get("publish_date", "")
    if pd:
        m = re.search(r"\b(\d{4})\b", pd)
        if m:
            return int(m.group(1))
    return None


def _parse_openlibrary(doc: dict) -> dict[str, Any]:
    """Parse an Open Library search result into normalized format."""
    isbn_list = doc.get("isbn") or []
    title = doc.get("title", "")
    subtitle = doc.get("subtitle", "")
    if subtitle:
        title = f"{title}: {subtitle}"
    return {
        "source": "openlibrary",
        "title": title,
        "authors": doc.get("author_name") or [],
        "year": doc.get("first_publish_year"),
        "doi": "",
        "venue": doc.get("publisher", [""])[0] if doc.get("publisher") else "",
        "citation_count": 0,
        "isbn": isbn_list[0] if isbn_list else "",
    }


def _parse_loc(item: dict) -> dict[str, Any]:
    """Parse a Library of Congress search result into normalized format."""
    contributors = item.get("contributor") or []
    date_str = item.get("date") or ""
    # LoC dates can be "1910" or "1910-01-01" etc.
    year = None
    if date_str:
        match = re.match(r"(\d{4})", date_str)
        if match:
            year = int(match.group(1))
    return {
        "source": "loc",
        "title": item.get("title", ""),
        "authors": contributors,
        "year": year,
        "doi": "",
        "venue": "",
        "citation_count": 0,
    }


def _name_variants(name: str) -> list[str]:
    """Generate common formatting variants of an author name.

    Given "John A. Smith", produces variants like:
    - "john a smith", "john smith", "j a smith", "j smith"
    - "smith john a", "smith john", "smith j a", "smith j"

    Handles comma-delimited "Family, Given" format, anyascii
    transliteration (Müller→Mueller, Cyrillic→Latin), given names
    as initials or full, middle names droppable, and both
    given-family and family-given orderings.

    For 3+ token names, tries both last-token-as-family (Western)
    and first-token-as-family (East Asian, Hungarian) to cover
    ambiguous name orderings without a comma delimiter.

    All-initials variants (e.g. "w w") are filtered out as too
    ambiguous. Family name is never reduced to an initial.
    """
    from anyascii import anyascii

    name = anyascii(name).lower().strip()
    if not name:
        return []

    # Handle "Family, Given" format before stripping punctuation
    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        if len(parts) == 2 and parts[1]:
            name = f"{parts[1]} {parts[0]}"

    name = re.sub(r"[.,;:'\"]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    tokens = name.split()
    if not tokens:
        return []
    if len(tokens) == 1:
        return [tokens[0]]

    # Build initial variants for each token: "john" → ["john", "j"]
    token_forms: list[list[str]] = []
    for t in tokens:
        forms = [t]
        if len(t) > 1:
            forms.append(t[0])  # initial
        token_forms.append(forms)

    # Generate given-name combinations: first given name is always present
    # (full or initial), middle names can be full/initial/dropped.
    def _middle_combos(forms: list[list[str]]) -> list[list[str]]:
        """Cartesian product of middle-name forms, allowing drops."""
        if not forms:
            return [[]]
        combos: list[list[str]] = []
        for f in forms[0]:
            for rest in _middle_combos(forms[1:]):
                combos.append([f, *rest])
        # Also drop this middle name entirely
        for rest in _middle_combos(forms[1:]):
            combos.append(rest)
        return combos

    def _variants_for_split(family: str, given_forms: list[list[str]]) -> set[str]:
        """Generate variants for one family/given assignment."""
        result: set[str] = set()
        first_forms_list = given_forms[0] if given_forms else []
        mid_forms = given_forms[1:] if len(given_forms) > 1 else []

        combos: list[list[str]] = []
        if first_forms_list:
            for ff in first_forms_list:
                for mc in _middle_combos(mid_forms):
                    combos.append([ff, *mc])
            # Also try without given names entirely (just family)
            combos.append([])
        else:
            combos.append([])

        for combo in combos:
            result.add(" ".join([*combo, family]))
            result.add(" ".join([family, *combo]))
        result.add(family)
        return result

    variants: set[str] = set()

    # Primary: assume last token is family (Western convention)
    variants |= _variants_for_split(tokens[-1], token_forms[:-1])

    # Also try first token as family (East Asian, Hungarian, etc.)
    # For 2-token names this is redundant (cross-product symmetry),
    # but for 3+ tokens the middle names shift.
    if len(tokens) >= 3:
        variants |= _variants_for_split(tokens[0], token_forms[1:])

    # Drop all-initials variants (e.g. "w w") — too ambiguous
    return [v for v in variants if any(len(t) > 1 for t in v.split())]


def _author_overlap(
    query_authors: list[str] | tuple[str, ...],
    item_authors: list[str] | tuple[str, ...],
) -> float:
    """Pairwise best-match author similarity (0.0 to 1.0).

    For each query author, generates all common name formatting variants
    and finds the best fuzzy match against all variants of each item
    author. Returns the average of these best matches across query authors.
    """
    from rapidfuzz import fuzz

    q_variant_lists = [_name_variants(a) for a in query_authors if a.strip()]
    i_variant_lists = [_name_variants(a) for a in item_authors if a.strip()]
    q_variant_lists = [v for v in q_variant_lists if v]
    i_variant_lists = [v for v in i_variant_lists if v]
    if not q_variant_lists or not i_variant_lists:
        return 0.0

    total = 0.0
    for q_variants in q_variant_lists:
        best = 0.0
        for i_variants in i_variant_lists:
            for qv in q_variants:
                for iv in i_variants:
                    score = fuzz.ratio(qv, iv) / 100.0
                    if score > best:
                        best = score
        total += best
    return total / len(q_variant_lists)


def _best_match(
    query_title: str,
    candidates: list[dict[str, Any]],
    threshold: float = 0.70,
    query_authors: list[str] | tuple[str, ...] | None = None,
    query_year: int | None = None,
    query_venue: str = "",
) -> dict[str, Any] | None:
    """Pick the candidate whose title best matches the query.

    API title searches can return commentaries, reviews, or unrelated
    papers above the actual match. Scores all candidates by fuzzy title
    similarity and returns the best one above *threshold*.

    If *query_authors* is provided, candidates are penalized based on
    pairwise fuzzy author name similarity (handles initials, name order,
    transliteration). Score is scaled from 0.5 (no match) to 1.0 (perfect).

    If *query_year* is provided, candidates whose year diverges are
    penalized with quadratic decay: ``1 / (1 + (delta/3)²)``.
    Tolerates ±1 year (preprint→published) while crushing large deltas.

    If *query_venue* is provided, candidates with a matching venue get
    a small boost (tiebreaker). Score is scaled from 0.9 (no match) to
    1.0 (perfect match). Never penalizes below 0.9 — venue is only
    used to prefer the right version of the same paper.
    """
    from rapidfuzz import fuzz

    query_lower = query_title.lower().strip()
    query_venue_lower = query_venue.lower().strip()

    best: dict[str, Any] | None = None
    best_score = 0.0

    for item in candidates:
        item_title = (item.get("title") or "").lower().strip()
        if not item_title:
            continue
        score = fuzz.ratio(query_lower, item_title) / 100.0

        # Graduated author penalty (pairwise fuzzy name matching)
        if query_authors:
            item_authors = item.get("authors") or []
            if item_authors:
                sim = _author_overlap(query_authors, item_authors)
                # Scale from 0.5 (no match) to 1.0 (perfect match)
                score *= 0.5 + 0.5 * sim

        # Graduated year penalty — quadratic decay (tolerates ±1 for
        # preprint→published, crushes large deltas like 13+ years)
        if query_year is not None:
            item_year = item.get("year")
            if item_year is not None:
                delta = abs(item_year - query_year)
                score *= 1.0 / (1.0 + (delta / 3.0) ** 2)

        # Venue tiebreaker — boost matching venue, never penalize below 0.9
        if query_venue_lower:
            item_venue = (item.get("venue") or "").lower().strip()
            if item_venue:
                vsim = fuzz.partial_ratio(query_venue_lower, item_venue) / 100.0
                score *= 0.9 + 0.1 * vsim

        if score > best_score:
            best_score = score
            best = item

    return best if best_score >= threshold else None


def _id_result_matches(citation: Citation, result: dict[str, Any]) -> bool:
    """Check whether an ID-based lookup result plausibly matches the citation.

    LLMs can hallucinate or corrupt DOIs, arXiv IDs, and PMIDs. When an ID
    lookup returns a completely different paper, the result should be
    discarded so the matching engine can find the correct one.

    Uses the same composite scoring as ``_best_match`` (title × author ×
    year) with a lenient threshold (0.40) — we only reject clear
    mismatches, not minor metadata variations.
    """
    # Treat the result as a single-candidate selection problem
    score = _best_match(
        citation.title or "",
        [result],
        threshold=0.0,  # always return a score, we check manually
        query_authors=citation.authors if citation.authors else None,
        query_year=int(citation.year) if citation.year.isdigit() else None,
        query_venue=citation.venue,
    )
    if score is None:
        # No title at all — can't validate, trust the ID
        return True

    # Re-score to get the actual numeric value (we need the score, not
    # just the match). Use title sim as primary gate since _best_match
    # returns the item not the score.
    from rapidfuzz import fuzz

    if not citation.title:
        return True
    api_title = (result.get("title") or "").strip()
    if not api_title:
        return True

    # Try all subtitle combinations: books often have subtitles in bib
    # but not in the API (or vice versa). Take the best match.
    bib_titles = [citation.title.lower()]
    if ":" in citation.title:
        bib_titles.append(citation.title.split(":")[0].strip().lower())
    api_titles = [api_title.lower()]
    if ":" in api_title:
        api_titles.append(api_title.split(":")[0].strip().lower())
    title_sim = max(
        fuzz.ratio(bt, at) / 100.0 for bt in bib_titles for at in api_titles
    )

    # Author check — if both have authors and overlap is very low, suspect
    author_sim = 1.0
    api_authors = result.get("authors") or []
    if citation.authors and api_authors:
        author_sim = _author_overlap(citation.authors, api_authors)

    # Year check — lenient for ID-based lookups. The ID already resolved
    # to a paper; we only need to catch hallucinated IDs pointing to a
    # completely different paper. Year metadata drifts (preprint→published,
    # API re-dating like OpenAlex updating Vaswani 2017 to 2025) should
    # not reject an otherwise perfect title+author match.  Floor at 0.50
    # so year alone cannot push composite below 0.40 when title≈1.0.
    year_factor = 1.0
    api_year = result.get("year")
    if citation.year.isdigit() and api_year is not None:
        delta = abs(int(citation.year) - api_year)
        year_factor = max(0.50, 1.0 / (1.0 + (delta / 3.0) ** 2))

    composite = title_sim * (0.5 + 0.5 * author_sim) * year_factor

    if composite >= 0.40:
        return True

    logger.debug(
        "ID lookup mismatch for {}: composite={:.2f} "
        "(title={:.2f}, author={:.2f}, year={:.2f}), "
        "bib='{}', API='{}'",
        citation.key,
        composite,
        title_sim,
        author_sim,
        year_factor,
        citation.title[:60],
        api_title[:60],
    )
    return False


# ---------------------------------------------------------------------------
# Cross-validate multiple identifiers in a single bib entry
# ---------------------------------------------------------------------------

# Regex for extracting DOI / arXiv ID from URLs
_URL_DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s,}]+)")
_URL_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})")


def _results_match(
    canonical: dict[str, Any],
    other: dict[str, Any],
) -> bool:
    """Check whether two API results refer to the same paper.

    Reuses ``_id_result_matches`` by building a temporary Citation from the
    canonical result, so the exact same scoring logic (title × author × year,
    threshold 0.40) applies.
    """
    proxy = Citation(
        key="_cross_validate",
        raw_text="",
        title=canonical.get("title") or "",
        authors=canonical.get("authors") or [],
        year=str(canonical["year"]) if canonical.get("year") is not None else "",
    )
    return _id_result_matches(proxy, other)


async def cross_validate_ids(
    citation: Citation,
    canonical: dict[str, Any],
    config: LintConfig | None = None,
    client: httpx.AsyncClient | None = None,
) -> list[str]:
    """Look up each bib ID independently via OpenAlex and verify they resolve to the same paper.

    Compares metadata from each secondary ID lookup against the canonical
    result using composite scoring (title × author × year). Reports
    conflicting identifiers where the score falls below 0.40.

    Uses the shared ``_openalex_limiter`` for rate limiting. Pass an existing
    ``client`` to reuse connections; if None, creates a temporary one.

    Returns list of issue strings (empty if all IDs agree or only one ID exists).
    """
    config = config or LintConfig()
    canon_title = (canonical.get("title") or "")[:60]

    # Collect secondary IDs to check — skip the one used for the canonical lookup
    canon_doi = (canonical.get("doi") or "").lower()
    canon_arxiv = canonical.get("arxiv_id") or ""
    canon_pmcid = canonical.get("pmcid") or ""

    lookups: list[tuple[str, str, str]] = []  # (id_type, id_value, oa_url_or_filter)

    # DOI
    if citation.doi:
        clean_doi = clean_and_validate_doi(citation.doi)
        if clean_doi and clean_doi.lower() != canon_doi:
            lookups.append(("DOI", clean_doi, f"doi:{clean_doi}"))

    # arXiv ID
    if citation.arxiv_id and is_valid_arxiv_id(citation.arxiv_id):
        if citation.arxiv_id != canon_arxiv:
            arxiv_doi = f"{_ARXIV_DOI_PREFIX}{citation.arxiv_id}"
            lookups.append(("arXiv ID", citation.arxiv_id, f"doi:{arxiv_doi}"))

    # PMID
    if citation.pmid and is_valid_pmid(citation.pmid):
        lookups.append(
            (
                "PMID",
                citation.pmid,
                f"filter:ids.pmid:https://pubmed.ncbi.nlm.nih.gov/{citation.pmid}",
            )
        )

    # PMC ID
    if citation.pmc_id and is_valid_pmcid(citation.pmc_id):
        if citation.pmc_id != canon_pmcid:
            lookups.append(
                (
                    "PMC ID",
                    citation.pmc_id,
                    f"filter:ids.pmcid:https://www.ncbi.nlm.nih.gov/pmc/articles/{citation.pmc_id}",
                )
            )

    # URL — extract embedded DOI or arXiv ID from the URL text.
    # If no ID in the URL, HEAD-request it to follow redirects and
    # extract a DOI from the final URL (many publisher URLs redirect
    # through doi.org).
    url_needs_resolve = False
    if citation.url:
        url_doi_match = _URL_DOI_RE.search(citation.url)
        url_arxiv_match = _URL_ARXIV_RE.search(citation.url)
        if url_doi_match:
            url_doi = url_doi_match.group(1).rstrip(".")
            clean_url_doi = clean_and_validate_doi(url_doi)
            if clean_url_doi and clean_url_doi.lower() != canon_doi:
                lookups.append(("URL (DOI)", clean_url_doi, f"doi:{clean_url_doi}"))
        elif url_arxiv_match:
            url_arxiv = url_arxiv_match.group(1)
            if url_arxiv != canon_arxiv:
                url_arxiv_doi = f"{_ARXIV_DOI_PREFIX}{url_arxiv}"
                lookups.append(("URL (arXiv)", url_arxiv, f"doi:{url_arxiv_doi}"))
        else:
            # No ID in URL text — will HEAD-request to resolve
            url_needs_resolve = True

    if not lookups and not url_needs_resolve:
        return []

    issues: list[str] = []
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=15.0, follow_redirects=True)
    try:
        # Resolve URL via HEAD request if no DOI/arXiv was in the URL text
        if url_needs_resolve:
            try:
                resolved_doi = await _resolve_url_to_doi(citation.url)
                if resolved_doi and resolved_doi.lower() != canon_doi:
                    lookups.append(
                        ("URL (resolved)", resolved_doi, f"doi:{resolved_doi}")
                    )
                elif not resolved_doi:
                    # URL is alive but no DOI in the final URL — can't validate
                    issues.append(
                        f"Unverified URL identifier: URL '{citation.url[:80]}' "
                        f"could not be cross-validated (no DOI in resolved URL)"
                    )
            except Exception as e:
                logger.debug("URL resolve failed for {}: {}", citation.url, e)
                issues.append(
                    f"Unverified URL identifier: URL '{citation.url[:80]}' "
                    f"could not be cross-validated ({e})"
                )

        for id_type, id_value, oa_query in lookups:
            is_url = id_type.startswith("URL")
            try:
                other = await _oa_lookup_by_query(oa_query, id_type, client)
                if other is None:
                    # Lookup returned no result — warn that we couldn't validate
                    if is_url:
                        issues.append(
                            f"Unverified URL identifier: {id_type} '{id_value}' "
                            f"could not be cross-validated (no OpenAlex record)"
                        )
                    else:
                        issues.append(
                            f"Unverified identifier mismatch: {id_type} '{id_value}' "
                            f"could not be cross-validated (no OpenAlex record)"
                        )
                    continue

                if not _results_match(canonical, other):
                    other_title = (other.get("title") or "")[:60]
                    issues.append(
                        f"Identifier mismatch: {id_type} '{id_value}' resolves to "
                        f"'{other_title}', "
                        f"not matching the verified paper '{canon_title}'"
                    )
                    logger.debug(
                        "Cross-validate mismatch for {}: {} '{}' → '{}'",
                        citation.key,
                        id_type,
                        id_value,
                        other_title,
                    )
            except Exception as e:
                logger.debug(
                    "Cross-validate lookup failed for {} {}: {}",
                    id_type,
                    id_value,
                    e,
                )
                continue
    finally:
        if owns_client and client is not None:
            await client.aclose()

    return issues


async def _oa_lookup_by_query(
    oa_query: str,
    id_type: str,
    client: httpx.AsyncClient,
) -> dict[str, Any] | None:
    """Single OpenAlex lookup for cross-validation. Returns parsed result or None."""
    if oa_query.startswith("filter:"):
        filter_str = oa_query.removeprefix("filter:")
        resp = await rate_limited_get(
            _openalex_limiter,
            "https://api.openalex.org/works",
            params={"filter": filter_str, "select": _OA_FIELDS},
            label=f"OpenAlex cross-validate {id_type}",
            client=client,
            service="openalex",
        )
        if resp.status_code != 200:
            return None
        results = resp.json().get("results", [])
        return _parse_openalex(results[0]) if results else None

    resp = await rate_limited_get(
        _openalex_limiter,
        f"https://api.openalex.org/works/{oa_query}",
        params={"select": _OA_FIELDS},
        label=f"OpenAlex cross-validate {id_type}",
        client=client,
        service="openalex",
    )
    if resp.status_code != 200:
        return None
    return _parse_openalex(resp.json())


async def _resolve_url_to_doi(
    url: str,
) -> str | None:
    """HEAD-request a URL, follow redirects, extract DOI from the final URL.

    Many publisher URLs redirect through doi.org or embed the DOI in the
    final landing page URL. Returns cleaned DOI or None if the URL is
    alive but contains no DOI.

    Uses SSRF-safe transport (blocks private/reserved IPs) and validates
    URL scheme (https/http only). Raises on network errors — caller
    decides how to report. Tracks usage via ``usage.tracked("fetch")``.
    """
    from urllib.parse import urlparse

    from sciwrite_lint._network import ssrf_safe_client
    from sciwrite_lint.usage import tracked

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r}")

    async with tracked("fetch"):
        async with ssrf_safe_client(timeout=10.0) as client:
            resp = await client.head(url)
            if resp.status_code == 405:
                resp = await client.get(url)
            final_url = str(resp.url)

    doi_match = _URL_DOI_RE.search(final_url)
    if doi_match:
        raw_doi = doi_match.group(1).rstrip(".")
        return clean_and_validate_doi(raw_doi)
    return None


def _first_surname(authors: list[str] | tuple[str, ...]) -> str:
    """Extract the surname of the first author for search enrichment."""
    if not authors:
        return ""
    name = authors[0].strip()
    # "Smith" or "J Smith" or "John Smith" → "Smith" (last token)
    # "Smith, J" → "Smith" (first token before comma)
    if "," in name:
        return name.split(",")[0].strip()
    parts = name.split()
    return parts[-1] if parts else ""


def _search_query(citation: Citation) -> str:
    """Build a search query string from a citation.

    Includes first author surname when available — improves API hit rate
    and reduces wrong-paper matches.
    """
    query = citation.title or citation.raw_text[:120]
    # Strip everything except word chars, spaces, and hyphens to avoid
    # breaking OpenAlex filter syntax (commas, apostrophes, colons, etc.).
    query = re.sub(r"[^\w\s-]", "", query).strip()
    surname = _first_surname(citation.authors)
    if surname:
        query = f"{surname} {query}"
    return query


# ---------------------------------------------------------------------------
# Verification pipeline
# ---------------------------------------------------------------------------


async def verify_citations(
    citations: list[Citation],
    api: CitationAPI | None = None,
    config: LintConfig | None = None,
    references_dir: Path | None = None,
    progress: bool = True,
    save: bool = True,
) -> None:
    """Verify a list of citations against APIs. Modifies citations in place.

    Web resources (@misc with URL, no DOI) skip academic APIs and instead
    verify the URL is alive + download content as markdown.
    """
    from sciwrite_lint.references.citations import is_web_resource
    from sciwrite_lint.references.metadata import (
        build_metadata_from_citation,
        load_metadata,
        merge_source_paper,
        save_metadata,
    )

    config = config or LintConfig()
    refs_dir = references_dir or config.effective_references_dir()

    own_api = api is None
    if own_api:
        api = CitationAPI(config=config)
    assert api is not None  # narrowing for mypy

    try:
        total = len(citations)
        for i, c in enumerate(citations, 1):
            if progress:
                print(f"  [{i}/{total}] {c.key}...", end="", flush=True)

            if is_web_resource(c):
                result = await _verify_web_resource(c, api._client, refs_dir)
            else:
                result = await _verify_academic(c, api, refs_dir)

            # Persist metadata
            if save:
                existing = load_metadata(c.key, refs_dir)
                if (
                    existing
                    and existing.api_match == "verified"
                    and c.api_match == "not_found"
                ):
                    merge_source_paper(existing, c.source_paper)
                    save_metadata(existing, refs_dir)
                    c.tier = existing.access.get("tier", "")
                else:
                    meta = build_metadata_from_citation(
                        c, result, references_dir=refs_dir
                    )
                    if existing:
                        merge_source_paper(meta, c.source_paper)
                        for sp in existing.bibitem.get("source_papers", []):
                            merge_source_paper(meta, sp)
                        if existing.manual_override:
                            meta.manual_override = existing.manual_override
                    save_metadata(meta, refs_dir)
                    c.tier = meta.access.get("tier", "")

            if progress:
                tier_str = f" [{c.tier}]" if c.tier else ""
                print(f" {c.api_match}{tier_str}")
    finally:
        if own_api:
            await api.close()


async def _verify_web_resource(
    c: Citation,
    client: httpx.AsyncClient,
    references_dir: Path,
) -> dict | None:
    """Verify a web resource citation: check URL + download content."""
    from sciwrite_lint.web import fetch_web_content

    web_result = await fetch_web_content(c.url, c.key, references_dir, client)
    resolved_url = web_result.resolved_url or c.url

    if web_result.url_alive:
        c.api_match = "web_verified"
        c.api_source = "web"
        c.api_data = {
            "source": "web",
            "url": resolved_url,
            "status_code": web_result.status_code,
            "content_type": web_result.content_type,
            "title": web_result.title,
        }
        if web_result.local_path:
            c.local_status = "md"
            c.local_path = str(references_dir / web_result.local_path)
            c.issues.append(f"Web content saved: {web_result.local_path}")
        else:
            c.issues.append(
                f"URL alive but content extraction failed: {web_result.error or 'unknown'}"
            )
    elif web_result.blocked:
        # Unverifiable: 4xx refusal, 5xx error, TLS/timeout/connection/
        # decoding/protocol/oversized. The URL may still be valid — we
        # just could not confirm. Downstream emits a WARN telling the
        # user to verify manually; distinct from web_dead (ERROR).
        c.api_match = "web_blocked"
        c.api_source = "web"
        c.api_data = {
            "source": "web",
            "url": resolved_url,
            "status_code": web_result.status_code,
        }
        reason = web_result.error or "unknown reason"
        c.issues.append(
            f"Blocked by {reason}: {resolved_url} — "
            f"unable to verify automatically, please check manually"
        )
    else:
        # Genuinely dead: server explicitly returned 404 or 410 (or the
        # URL was structurally invalid). Safe to flag as ERROR — cited
        # resource cannot be retrieved at this URL and likely cannot be
        # retrieved anywhere at this URL.
        c.api_match = "web_dead"
        c.api_source = "web"
        c.api_data = {"source": "web", "url": resolved_url}
        reason = web_result.error or f"HTTP {web_result.status_code}"
        c.issues.append(f"Dead URL ({reason}): {resolved_url}")

    return c.api_data


async def _verify_academic(
    c: Citation,
    api: CitationAPI,
    references_dir: Path | None = None,
) -> dict | None:
    """Verify an academic citation against CrossRef, OpenAlex, S2.

    When all academic APIs fail but the citation has a URL, verifies the URL
    is alive as a last resort before marking not_found.
    """
    from sciwrite_lint.references.matching import compare_citation_detailed

    result = await api.lookup(c)
    if result and not result.get("error"):
        c.api_data = result
        c.api_source = result.get("source", "")
        c.issues.extend(compare_citation_detailed(c, result))
        c.issues.extend(await cross_validate_ids(c, result, api._config, api._client))
        c.api_match = (
            "mismatch" if any("mismatch" in i.lower() for i in c.issues) else "verified"
        )
    elif c.url and references_dir is not None:
        # All academic APIs failed but the citation has a URL —
        # verify URL liveness as last resort before marking T3.
        logger.debug("Academic APIs failed for '{}'; trying URL: {}", c.key, c.url)
        result = await _verify_web_resource(c, api._client, references_dir)
    else:
        c.api_match = "not_found"
        c.issues.append(
            "Not found in CrossRef, OpenAlex, Semantic Scholar, Open Library, or Library of Congress"
        )

    return result
