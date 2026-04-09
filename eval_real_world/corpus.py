"""Download papers (LaTeX/PDF) from arXiv and bioRxiv for real-world evaluation.

arXiv provides LaTeX source — ideal for text-rule evaluation.
bioRxiv provides PDFs only — exercises the GROBID parsing path.

Usage:
    from eval_real_world.corpus import build_corpus
    papers = await build_corpus(n=100, categories=["cs.CL", "cs.AI"])
"""

from __future__ import annotations

import asyncio
import gzip
import io
import random
import tarfile
import xml.etree.ElementTree as ET
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Any

import httpx

from sciwrite_lint.rate_limiter import MonotonicRateLimiter, retry_on_transient

# arXiv API base (Atom feed)
ARXIV_API = "https://export.arxiv.org/api/query"
# arXiv e-print source download
ARXIV_EPRINT = "https://export.arxiv.org/e-print"

# bioRxiv API
BIORXIV_API = "https://api.biorxiv.org/details/biorxiv"

# Default categories — broad distribution across disciplines
DEFAULT_CATEGORIES = [
    # Computer science
    "cs.CL",
    "cs.AI",
    "cs.LG",
    # Physics
    "hep-ph",
    "cond-mat.mtrl-sci",
    # Mathematics / statistics
    "math.ST",
    "stat.ML",
    # Economics
    "econ.EM",
    # Quantitative biology
    "q-bio.QM",
]

# Atom namespace
ATOM_NS = "{http://www.w3.org/2005/Atom}"

# Polite delay between API requests
_arxiv_limiter = MonotonicRateLimiter(1, 3.0)  # 1 req per 3 seconds
_biorxiv_limiter = MonotonicRateLimiter(1, 1.0)  # 1 req per second


class ArxivPaper(BaseModel):
    """Metadata for a downloaded paper."""

    arxiv_id: str
    title: str
    authors: list[str]
    categories: list[str]
    source: str = "arxiv"  # "arxiv" or "biorxiv"
    tex_path: Path | None = None
    pdf_path: Path | None = None
    error: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


async def _search_arxiv(
    categories: list[str],
    max_results: int = 200,
    start: int = 0,
) -> list[dict[str, Any]]:
    """Query arXiv API for recent papers in given categories.

    Returns list of dicts with id, title, authors, categories.
    Fetches in batches of 100 (arXiv API limit per request).
    """
    cat_query = " OR ".join(f"cat:{c}" for c in categories)
    results: list[dict[str, Any]] = []
    batch_size = min(100, max_results)

    async with httpx.AsyncClient(timeout=90) as client:
        while len(results) < max_results:
            params: dict[str, str | int] = {
                "search_query": cat_query,
                "start": start + len(results),
                "max_results": min(batch_size, max_results - len(results)),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }

            async with _arxiv_limiter:
                pass
            resp = await retry_on_transient(
                lambda: client.get(ARXIV_API, params=params),
                label="arXiv search",
            )
            resp.raise_for_status()

            root = ET.fromstring(resp.text)
            entries = root.findall(f"{ATOM_NS}entry")
            if not entries:
                break

            for entry in entries:
                title_el = entry.find(f"{ATOM_NS}title")
                id_el = entry.find(f"{ATOM_NS}id")
                if title_el is None or id_el is None:
                    continue

                arxiv_url = (id_el.text or "").strip()
                # Extract ID: http://arxiv.org/abs/2603.12345v1 -> 2603.12345
                arxiv_id = arxiv_url.split("/abs/")[-1]
                if "v" in arxiv_id:
                    arxiv_id = arxiv_id.rsplit("v", 1)[0]

                authors = [
                    name_el.text or ""
                    for a in entry.findall(f"{ATOM_NS}author")
                    if (name_el := a.find(f"{ATOM_NS}name")) is not None
                ]

                cats = [
                    c.get("term", "")
                    for c in entry.findall("{http://arxiv.org/schemas/atom}category")
                ]

                results.append(
                    {
                        "arxiv_id": arxiv_id,
                        "title": " ".join((title_el.text or "").split()),
                        "authors": authors,
                        "categories": cats,
                    }
                )

            if len(entries) < batch_size:
                break

    return results


async def _download_source(arxiv_id: str, workspace: Path) -> Path | None:
    """Download and extract LaTeX source for an arXiv paper.

    Returns path to the main .tex file, or None if extraction fails.
    """
    paper_dir = workspace / arxiv_id.replace("/", "_")
    paper_dir.mkdir(parents=True, exist_ok=True)

    url = f"{ARXIV_EPRINT}/{arxiv_id}"
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        resp = await retry_on_transient(
            lambda: client.get(url),
            label=f"arXiv download {arxiv_id}",
        )
        resp.raise_for_status()

    content = resp.content

    # arXiv returns either a tar.gz, a gzipped single file, or raw LaTeX
    tex_files: list[Path] = []

    try:
        # Try tar.gz first
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
            # Security: skip absolute paths and ..
            safe_members = [
                m
                for m in tar.getmembers()
                if not m.name.startswith("/")
                and ".." not in m.name
                and m.isfile()
                and any(
                    m.name.endswith(ext) for ext in (".tex", ".bib", ".bbl", ".sty")
                )
            ]
            tar.extractall(paper_dir, members=safe_members)
            tex_files = list(paper_dir.rglob("*.tex"))
    except (tarfile.TarError, gzip.BadGzipFile):
        pass

    if not tex_files:
        try:
            # Try gzipped single file
            text = gzip.decompress(content).decode("utf-8", errors="replace")
            if "\\begin{document}" in text or "\\documentclass" in text:
                out = paper_dir / "main.tex"
                out.write_text(text, encoding="utf-8")
                tex_files = [out]
        except (gzip.BadGzipFile, OSError):
            pass

    if not tex_files:
        # Try raw LaTeX
        try:
            text = content.decode("utf-8", errors="replace")
            if "\\begin{document}" in text or "\\documentclass" in text:
                out = paper_dir / "main.tex"
                out.write_text(text, encoding="utf-8")
                tex_files = [out]
        except UnicodeDecodeError:
            pass

    if not tex_files:
        return None

    # Find main .tex file (the one with \begin{document})
    for tf in tex_files:
        try:
            text = tf.read_text(encoding="utf-8", errors="replace")
            if "\\begin{document}" in text:
                return tf
        except OSError:
            continue

    # Fallback: largest .tex file
    return max(tex_files, key=lambda f: f.stat().st_size)


async def _search_biorxiv(
    max_results: int = 100,
) -> list[dict[str, Any]]:
    """Query bioRxiv API for recent preprints.

    Returns list of dicts with id, title, authors, categories.
    Uses the /details endpoint which returns recent papers in reverse-chronological order.
    """
    results: list[dict[str, Any]] = []
    # bioRxiv API pages by date range; use a recent 30-day window
    from datetime import datetime, timedelta

    end = datetime.now()
    start = end - timedelta(days=30)
    date_range = f"{start:%Y-%m-%d}/{end:%Y-%m-%d}"

    cursor = 0
    async with httpx.AsyncClient(timeout=90) as client:
        while len(results) < max_results:
            url = f"{BIORXIV_API}/{date_range}/{cursor}"
            async with _biorxiv_limiter:
                pass
            resp = await retry_on_transient(
                lambda: client.get(url),
                label="bioRxiv search",
            )
            resp.raise_for_status()

            data = resp.json()
            messages = data.get("messages", [{}])
            total = int(messages[0].get("total", 0)) if messages else 0
            collection = data.get("collection", [])

            if not collection:
                break

            for item in collection:
                doi = item.get("doi", "")
                title = item.get("title", "")
                authors_str = item.get("authors", "")
                authors = [a.strip() for a in authors_str.split(";") if a.strip()]
                category = item.get("category", "")

                results.append(
                    {
                        "arxiv_id": doi,  # reuse field name for consistency
                        "title": title,
                        "authors": authors,
                        "categories": [category] if category else [],
                        "source": "biorxiv",
                    }
                )

                if len(results) >= max_results:
                    break

            cursor += len(collection)
            if cursor >= total:
                break

    return results


async def _download_biorxiv_pdf(doi: str, workspace: Path) -> Path | None:
    """Download PDF for a bioRxiv paper by DOI.

    Returns path to the downloaded PDF, or None if download fails.
    """
    paper_dir = workspace / doi.replace("/", "_")
    paper_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = paper_dir / "paper.pdf"
    if pdf_path.exists() and pdf_path.stat().st_size > 5000:
        return pdf_path

    url = f"https://www.biorxiv.org/content/{doi}v1.full.pdf"
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        resp = await retry_on_transient(
            lambda: client.get(url),
            label=f"bioRxiv PDF {doi}",
        )
        resp.raise_for_status()

    if not resp.content[:5] == b"%PDF-":
        return None

    if len(resp.content) < 5000:
        return None

    pdf_path.write_bytes(resp.content)
    return pdf_path


async def build_corpus(
    workspace: Path,
    n: int = 100,
    categories: list[str] | None = None,
    seed: int | None = 42,
    sources: list[str] | None = None,
) -> list[ArxivPaper]:
    """Download n papers into workspace directory.

    Args:
        workspace: Directory to store downloaded papers.
        n: Number of papers to download.
        categories: arXiv categories (default: broad cross-discipline set).
        seed: Random seed for sampling (None = no shuffle).
        sources: Which sources to use (default: ["arxiv", "biorxiv"]).
            "arxiv" provides LaTeX source, "biorxiv" provides PDFs.

    Returns:
        List of ArxivPaper with tex_path/pdf_path set for successful downloads.
    """
    active_sources = sources or ["arxiv", "biorxiv"]
    cats = categories or DEFAULT_CATEGORIES
    workspace.mkdir(parents=True, exist_ok=True)

    # Allocate quota per source
    source_count = len(active_sources)
    per_source = n // source_count
    remainder = n % source_count

    candidates: list[dict[str, Any]] = []

    if "arxiv" in active_sources:
        arxiv_n = per_source + (1 if remainder > 0 else 0)
        remainder = max(0, remainder - 1)
        print(f"Searching arXiv for papers in {', '.join(cats)}...")
        arxiv_candidates = await _search_arxiv(cats, max_results=arxiv_n * 3)
        for c in arxiv_candidates:
            c["source"] = "arxiv"
        candidates.extend(arxiv_candidates)
        print(f"  Found {len(arxiv_candidates)} arXiv candidates")

    if "biorxiv" in active_sources:
        biorxiv_n = per_source + (1 if remainder > 0 else 0)
        print("Searching bioRxiv for recent preprints...")
        biorxiv_candidates = await _search_biorxiv(max_results=biorxiv_n * 3)
        candidates.extend(biorxiv_candidates)
        print(f"  Found {len(biorxiv_candidates)} bioRxiv candidates")

    if seed is not None:
        random.seed(seed)
        random.shuffle(candidates)

    # Separate cached from needing download
    cached: list[ArxivPaper] = []
    to_download: list[dict[str, Any]] = []
    for meta in candidates:
        if len(cached) + len(to_download) >= n:
            break
        paper_id = meta["arxiv_id"]
        source = meta.get("source", "arxiv")
        paper_dir = workspace / paper_id.replace("/", "_")

        if source == "biorxiv":
            existing_pdf = list(paper_dir.rglob("*.pdf")) if paper_dir.exists() else []
            if existing_pdf:
                p = ArxivPaper(
                    arxiv_id=paper_id,
                    title=meta["title"],
                    authors=meta["authors"],
                    categories=meta["categories"],
                    source="biorxiv",
                    pdf_path=existing_pdf[0],
                )
                cached.append(p)
                print(f"  [{len(cached)}/{n}] {paper_id} (cached, biorxiv)")
                continue
        else:
            existing_tex = list(paper_dir.rglob("*.tex")) if paper_dir.exists() else []
            if existing_tex:
                p = ArxivPaper(
                    arxiv_id=paper_id,
                    title=meta["title"],
                    authors=meta["authors"],
                    categories=meta["categories"],
                    source="arxiv",
                    tex_path=existing_tex[0],
                )
                cached.append(p)
                print(f"  [{len(cached)}/{n}] {paper_id} (cached)")
                continue

        to_download.append(meta)

    # Download concurrently with rate limiter + semaphore
    sem = asyncio.Semaphore(3)
    downloaded: list[ArxivPaper] = []

    async def _dl_one(meta: dict[str, Any]) -> ArxivPaper | None:
        paper_id = meta["arxiv_id"]
        source = meta.get("source", "arxiv")
        paper = ArxivPaper(
            arxiv_id=paper_id,
            title=meta["title"],
            authors=meta["authors"],
            categories=meta["categories"],
            source=source,
        )
        try:
            async with sem:
                if source == "biorxiv":
                    async with _biorxiv_limiter:
                        pass
                    pdf_path = await _download_biorxiv_pdf(paper_id, workspace)
                    if pdf_path:
                        paper.pdf_path = pdf_path
                        return paper
                    else:
                        paper.error = "PDF download failed or invalid"
                        return None
                else:
                    async with _arxiv_limiter:
                        pass
                    tex_path = await _download_source(paper_id, workspace)
                    if tex_path:
                        paper.tex_path = tex_path
                        return paper
                    else:
                        paper.error = "no .tex file found in source"
                        return None
        except (httpx.HTTPError, OSError) as e:
            paper.error = str(e)
            return None

    results = await asyncio.gather(*[_dl_one(m) for m in to_download])
    for r in results:
        if r is not None:
            downloaded.append(r)
            tag = f" [{r.source}]" if r.source != "arxiv" else ""
            print(
                f"  [{len(cached) + len(downloaded)}/{n}] {r.arxiv_id}{tag} — {r.title[:60]}"
            )

    papers = cached + downloaded
    arxiv_count = sum(1 for p in papers if p.source == "arxiv")
    biorxiv_count = sum(1 for p in papers if p.source == "biorxiv")
    print(
        f"\nCorpus: {len(papers)} papers ({arxiv_count} arXiv, {biorxiv_count} bioRxiv) in {workspace}"
    )
    return papers
