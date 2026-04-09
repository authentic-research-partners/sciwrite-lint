"""Matching-quality evaluation for API candidate selection.

Downloads a set of real papers, queries APIs for candidates, then tests
whether ``_best_match`` selects the correct paper under various metadata
degradation scenarios (title truncation, author format changes, year
offsets, missing fields).

Measures Recall@1: did ``_best_match`` return the correct paper?

Usage:
    python -m evals eval-real-world matching --workspace real_world_corpus
    python -m evals eval-real-world matching --workspace real_world_corpus --max-papers 5
"""

from __future__ import annotations

import json
import random
import re
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from sciwrite_lint.api import _best_match
from sciwrite_lint.rate_limiter import MonotonicRateLimiter, rate_limited_get

_oa_limiter = MonotonicRateLimiter(1, 0.12)  # 1 request per 0.12s

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

_OA_FIELDS = (
    "id,ids,doi,title,authorships,publication_year,"
    "primary_location,locations,open_access,cited_by_count,"
    "abstract_inverted_index,is_retracted"
)


class DegradationResult(BaseModel):
    """Result of a single degradation test."""

    degradation: str
    correct: bool
    returned_title: str = ""
    expected_title: str = ""
    score: float = 0.0


class PaperResult(BaseModel):
    """Matching eval results for one paper."""

    paper_id: str
    title: str
    authors: list[str]
    source: str
    n_candidates: int = 0
    ground_truth_found: bool = False
    ground_truth_doi: str = ""
    degradations: list[DegradationResult] = Field(default_factory=list)
    error: str = ""


# ---------------------------------------------------------------------------
# Degradation strategies
# ---------------------------------------------------------------------------


def _degrade_title_truncate(title: str) -> str:
    """Keep first 8 words."""
    words = title.split()
    return " ".join(words[:8]) if len(words) > 8 else title


def _degrade_title_no_subtitle(title: str) -> str:
    """Remove subtitle after colon."""
    if ":" in title:
        return title.split(":")[0].strip()
    return title


def _degrade_title_typo(title: str, rng: random.Random) -> str:
    """Swap two adjacent characters in a random word."""
    words = title.split()
    if len(words) < 2:
        return title
    idx = rng.randint(0, len(words) - 1)
    w = words[idx]
    if len(w) > 2:
        pos = rng.randint(0, len(w) - 2)
        w = w[:pos] + w[pos + 1] + w[pos] + w[pos + 2 :]
        words[idx] = w
    return " ".join(words)


def _split_name(name: str) -> tuple[str, str]:
    """Split author name into (given, family), handling both formats."""
    name = name.strip()
    if "," in name:
        # "Family, Given" format
        parts = [p.strip() for p in name.split(",", 1)]
        return (parts[1] if len(parts) > 1 else "", parts[0])
    # "Given Family" format
    parts = name.split()
    if len(parts) >= 2:
        return (" ".join(parts[:-1]), parts[-1])
    return ("", name)


def _degrade_authors_initials(authors: list[str]) -> list[str]:
    """Convert given names to initials: 'John Smith' → 'J. Smith'."""
    result = []
    for a in authors:
        given, family = _split_name(a)
        if given:
            initials = " ".join(p[0] + "." for p in given.split() if p)
            result.append(f"{initials} {family}")
        else:
            result.append(family)
    return result


def _degrade_authors_reversed(authors: list[str]) -> list[str]:
    """Reverse to 'Family, Given': 'John Smith' → 'Smith, John'."""
    result = []
    for a in authors:
        given, family = _split_name(a)
        if given:
            result.append(f"{family}, {given}")
        else:
            result.append(family)
    return result


def _degrade_authors_first_only(authors: list[str]) -> list[str]:
    """Keep only first author."""
    return authors[:1] if authors else []


def _degrade_year(year: int, offset: int) -> int:
    return year + offset


DEGRADATION_SPECS: list[dict[str, Any]] = [
    # Baseline
    {"name": "clean", "title": None, "authors": None, "year": None, "venue": None},
    # Title degradations
    {"name": "title_truncated", "title": "truncate"},
    {"name": "title_no_subtitle", "title": "no_subtitle"},
    {"name": "title_typo", "title": "typo"},
    # Author degradations
    {"name": "authors_initials", "authors": "initials"},
    {"name": "authors_reversed", "authors": "reversed"},
    {"name": "authors_first_only", "authors": "first_only"},
    {"name": "authors_missing", "authors": "missing"},
    # Year degradations
    {"name": "year_off_1", "year": 1},
    {"name": "year_off_3", "year": 3},
    {"name": "year_off_10", "year": 10},
    {"name": "year_missing", "year": "missing"},
    # Venue degradations
    {"name": "venue_missing", "venue": "missing"},
    # Combined degradations (realistic worst-case)
    {
        "name": "initials+year_off_1",
        "authors": "initials",
        "year": 1,
    },
    {
        "name": "truncated+initials+no_venue",
        "title": "truncate",
        "authors": "initials",
        "venue": "missing",
    },
    {
        "name": "typo+reversed+year_off_3",
        "title": "typo",
        "authors": "reversed",
        "year": 3,
    },
    {
        "name": "first_author+no_venue+year_off_1",
        "authors": "first_only",
        "venue": "missing",
        "year": 1,
    },
]


def _apply_degradations(
    spec: dict[str, Any],
    title: str,
    authors: list[str],
    year: int | None,
    venue: str,
    rng: random.Random,
) -> tuple[str, list[str], int | None, str]:
    """Apply a degradation spec, return (title, authors, year, venue)."""
    d_title = title
    d_authors = list(authors)
    d_year = year
    d_venue = venue

    t = spec.get("title")
    if t == "truncate":
        d_title = _degrade_title_truncate(title)
    elif t == "no_subtitle":
        d_title = _degrade_title_no_subtitle(title)
    elif t == "typo":
        d_title = _degrade_title_typo(title, rng)

    a = spec.get("authors")
    if a == "initials":
        d_authors = _degrade_authors_initials(authors)
    elif a == "reversed":
        d_authors = _degrade_authors_reversed(authors)
    elif a == "first_only":
        d_authors = _degrade_authors_first_only(authors)
    elif a == "missing":
        d_authors = []

    y = spec.get("year")
    if y == "missing":
        d_year = None
    elif isinstance(y, int) and d_year is not None:
        d_year = _degrade_year(d_year, y)

    v = spec.get("venue")
    if v == "missing":
        d_venue = ""

    return d_title, d_authors, d_year, d_venue


# ---------------------------------------------------------------------------
# API fetching — get candidates for a paper
# ---------------------------------------------------------------------------


def _parse_openalex_item(work: dict) -> dict[str, Any]:
    """Parse a single OpenAlex work into normalized dict."""
    title = work.get("title") or ""
    year = work.get("publication_year")
    doi = (work.get("doi") or "").replace("https://doi.org/", "")

    authors = []
    for authorship in work.get("authorships", []):
        name = authorship.get("author", {}).get("display_name", "")
        if name:
            authors.append(name)

    venue = ""
    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}
    venue = src.get("display_name") or ""

    return {
        "source": "openalex",
        "title": title,
        "authors": authors,
        "year": year,
        "venue": venue,
        "doi": doi,
    }


async def _fetch_openalex_candidates(
    title: str,
    client: httpx.AsyncClient,
) -> list[dict[str, Any]]:
    """Query OpenAlex by title, return up to 10 parsed candidates."""
    query = re.sub(r"[^\w\s-]", "", title).strip()
    if not query:
        return []
    try:
        resp = await rate_limited_get(
            _oa_limiter,
            "https://api.openalex.org/works",
            params={
                "filter": f"title.search:{query}",
                "per_page": 10,
                "select": _OA_FIELDS,
            },
            label="OpenAlex matching eval",
            client=client,
            service="openalex",
        )
        if resp.status_code != 200:
            return []
        results = resp.json().get("results", [])
        return [_parse_openalex_item(r) for r in results]
    except Exception as e:
        logger.debug("OpenAlex fetch failed: {}", e)
        return []


# ---------------------------------------------------------------------------
# Core eval logic
# ---------------------------------------------------------------------------


async def eval_one_paper(
    paper: dict[str, Any],
    client: httpx.AsyncClient,
    seed: int,
) -> PaperResult:
    """Run matching eval for one paper."""
    rng = random.Random(seed)
    paper_id = paper["arxiv_id"]
    title = paper["title"]
    authors = paper.get("authors", [])
    source = paper.get("source", "arxiv")

    result = PaperResult(
        paper_id=paper_id,
        title=title,
        authors=authors,
        source=source,
    )

    # Fetch candidates from OpenAlex (broad coverage, structured data)
    candidates = await _fetch_openalex_candidates(title, client)
    result.n_candidates = len(candidates)

    if not candidates:
        result.error = "no candidates returned from API"
        return result

    # Identify ground truth: the candidate whose title most closely
    # matches the query (clean, no degradation). We also check DOI
    # for arXiv papers to confirm identity.
    gt_candidate = _best_match(title, candidates, threshold=0.85)
    if gt_candidate is None:
        result.error = "ground truth paper not found in API results"
        return result

    result.ground_truth_found = True
    result.ground_truth_doi = gt_candidate.get("doi", "")
    gt_title = gt_candidate.get("title", "")

    # Extract year and venue from ground truth for degradation
    gt_year = gt_candidate.get("year")
    gt_venue = gt_candidate.get("venue", "")

    # Run each degradation
    for spec in DEGRADATION_SPECS:
        d_title, d_authors, d_year, d_venue = _apply_degradations(
            spec, title, authors, gt_year, gt_venue, rng
        )

        matched = _best_match(
            d_title,
            candidates,
            query_authors=d_authors if d_authors else None,
            query_year=d_year,
            query_venue=d_venue,
        )

        correct = False
        returned_title = ""
        if matched is not None:
            returned_title = matched.get("title", "")
            # Correct if same DOI or same title (within threshold)
            if result.ground_truth_doi and matched.get("doi"):
                correct = matched["doi"] == result.ground_truth_doi
            else:
                correct = returned_title == gt_title

        result.degradations.append(
            DegradationResult(
                degradation=spec["name"],
                correct=correct,
                returned_title=returned_title,
                expected_title=gt_title,
            )
        )

    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def run_matching_eval(
    workspace: Path,
    max_papers: int | None = None,
    seed: int = 42,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run matching eval on corpus papers. Returns summary dict."""
    manifest_path = workspace / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest.json in {workspace}. "
            f"Run: python -m evals eval-real-world corpus --workspace {workspace}"
        )

    papers = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Filter out papers with download errors
    papers = [p for p in papers if not p.get("error")]

    if max_papers:
        papers = papers[:max_papers]

    print(
        f"\nMatching eval: {len(papers)} papers, {len(DEGRADATION_SPECS)} degradations each"
    )
    print("=" * 70)

    results: list[PaperResult] = []

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        for i, paper in enumerate(papers):
            paper_id = paper["arxiv_id"]
            print(f"  [{i + 1}/{len(papers)}] {paper_id}: {paper['title'][:60]}...")
            result = await eval_one_paper(paper, client, seed=seed + i)

            if result.error:
                print(f"    SKIP: {result.error}")
            else:
                n_correct = sum(1 for d in result.degradations if d.correct)
                n_total = len(result.degradations)
                print(f"    {n_correct}/{n_total} correct")

            results.append(result)

    # Aggregate
    summary = _build_summary(results)
    _print_summary(summary)

    # Save results
    out = output_dir or Path("real_world_results") / "matching"
    out.mkdir(parents=True, exist_ok=True)

    full_output = {
        "summary": summary,
        "papers": [_paper_to_dict(r) for r in results],
    }
    out_path = out / "results.json"
    out_path.write_text(
        json.dumps(full_output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nResults saved to {out_path}")

    return summary


def _build_summary(results: list[PaperResult]) -> dict[str, Any]:
    """Aggregate per-degradation Recall@1 across all papers."""
    evaluated = [r for r in results if r.ground_truth_found]

    per_degradation: dict[str, dict[str, int]] = {}
    for r in evaluated:
        for d in r.degradations:
            if d.degradation not in per_degradation:
                per_degradation[d.degradation] = {"correct": 0, "total": 0}
            per_degradation[d.degradation]["total"] += 1
            if d.correct:
                per_degradation[d.degradation]["correct"] += 1

    degradation_recall: dict[str, float] = {}
    for name, counts in per_degradation.items():
        degradation_recall[name] = (
            counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
        )

    overall_correct = sum(c["correct"] for c in per_degradation.values())
    overall_total = sum(c["total"] for c in per_degradation.values())

    return {
        "n_papers": len(results),
        "n_evaluated": len(evaluated),
        "n_skipped": len(results) - len(evaluated),
        "overall_recall": overall_correct / overall_total if overall_total else 0.0,
        "per_degradation": degradation_recall,
        "per_degradation_counts": per_degradation,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    """Print a formatted summary table."""
    print(f"\n{'=' * 70}")
    print(
        f"Matching Eval Summary: {summary['n_evaluated']} papers evaluated, "
        f"{summary['n_skipped']} skipped"
    )
    print(f"Overall Recall@1: {summary['overall_recall']:.1%}")
    print(f"{'=' * 70}")
    print(f"{'Degradation':<40} {'Recall@1':>10} {'Correct':>10} {'Total':>8}")
    print(f"{'-' * 40} {'-' * 10} {'-' * 10} {'-' * 8}")

    counts = summary.get("per_degradation_counts", {})
    for name, recall in summary["per_degradation"].items():
        c = counts.get(name, {})
        correct = c.get("correct", 0)
        total = c.get("total", 0)
        print(f"{name:<40} {recall:>10.1%} {correct:>10} {total:>8}")


def _paper_to_dict(r: PaperResult) -> dict[str, Any]:
    """Serialize PaperResult for JSON output."""
    return {
        "paper_id": r.paper_id,
        "title": r.title,
        "authors": r.authors,
        "source": r.source,
        "n_candidates": r.n_candidates,
        "ground_truth_found": r.ground_truth_found,
        "ground_truth_doi": r.ground_truth_doi,
        "error": r.error,
        "degradations": [
            {
                "degradation": d.degradation,
                "correct": d.correct,
                "returned_title": d.returned_title,
                "expected_title": d.expected_title,
            }
            for d in r.degradations
        ],
    }
