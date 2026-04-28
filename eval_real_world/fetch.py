"""OA fetch-only evaluation — exercise the acquisition flow end-to-end.

This runs the full 14-source OA waterfall (``acquire_fulltext``) over a
small curated reference set and reports per-source outcomes. It's
narrower than ``eval-real-world pipeline`` — no GROBID parse, no LLM,
no claim verification. The goal is a fast, focused check that:

- each source's adapter still works against live APIs (upstream HTML /
  JSON has not drifted),
- the pre-download ranker picks sensible candidates,
- the post-download validator rejects wrong PDFs,
- non-English and Unicode titles pass through the Solr escape correctly.

Usage::

    python -m evals eval-real-world fetch
    python -m evals eval-real-world fetch --output results/fetch.json
    python -m evals eval-real-world fetch --download /tmp/oa-check

Each reference is processed once; outcomes roll up into:

- Per-source attempt / success / rejection counts
- Overall success rate
- Validator rejection reasons
- Wall time

This is NOT a pass/fail eval (upstream source quality varies) — treat it
as a smoke diagnostic before shipping changes to ``fulltext/`` adapters.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from collections import Counter
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field

from sciwrite_lint.config import LintConfig
from sciwrite_lint.fulltext import acquire_fulltext


class FetchRef(BaseModel):
    """One reference to attempt OA acquisition for."""

    key: str
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    entry_type: str = "article"
    expected_source: str | None = None  # soft hint for the report


class FetchOutcome(BaseModel):
    """Per-reference result."""

    key: str
    expected_source: str | None
    actual_source: str | None = None
    found: bool
    url: str | None = None
    reason: str = ""


# Curated reference set. Each case targets a specific source in the
# waterfall so the per-source success columns are interpretable. Stretch
# goals like "fallback past wrong candidate" live in
# ``scripts/verify_oa_economics_sources.py --rank`` — this set just
# confirms the happy path end-to-end.
DEFAULT_CASES: list[FetchRef] = [
    # arXiv (direct PDF)
    FetchRef(
        key="attention_is_all_you_need",
        title="Attention Is All You Need",
        authors=["Vaswani", "Shazeer", "Parmar"],
        year=2017,
        arxiv_id="1706.03762",
        expected_source="arxiv",
    ),
    # PMC via DOI (open-access biology)
    FetchRef(
        key="pfund_mentor_training_2014",
        title="Mentor Training for Clinical and Translational Researchers",
        authors=["Pfund", "House", "Asquith"],
        year=2014,
        doi="10.1007/978-1-4939-1309-2",
        entry_type="article",
        expected_source="pmc-or-epmc",
    ),
    # bioRxiv (DOI prefix 10.1101/)
    FetchRef(
        key="biorxiv_preprint",
        title="An integrated cell atlas of the lung in health and disease",
        authors=["Sikkema"],
        year=2022,
        doi="10.1101/2022.03.10.483747",
        expected_source="biorxiv",
    ),
    # NBER (title search + author)
    FetchRef(
        key="dale_krueger_college",
        title="Estimating the payoff to attending a more selective college",
        authors=["Dale", "Krueger"],
        year=2002,
        expected_source="nber",
    ),
    # HAL (French archive, Unicode-safe title)
    FetchRef(
        key="hal_attention_unicode",
        title="Attention Is All You Need",
        authors=["Vaswani"],
        year=2017,
        expected_source="hal",
    ),
    # ERIC (education research, ED* document)
    FetchRef(
        key="eric_mentoring",
        title="Growing Insights and Innovations Research Agenda Mentoring",
        expected_source="eric",
    ),
    # OSF Preprints
    FetchRef(
        key="osf_replication_psychology",
        title="Estimating the reproducibility of psychological science",
        authors=["Open Science Collaboration"],
        year=2015,
        expected_source="osf",
    ),
]


async def _run_one(ref: FetchRef, workspace_dir: Path, email: str) -> FetchOutcome:
    """Attempt OA acquisition for a single reference."""
    result = await acquire_fulltext(
        key=ref.key,
        references_dir=workspace_dir,
        config=LintConfig(polite_email=email),
        doi=ref.doi,
        arxiv_id=ref.arxiv_id,
        expected_title=ref.title,
        expected_authors=ref.authors,
        expected_year=ref.year,
        expected_entry_type=ref.entry_type,
        progress=False,
    )
    return FetchOutcome(
        key=ref.key,
        expected_source=ref.expected_source,
        actual_source=result.source or None,
        found=result.found,
        url=result.url,
        reason=result.reason,
    )


async def run_fetch_eval(
    output_dir: Path | None = None,
    download_to: Path | None = None,
    email: str = "",
    cases: list[FetchRef] | None = None,
) -> dict:
    """Run the fetch eval across every case.

    When ``download_to`` is None, downloads go to a temporary directory
    that is removed after the run — we only care about the acquisition
    outcome, not persistence. When set, PDFs are kept so humans can
    inspect what the validator accepted.
    """
    cases = cases if cases is not None else DEFAULT_CASES
    email = email or "eval-real-world-fetch@example.invalid"

    if download_to is not None:
        download_to.mkdir(parents=True, exist_ok=True)
        workspace_dir = download_to
        cleanup: tempfile.TemporaryDirectory | None = None
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="oa_fetch_eval_")
        workspace_dir = Path(cleanup.name)

    start = time.monotonic()
    outcomes: list[FetchOutcome] = []
    try:
        for ref in cases:
            outcome = await _run_one(ref, workspace_dir, email)
            _log_outcome(outcome)
            outcomes.append(outcome)
    finally:
        if cleanup is not None:
            cleanup.cleanup()
    elapsed = time.monotonic() - start

    report = _build_report(outcomes, elapsed_seconds=elapsed)
    _print_report(report)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / "oa_fetch_eval.json"
        out_file.write_text(json.dumps(report, indent=2))
        logger.info("Wrote {}", out_file)

    return report


def _log_outcome(outcome: FetchOutcome) -> None:
    status = "OK" if outcome.found else "MISS"
    detail = outcome.actual_source or outcome.reason or "no source"
    logger.info(
        "  [{}] {} (expected={}): {}",
        status,
        outcome.key,
        outcome.expected_source or "any",
        detail,
    )


def _build_report(outcomes: list[FetchOutcome], elapsed_seconds: float) -> dict:
    """Aggregate outcomes into a JSON-serialisable report."""
    per_source = Counter(o.actual_source for o in outcomes if o.found)
    miss_reasons = Counter(o.reason for o in outcomes if not o.found and o.reason)
    found = sum(1 for o in outcomes if o.found)
    return {
        "total": len(outcomes),
        "found": found,
        "missing": len(outcomes) - found,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "per_source": dict(per_source),
        "miss_reasons": dict(miss_reasons),
        "outcomes": [o.model_dump() for o in outcomes],
    }


def _print_report(report: dict) -> None:
    """Render the report to stdout for human review."""
    total = report["total"]
    found = report["found"]
    pct = 100.0 * found / total if total else 0.0
    print("\n" + "=" * 60)
    print(f"OA fetch eval — {found}/{total} found ({pct:.1f}%)")
    print(f"Wall time: {report['elapsed_seconds']}s")
    if report["per_source"]:
        print("\nPer source:")
        for source, count in sorted(
            report["per_source"].items(), key=lambda x: (-x[1], x[0])
        ):
            print(f"  {source:20s} {count}")
    if report["miss_reasons"]:
        print("\nMiss reasons:")
        for reason, count in sorted(
            report["miss_reasons"].items(), key=lambda x: (-x[1], x[0])
        ):
            print(f"  [{count}] {reason}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory to write oa_fetch_eval.json into (default: no file output).",
    )
    parser.add_argument(
        "--download",
        type=Path,
        default=None,
        help=(
            "If set, PDFs are kept in this directory. Without --download, "
            "downloads go to a temp dir that is removed after the run."
        ),
    )
    parser.add_argument(
        "--email",
        default="",
        help="Polite-contact email. Required by Unpaywall; otherwise optional.",
    )
    args = parser.parse_args(argv)
    asyncio.run(
        run_fetch_eval(
            output_dir=args.output,
            download_to=args.download,
            email=args.email,
        )
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
