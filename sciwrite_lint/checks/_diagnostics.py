"""System-issue findings — operational gaps in a linter run.

Two orthogonal axes describe a Finding:

1. **Bucket** — *whose* problem is it?

   - **manuscript**  → something is wrong with the paper being checked.
     Lands in ``check_*.json:findings`` and counts toward ``errors`` /
     ``warnings`` / ``total_findings`` / ``compute_internal_score``.
   - **system**      → the linter itself could not run a check.
     Lands in ``check_*.json:system_issues`` and counts toward
     ``total_system_issues``. **Never** affects manuscript metrics.

   Routing is declarative: ``rule_id ∈ SYSTEM_RULE_IDS`` ⇒ system
   bucket. There is no ``is_system`` field on ``Finding``; the rule_id
   *is* the route. ``split_findings()`` (and its dict-form sibling)
   partition a mixed list using only this constant. Every output
   surface — JSON writer, terminal renderer, scorer, eval aggregator —
   uses the same split.

2. **Level** — *how severe* is it? Independent of the bucket.

   System issues today all emit at ``level="warning"`` (the linter
   couldn't fully run, but produced a partial / usable report). The
   convention for future system issues:

   - ``warning`` → a check or stage was skipped or partially ran;
     the rest of the pipeline continued and the report is meaningful.
     **Use this for all current diagnostics.**
   - ``error``   → reserved for cases where the linter cannot produce
     a meaningful report at all (currently unused).
   - ``info``    → reserved for advisory operational notes that don't
     reflect a coverage gap (currently unused).

   Mixing the two axes: a ``WARN`` line in the *manuscript* panel means
   "the paper has a warning-level issue"; a ``WARN`` line in the
   *system* panel means "the linter has a warning-level operational
   issue". Same level vocabulary, different semantics — the panel /
   JSON-field disambiguates.

The diagnostic rule_ids:

- ``llm-unavailable``  — ``llm_query`` exhausted its retry ladder
  (medium → low → off) and returned ``None``. One per skipped check
  (per ref, where applicable). Emitted by check ``_process_results``
  hooks when they see a ``None`` slot in their batched results.
- ``vision-incomplete`` — ``_stage_cited_vision`` finished with at
  least one cited paper missing figure descriptions. One summary
  finding per run, listing the affected refs. Emitted by the cited
  vision stage itself.
- ``parse-failed``      — ``_stage_parse`` finished with at least one
  cited PDF that could not be parsed (GROBID transport error after
  retries, GROBID rejected the PDF, or a transient pipeline failure
  caught at the per-key level). One summary finding per run, listing
  the affected ref keys + their failure reasons. Subsequent stages
  (claim verification) run with reduced retrieval coverage for the
  affected refs.
- ``internal-error``   — a check raised an unexpected exception
  during ``build_queries`` or ``process_results`` / a text check's
  body raised mid-run. One per failing check. Emitted by the LLM
  batch runner and the text-check runner.
- ``source-unsupported`` — one or more checks declare
  ``supports_markdown=False`` (they parse LaTeX citation/cross-reference/
  figure syntax) and the manuscript is markdown. The text-check runner
  skips those checks and emits one summary finding listing them, so the
  coverage gap is explicit rather than a silent zero-result.
- ``no-references`` — citation extraction found zero references in the
  manuscript, so the citation / reference / claim stages had nothing to
  verify. Emitted once per run by the pipeline setup. A genuinely
  citation-free note is harmless, but a zero usually means the
  bibliography was not detected — surfacing it stops a clean-looking
  report from hiding that the reference checks never ran.

Adding a new system rule_id:

1. Add it to :data:`SYSTEM_RULE_IDS`.
2. Provide a builder function in this module so emission sites don't
   reinvent the message format. Always emit at the bucket-appropriate
   level (``warning`` per the convention above).
3. Add a one-line entry to the list above and a CHANGELOG note.
4. Cover the new bucket-routing case in ``test_diagnostics.py``.

System rule_ids are intentionally *not* registered in the check
registry — ``sciwrite-lint checks`` lists what the tool examines in
the manuscript, not its own operational state.
"""

from __future__ import annotations

from typing import Any

from sciwrite_lint.models import Finding


# Rule IDs that mark operational/system issues (linter could not run a
# check) rather than manuscript problems. Consumers MUST exclude these
# from manuscript-quality counts, scores, and per-rule eval metrics.
# Adding a rule_id here is the only step that re-routes its findings;
# every output surface partitions through :func:`split_findings`.
SYSTEM_RULE_IDS: frozenset[str] = frozenset(
    {
        "llm-unavailable",
        "vision-incomplete",
        "internal-error",
        "parse-failed",
        "source-unsupported",
        "no-references",
    }
)


def split_findings(
    findings: list[Finding],
) -> tuple[list[Finding], list[Finding]]:
    """Partition ``findings`` into ``(manuscript, system)``.

    System issues report on the linter's own state — they must not
    contribute to manuscript scores, ``total_findings`` counts, or
    per-rule eval aggregation. Use this helper at every output surface
    to apply the same split; routing by ``rule_id`` against
    :data:`SYSTEM_RULE_IDS` keeps a single source of truth.

    The relative ordering inside each bucket is preserved (stable
    partition) so downstream renderers see findings in the same order
    they were emitted.
    """
    manuscript: list[Finding] = []
    system: list[Finding] = []
    for f in findings:
        (system if f.rule_id in SYSTEM_RULE_IDS else manuscript).append(f)
    return manuscript, system


def split_findings_dicts(
    findings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Dict-form sibling of :func:`split_findings`.

    Eval pipelines and report writers often pass already-serialized
    findings (``Finding.model_dump()``) around as plain dicts. This
    helper applies the same partition rule on the dict form so callers
    don't have to round-trip through ``Finding`` objects just to split.
    """
    manuscript: list[dict[str, Any]] = []
    system: list[dict[str, Any]] = []
    for f in findings:
        (system if f.get("rule_id") in SYSTEM_RULE_IDS else manuscript).append(f)
    return manuscript, system


def llm_unavailable_finding(
    check_id: str,
    *,
    file: str = "",
    ref_key: str | None = None,
) -> Finding:
    """Build a finding for a check skipped because the LLM gave up.

    Emitted from check ``_process_results`` hooks when their batched
    LLM result slot is ``None`` (retry ladder exhausted in
    ``llm_utils.llm_query``).

    Parameters
    ----------
    check_id:
        The check that failed to run (e.g. ``"axis-label-consistency"``).
        Embedded in the message so the reviewer can identify the gap.
    file:
        Manuscript or reference filename, when available.
    ref_key:
        Cited-paper key when the give-up happened on a ref-internal check;
        ``None`` for manuscript-level checks.
    """
    prefix = f"[{ref_key}] " if ref_key else ""
    return Finding(
        level="warning",
        rule_id="llm-unavailable",
        message=(f"{prefix}{check_id}: LLM call exhausted retries; check did not run"),
        file=file,
    )


def vision_incomplete_finding(missing_refs: list[str]) -> Finding:
    """Build a finding for cited papers left without figure descriptions.

    Emitted by ``_stage_cited_vision`` once per run (a single summary
    finding listing every affected ref) when the vision subprocess
    finishes — whether by completion, timeout, or crash — with at
    least one ref missing its figure descriptions.
    """
    refs = ", ".join(sorted(missing_refs))
    return Finding(
        level="warning",
        rule_id="vision-incomplete",
        message=(
            f"Cited paper figures missing for {len(missing_refs)} ref(s): "
            f"{refs}. Ref-internal checks ran with reduced visual context."
        ),
    )


def parse_failed_finding(failed_refs: list[str]) -> Finding:
    """Build a finding for cited PDFs that could not be parsed.

    Emitted by ``_stage_parse`` once per run when ``parse_all_missing``
    leaves at least one ref in an ``"error"`` or ``"failed"`` status.
    Per-key failure reasons are written to the loguru log by
    ``parse_all_missing._parse_one`` (``Failed to parse {key}: {exc}``);
    this finding is a summary listing the affected refs so the
    operator notices that retrieval coverage will be reduced for the
    subsequent claim-verification stage. The system bucket — not a
    manuscript problem.
    """
    keys = sorted(failed_refs)
    summary = ", ".join(keys)
    return Finding(
        level="warning",
        rule_id="parse-failed",
        message=(
            f"Could not parse {len(keys)} cited PDF(s): {summary}. "
            "Subsequent claim verification ran with reduced retrieval "
            "coverage for these refs (see log for per-key error details)."
        ),
    )


def internal_error_finding(check_id: str, error: BaseException) -> Finding:
    """Build a finding for a check that crashed with an unexpected exception.

    Emitted by the text-check runner and the LLM batch runner when a
    check's ``build_queries`` / ``process_results`` body — or the
    check function itself — raises something other than the documented
    failure modes the pipeline already handles (LLM timeouts, model
    parse errors). The exception type and message are included as
    context so an operator can diagnose without re-running.

    Parameters
    ----------
    check_id:
        The check that crashed (the registered rule_id).
    error:
        The caught exception; its type name and ``str()`` are
        truncated to 200 chars and stored in the finding's ``context``.
    """
    return Finding(
        level="warning",
        rule_id="internal-error",
        message=f"{check_id}: check raised an unexpected exception; check did not run",
        context=f"{type(error).__name__}: {error!s}"[:200],
    )


def source_unsupported_finding(
    source_type: str, skipped_check_ids: list[str]
) -> Finding:
    """Build a finding for checks skipped because the source type lacks the structure they need.

    Emitted once per run by the text-check runner when the manuscript is
    markdown and one or more registered checks declare
    ``supports_markdown=False`` (they parse LaTeX citation/cross-reference/
    figure syntax or need a populated bibliography). Listing the skipped
    checks makes the coverage gap explicit — a markdown manuscript still
    gets prose and structure checks, but these did not run.

    Parameters
    ----------
    source_type:
        The manuscript source type (e.g. ``"markdown"``).
    skipped_check_ids:
        Registered check IDs that were skipped for this source.
    """
    checks = ", ".join(sorted(skipped_check_ids))
    return Finding(
        level="warning",
        rule_id="source-unsupported",
        message=(
            f"{len(skipped_check_ids)} check(s) need citation/reference structure "
            f"not extracted from {source_type} manuscripts and did not run: {checks}. "
            "Prose and structure checks ran normally."
        ),
    )


def no_references_finding() -> Finding:
    """Build a finding for a run that extracted zero references.

    Emitted once by the pipeline setup when citation extraction yields an
    empty list. The reference, citation, and claim stages then have
    nothing to verify — surfacing this keeps a clean-looking report from
    hiding that those checks never ran (usually a bibliography that was
    not detected, not a genuinely citation-free manuscript).
    """
    return Finding(
        level="warning",
        rule_id="no-references",
        message=(
            "No references found in the manuscript — citation, reference, and "
            "claim checks had nothing to verify. If the manuscript cites sources, "
            "check that its bibliography is detected (LaTeX \\bibliography{} / "
            ".bib, or for markdown the YAML bibliography: field or a sibling "
            "{name}.bib)."
        ),
    )
