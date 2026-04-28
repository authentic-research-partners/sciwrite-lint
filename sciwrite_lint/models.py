"""Data models for sciwrite-lint."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Finding(BaseModel):
    """A single issue found by a check."""

    level: Literal["error", "warning", "info"]
    rule_id: str  # check ID, e.g., "dangling-cite", "reference-exists"
    message: str
    file: str = ""
    line: int | None = None
    context: str = ""


class CheckMeta(BaseModel):
    """Metadata for a registered check."""

    id: str  # "dangling-cite", "cross-section-consistency", etc.
    severity: Literal["error", "warning", "info"]
    category: Literal["manuscript", "reference-db", "local-llm"]
    description: str


# Legacy alias for backwards compatibility
RuleMeta = CheckMeta


class CheckResult(BaseModel):
    """Result from running checks on a paper."""

    checker: str  # e.g., "check", "verify"
    paper: str  # e.g., "my_paper" or filename
    findings: list[Finding] = Field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors


class Citation(BaseModel):
    """A citation extracted from a .tex file."""

    key: str
    raw_text: str
    authors: list[str] = Field(default_factory=list)
    title: str = ""
    year: str = ""
    venue: str = ""
    doi: str = ""
    url: str = ""
    arxiv_id: str = ""  # e.g. "2310.01798"
    pmid: str = ""  # e.g. "12345678"
    pmc_id: str = ""  # e.g. "PMC1234567"
    isbn: str = ""  # e.g. "9780226458113"
    lccn: str = ""  # e.g. "2019012345"
    source_paper: str = ""
    bib_format: str = ""  # "simple", "natbib", or "bib"
    entry_type: str = ""  # BibTeX entry type: "article", "misc", etc.

    # Local source
    local_status: Literal["pdf", "md", "none"] = "none"
    local_path: str = ""

    # API verification
    api_match: Literal[
        "",
        "verified",
        "mismatch",
        "not_found",
        "skipped",
        "web_verified",
        "web_dead",
        "web_blocked",
        "manual",
    ] = ""
    api_source: str = ""
    api_data: dict = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)

    # Tier (computed from metadata)
    tier: Literal["T1", "T2", "T3", ""] = ""


class CitationMetadata(BaseModel):
    """Persistent per-citation verification record stored as JSON."""

    key: str
    verified_date: str = ""
    api_source: str = ""  # "crossref", "openalex", "semantic_scholar"
    api_match: Literal[
        "",
        "verified",
        "mismatch",
        "not_found",
        "skipped",
        "web_verified",
        "web_dead",
        "web_blocked",
        "manual",
    ] = ""

    # Canonical data from API (what the world says)
    canonical: dict[str, Any] = Field(default_factory=dict)

    # What our .tex bibitem says
    bibitem: dict[str, Any] = Field(default_factory=dict)

    # Access status
    access: dict[str, Any] = Field(default_factory=dict)

    mismatches: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)

    # Manual override
    manual_override: dict[str, Any] = Field(default_factory=dict)


class SectionInfo(BaseModel):
    """A section in a LaTeX document."""

    label: str  # e.g., "sec:intro" (empty if no label)
    title: str
    depth: int  # 0=section, 1=subsection, 2=subsubsection
    start_line: int
    end_line: int
    word_count: int
    cite_count: int = 0
