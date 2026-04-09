"""Shared data models and LaTeX helpers for synthetic evaluation.

These are extracted into a leaf module (no intra-evals imports) to
break the circular dependency between synthetic.py and
synthetic_generators.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ExpectedFinding(BaseModel):
    """One expected finding in a synthetic case."""

    rule_id: str
    context: str = ""  # substring to match in finding context/message


class SyntheticCase(BaseModel):
    """One synthetic test case with ground truth."""

    name: str
    check_id: str  # which check this tests
    tex_content: str = ""  # LaTeX content (for LaTeX-mode cases)
    expected: list[ExpectedFinding] = Field(default_factory=list)
    # For database checks: fake metadata; for claim-support: _claim_results key
    metadata: dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    # For PDF-mode cases: a GrobidResult (or dict) to build ManuscriptContext from
    grobid_result: Any = None
    # For figure checks: pre-computed VL descriptions injected into the system prompt.
    # When set, the eval runner creates a temp workspace with these descriptions
    # in the vision_cache table so _load_figure_descriptions picks them up.
    figure_descriptions: str = ""


class CaseResult(BaseModel):
    """Result of running one synthetic case."""

    name: str
    check_id: str
    tp: int = 0  # expected findings that were detected
    fp: int = 0  # unexpected findings from this check
    fn: int = 0  # expected findings that were missed
    findings: list[dict[str, Any]] = Field(default_factory=list)
    expected: list[dict[str, Any]] = Field(default_factory=list)
    matched: list[bool] = Field(default_factory=list)
    elapsed_s: float = 0.0  # wall-clock time for this case


class CheckMetrics(BaseModel):
    """Aggregate metrics for one check across all cases."""

    check_id: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    cases_run: int = 0

    @property
    def precision(self) -> float:
        total = self.tp + self.fp
        return self.tp / total if total > 0 else 0.0

    @property
    def recall(self) -> float:
        total = self.tp + self.fn
        return self.tp / total if total > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "cases_run": self.cases_run,
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "f1": round(self.f1, 3),
        }


class EvalResult(BaseModel):
    """Complete synthetic evaluation result."""

    cases: list[CaseResult] = Field(default_factory=list)
    metrics: dict[str, CheckMetrics] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                check_id: m.to_dict() for check_id, m in sorted(self.metrics.items())
            },
            "cases": [c.model_dump() for c in self.cases],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("Synthetic eval results saved to {}", path)


# ---------------------------------------------------------------------------
# LaTeX document builders (used by gen_* modules)
# ---------------------------------------------------------------------------

_DOCUMENT_TEMPLATE = r"""\documentclass{{article}}
\begin{{document}}

{body}

\begin{{thebibliography}}{{99}}
{bibliography}
\end{{thebibliography}}
\end{{document}}
"""

_SECTIONS_TEMPLATE = r"""\documentclass{{article}}
\begin{{document}}

\begin{{abstract}}
{abstract}
\end{{abstract}}

\section{{Introduction}}
{intro}

\section{{Methods}}
{methods}

\section{{Results}}
{results}

\section{{Conclusion}}
{conclusion}

\begin{{thebibliography}}{{99}}
{bibliography}
\end{{thebibliography}}
\end{{document}}
"""


def _make_doc(body: str, bibitems: list[str]) -> str:
    """Build a simple LaTeX document."""
    return _DOCUMENT_TEMPLATE.format(
        body=body,
        bibliography="\n".join(bibitems),
    )


def _make_sectioned_doc(
    abstract: str,
    intro: str,
    methods: str,
    results: str,
    conclusion: str,
    bibitems: list[str],
) -> str:
    """Build a LaTeX document with standard sections."""
    return _SECTIONS_TEMPLATE.format(
        abstract=abstract,
        intro=intro,
        methods=methods,
        results=results,
        conclusion=conclusion,
        bibliography="\n".join(bibitems),
    )


_BIBITEMS = [
    r"\bibitem{smith2020} Smith, J. (2020). A study of deep learning. In ICML.",
    r"\bibitem{jones2021} Jones, A. (2021). Neural networks revisited. In NeurIPS.",
    r"\bibitem{wang2022} Wang, L. (2022). Transformers for NLP. In ACL.",
    r"\bibitem{chen2019} Chen, R. (2019). Attention mechanisms. In AAAI.",
]
