"""Shared data models for SciLint Score evaluation.

Extracted into a leaf module (no intra-evals imports) to break the
circular dependency between scilint_score_eval.py and
scilint_score_cases.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


TAXONOMY_DIMS = ["type", "specificity", "testability", "support", "scope"]


class TaxonomyCase(BaseModel):
    """One claim with ground-truth taxonomy labels."""

    name: str
    claim_text: str
    key: str
    context: str  # surrounding text for the LLM
    expected: dict[str, str]  # dimension → expected label
    description: str = ""


class TaxonomyCaseResult(BaseModel):
    """Result of classifying one claim."""

    name: str
    correct: dict[str, bool] = Field(default_factory=dict)  # dim → match
    predicted: dict[str, str] = Field(default_factory=dict)
    expected: dict[str, str] = Field(default_factory=dict)
    failed: bool = False  # LLM returned no result


class TaxonomyMetrics(BaseModel):
    """Per-dimension accuracy across all cases."""

    per_dim: dict[str, dict[str, int]] = Field(default_factory=dict)
    # per_dim[dim] = {"correct": N, "total": N}
    cases_run: int = 0
    cases_failed: int = 0

    @property
    def overall_accuracy(self) -> float:
        correct = sum(d["correct"] for d in self.per_dim.values())
        total = sum(d["total"] for d in self.per_dim.values())
        return correct / total if total > 0 else 0.0

    def dim_accuracy(self, dim: str) -> float:
        d = self.per_dim.get(dim, {"correct": 0, "total": 0})
        return d["correct"] / d["total"] if d["total"] > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases_run": self.cases_run,
            "cases_failed": self.cases_failed,
            "overall_accuracy": round(self.overall_accuracy, 3),
            "per_dimension": {
                dim: {
                    "accuracy": round(self.dim_accuracy(dim), 3),
                    "correct": d["correct"],
                    "total": d["total"],
                }
                for dim, d in self.per_dim.items()
            },
        }


class LaudanCase(BaseModel):
    """One problem-solving eval case with expected score range."""

    name: str
    intro_text: str
    limitations_text: str
    expected_min: float  # score should be >= this
    expected_max: float  # score should be <= this
    description: str = ""


class LaudanCaseResult(BaseModel):
    """Result of one Laudan scoring."""

    name: str
    score: float
    expected_min: float
    expected_max: float
    in_range: bool
    reasoning: str = ""


class LaudanMetrics(BaseModel):
    """Aggregate metrics for Laudan problem-solving axis."""

    cases_run: int = 0
    in_range: int = 0
    mean_score: float = 0.0

    @property
    def range_accuracy(self) -> float:
        return self.in_range / self.cases_run if self.cases_run > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases_run": self.cases_run,
            "in_range": self.in_range,
            "range_accuracy": round(self.range_accuracy, 3),
            "mean_score": round(self.mean_score, 3),
        }


class SciLintScoreEvalResult(BaseModel):
    """Complete SciLint Score evaluation result."""

    taxonomy: TaxonomyMetrics = Field(default_factory=TaxonomyMetrics)
    taxonomy_cases: list[TaxonomyCaseResult] = Field(default_factory=list)
    laudan: LaudanMetrics = Field(default_factory=LaudanMetrics)
    laudan_cases: list[LaudanCaseResult] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.taxonomy.cases_run > 0:
            d["taxonomy"] = {
                "metrics": self.taxonomy.to_dict(),
                "cases": [
                    {
                        "name": c.name,
                        "correct": c.correct,
                        "predicted": c.predicted,
                        "expected": c.expected,
                        "failed": c.failed,
                    }
                    for c in self.taxonomy_cases
                ],
            }
        if self.laudan.cases_run > 0:
            d["laudan"] = {
                "metrics": self.laudan.to_dict(),
                "cases": [
                    {
                        "name": c.name,
                        "score": round(c.score, 4),
                        "expected_min": c.expected_min,
                        "expected_max": c.expected_max,
                        "in_range": c.in_range,
                        "reasoning": c.reasoning,
                    }
                    for c in self.laudan_cases
                ],
            }
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("SciLint Score eval results saved to {}", path)
