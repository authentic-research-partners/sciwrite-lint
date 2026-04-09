"""Metrics computation and reporting for real-world evaluation.

Computes precision, recall, F1, false-positive rate per rule and category.
"""

from __future__ import annotations

import json

from loguru import logger
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Any


class RuleMetrics(BaseModel):
    """Accuracy metrics for one rule."""

    rule_id: str
    tp: int = 0
    fp: int = 0
    fn: int = 0  # missed injections
    uncertain: int = 0

    @property
    def precision(self) -> float | None:
        total = self.tp + self.fp
        return self.tp / total if total > 0 else None

    @property
    def recall(self) -> float | None:
        total = self.tp + self.fn
        return self.tp / total if total > 0 else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None:
            return None
        return 2 * p * r / (p + r) if (p + r) > 0 else None

    def to_dict(self) -> dict[str, Any]:
        d = self.model_dump()
        d["precision"] = (
            round(self.precision, 3) if self.precision is not None else None
        )
        d["recall"] = round(self.recall, 3) if self.recall is not None else None
        d["f1"] = round(self.f1, 3) if self.f1 is not None else None
        return d


class PaperResult(BaseModel):
    """Results for one paper."""

    paper_id: str
    total_findings: int = 0
    tp: int = 0
    fp: int = 0
    uncertain: int = 0
    injections_total: int = 0
    injections_detected: int = 0
    rule_details: dict[str, dict[str, Any]] = Field(default_factory=dict)


class EvalReport(BaseModel):
    """Complete evaluation report."""

    corpus_size: int = 0
    papers: list[PaperResult] = Field(default_factory=list)
    rule_metrics: dict[str, RuleMetrics] = Field(default_factory=dict)
    category_metrics: dict[str, RuleMetrics] = Field(default_factory=dict)

    def add_fpr_verdicts(
        self,
        paper_id: str,
        verdicts: list[dict[str, Any]],
    ) -> None:
        """Add Sonnet verdicts from FPR evaluation."""
        pr = PaperResult(paper_id=paper_id)
        pr.total_findings = len(verdicts)

        for v in verdicts:
            rule_id = v.get("rule_id", "unknown")
            judgment = v.get("judgment", "UNCERTAIN")

            if rule_id not in self.rule_metrics:
                self.rule_metrics[rule_id] = RuleMetrics(rule_id=rule_id)
            rm = self.rule_metrics[rule_id]

            if judgment == "TP":
                rm.tp += 1
                pr.tp += 1
            elif judgment == "FP":
                rm.fp += 1
                pr.fp += 1
            else:
                rm.uncertain += 1
                pr.uncertain += 1

        self.papers.append(pr)

    def add_injection_results(
        self,
        paper_id: str,
        injections: list[dict[str, Any]],
        detected: list[bool],
    ) -> None:
        """Add per-injection detection results.

        Args:
            paper_id: arXiv ID or paper identifier.
            injections: List of injection dicts (must have "rule_id").
            detected: Parallel list of booleans — True if injection was found.
        """
        pr = PaperResult(paper_id=paper_id)
        pr.injections_total = len(injections)

        for inj, was_detected in zip(injections, detected):
            rule_id = inj.get("rule_id", "unknown")
            if rule_id not in self.rule_metrics:
                self.rule_metrics[rule_id] = RuleMetrics(rule_id=rule_id)
            rm = self.rule_metrics[rule_id]

            if was_detected:
                rm.tp += 1
                pr.injections_detected += 1
            else:
                rm.fn += 1

        self.papers.append(pr)

    def compute_category_metrics(self) -> None:
        """Aggregate rule metrics into category-level metrics."""
        cats: dict[str, RuleMetrics] = {}
        for rule_id, rm in self.rule_metrics.items():
            # Category is the first segment: "dangling-ref" → "dangling"
            cat = rule_id.split("-")[0] if "-" in rule_id else rule_id
            if cat not in cats:
                cats[cat] = RuleMetrics(rule_id=cat)
            cats[cat].tp += rm.tp
            cats[cat].fp += rm.fp
            cats[cat].fn += rm.fn
            cats[cat].uncertain += rm.uncertain
        self.category_metrics = cats

    def to_dict(self) -> dict[str, Any]:
        self.compute_category_metrics()
        return {
            "corpus_size": self.corpus_size,
            "summary": {
                "total_papers": len(self.papers),
                "total_tp": sum(p.tp for p in self.papers),
                "total_fp": sum(p.fp for p in self.papers),
                "total_uncertain": sum(p.uncertain for p in self.papers),
                "mean_findings_per_paper": (
                    sum(p.total_findings for p in self.papers) / len(self.papers)
                    if self.papers
                    else 0
                ),
            },
            "per_rule": {
                rid: rm.to_dict() for rid, rm in sorted(self.rule_metrics.items())
            },
            "per_category": {
                cat: cm.to_dict() for cat, cm in sorted(self.category_metrics.items())
            },
            "per_paper": [p.model_dump() for p in self.papers],
        }

    def save(self, path: Path) -> None:
        """Save report to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info(f"Report saved to {path}")


def print_summary(report: EvalReport) -> None:
    """Print human-readable summary."""
    data = report.to_dict()
    s = data["summary"]

    print("\n" + "=" * 60)
    print("REAL-WORLD EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nCorpus: {data['corpus_size']} papers")

    # FPR stats (only if Sonnet judging was done)
    total_judged = s["total_tp"] + s["total_fp"] + s["total_uncertain"]
    if total_judged:
        print(f"Sonnet-judged findings: {total_judged}")
        print(f"  True positives:  {s['total_tp']}")
        print(f"  False positives: {s['total_fp']}")
        print(f"  Uncertain:       {s['total_uncertain']}")
        tp_fp = s["total_tp"] + s["total_fp"]
        if tp_fp:
            print(f"  FPR (excl. uncertain): {s['total_fp'] / tp_fp:.1%}")

    # Injection stats (only if injections were done)
    total_injected = sum(p.get("injections_total", 0) for p in data["per_paper"])
    total_detected = sum(p.get("injections_detected", 0) for p in data["per_paper"])
    if total_injected:
        print(
            f"Injections: {total_detected}/{total_injected} detected ({total_detected / total_injected:.0%})"
        )

    def _fmt(val: float | None) -> str:
        return f"{val:>6.1%}" if val is not None else "   N/A"

    if data["per_category"]:
        print(
            f"\n{'Category':<30} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Rec':>7} {'F1':>7}"
        )
        print("-" * 74)
        for cat, m in data["per_category"].items():
            print(
                f"{cat:<30} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5}"
                f" {_fmt(m['precision'])} {_fmt(m['recall'])} {_fmt(m['f1'])}"
            )

    if data["per_rule"]:
        print(
            f"\n{'Rule':<30} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Rec':>7} {'F1':>7}"
        )
        print("-" * 74)
        for rid, m in data["per_rule"].items():
            print(
                f"{rid:<30} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5}"
                f" {_fmt(m['precision'])} {_fmt(m['recall'])} {_fmt(m['f1'])}"
            )

    print()
