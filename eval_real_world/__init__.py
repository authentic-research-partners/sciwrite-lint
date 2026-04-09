"""Real-world evaluation of sciwrite-lint on external papers.

Downloads papers from arXiv (LaTeX source) and bioRxiv (PDF) across a broad
set of disciplines, runs all linter rules, and measures accuracy via
Sonnet-adjudicated TP/FP classification and synthetic error injection.

Sources:
    arXiv  — LaTeX source (cs, physics, math, econ, q-bio, stats)
    bioRxiv — PDF only (biology preprints, exercises GROBID path)

Evaluation modes:
    fpr    — run linter on clean papers, Sonnet judges each finding as TP/FP
    inject — inject known errors into LaTeX papers, measure detection rate

Reproducibility:
    Corpus metadata (paper IDs, titles, authors, categories) is saved to
    evals/results/real_world_corpus.json for version control. The actual
    downloaded papers live in real_world_corpus/ (gitignored).

Requires: httpx (core dep), claude CLI (for Sonnet judge).
"""
