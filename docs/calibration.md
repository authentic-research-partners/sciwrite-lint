# Calibration

Calibration iteratively adjusts SciLint Score weights, prompts, and taxonomy until the scoring system correctly ranks 20 papers with known ground truth.

## How it works

20 papers span the quality spectrum: Nobel Prize work (Graphene, LIGO), landmark methods (Transformer, AlphaFold, CRISPR), retracted fraud (Macchiarini, LaCour, Baughman), fabricated references (Shoukat), failed replication (LK-99), and more.

~30 ordinal constraints define expected rankings (e.g., `LIGO > LK-99`, `Graphene > Shoukat`). The eval scores all papers and reports which constraints pass or fail.

## Iteration cycle

```
1. Diagnose: which axes/constraints are failing and why?
2. Adjust weights, prompts, or taxonomy in sciwrite_lint/

3. python -m evals eval-calibration --rerun   # score all 20 papers (~30 min)
4. Check constraint pass rate — did it improve?
5. Repeat from 1
```

**Fix the system, not the scores.** Every change should improve how the scoring system reasons about papers *in general*. Never add logic targeting specific calibration papers. A good fix is a general principle ("theoretical proofs count as severe tests") that happens to fix multiple constraints. A bad fix is a special case that games one paper's score.

Prompt changes to the claim taxonomy and problem-solving scoring encode philosophy-of-science reasoning — treat them as academic writing, not mechanical edits. Think through how wording changes affect the full range of paper types (empirical, theoretical, clinical, survey, replication).

Constraint failures show exact values so you can see where the ranking breaks:

```
FAILURES (2):
  x Transformer > BERT-finetune: 0.262 vs 0.310
  x LIGO ≈ AlphaFold: Q1 vs Q2
```

## Commands

```bash
python -m evals eval-calibration                         # all 20 papers
python -m evals eval-calibration --papers Graphene,LK-99 # quick subset
python -m evals eval-calibration --rerun                 # force re-score
python -m evals eval-calibration --concurrency 5         # more vLLM parallelism
```

Requires GROBID + vLLM running. Missing PDFs are auto-downloaded on first run.

## Adding your own papers

1. Place the PDF in the calibration directory (see `CALIBRATION_DIR` in `evals/calibration.py`)
2. Add a short name → filename mapping in `NAME_TO_FILE` in `evals/calibration.py`
3. Add ordinal constraints in `MANIFEST.md` using the syntax: `NewPaper > KnownBadPaper` or `PaperA ≈ PaperB` (same quartile)
4. Constraints can target specific axes: `NewPaper test-severity > OtherPaper test-severity`
5. If the paper has experiments, add it to `EXPERIMENTAL_PAPERS`
6. Run `python -m evals eval-calibration --papers NewPaper` to score it, then `--rerun` for the full set

## Constraint syntax

Constraints are parsed from a fenced code block in `MANIFEST.md`:

```
Graphene > LK-99                    # overall SciLint Score
LIGO test-severity > Wu-survey test-severity   # specific axis
AlphaFold ≈ CRISPR                  # same quartile
Wu-survey test-severity < any experimental paper  # expands to all experimental papers
```

Operators: `>` (strictly greater), `<` (strictly less), `≈` (same quartile).
