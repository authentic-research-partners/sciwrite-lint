# Changelog

## [0.2.1] — 2026-04-10

### Added
- **Higher-quality vision option.** Choose between the default Qwen3-VL-2B model and a larger Qwen3-VL-8B (FP8, via vLLM) for figure analysis. Select via `--vision-backend` or the `[vision]` section of `.sciwrite-lint.toml`.
- **Automatic container lifecycle.** The pipeline starts, swaps, and stops text and vision vLLM containers as needed, skipping swaps when all figures are already cached.
- **GPU embeddings on native Linux** *(preliminary)*: embeddings run on GPU automatically when available.

### Changed
- **Faster end-to-end runs.** Vision processing now runs concurrently, claim-verification query embeddings are pre-computed, and redundant container swaps are avoided.
- **Consistency checks return a curated shortlist.** `full-paper-consistency` reports up to 5 findings per check and `cross-section-consistency` up to 4, ranked by the model. Papers with few issues are unchanged; papers with many issues now get the most important ones instead of an unbounded list.
- **Figure readability issues are itemized.** Each problem with a figure is reported as its own bullet instead of one combined note, so downstream consistency checks see one issue at a time.

### Reliability
- **Long-paper consistency analysis is now robust under heavy finding load.** Results from consistency and full-paper checks are always well-formed, even when the model identifies many issues in a single pass.

## [0.2.0] — 2026-04-09

Initial public release (alpha), accompanying [arXiv:2603.17893](https://arxiv.org/abs/2603.17893).

See [README.md](README.md) for features and usage.
