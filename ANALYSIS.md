# Code and Paper Analysis

## Executive assessment

The codebase is unusually disciplined for a paper-companion release. It is not a loose simulation script collection; it is a release-contract system designed to prevent ambiguous, partial, or overclaimed outputs.

The manuscript and the code are tightly aligned on one core boundary: the work characterizes **offered coordination workload under explicit assumptions** and deliberately avoids claiming implementation performance.

## Code analysis

### `simUp.py`

This file is the analytical core and publication-data publisher.

Strengths:

- Strong input validation through exact-schema configuration checks
- Clear modeling separation between analytical equations and stochastic verification
- Deterministic re-generation checks before publication
- Machine-readable claim discipline through `claim_matrix.csv`
- Immutable published runs with manifest and checksum binding

Research relevance:

- The code supports reproducibility arguments well
- It is suitable for peer inspection because interpretation boundaries are encoded, not left implicit

Technical caution:

- The script is large and monolithic, which raises maintenance cost
- The stochastic verification layer could be mistaken by casual readers as an empirical simulation unless the README is explicit

### `plots.py`

This file is more than a plotting script. It is a release validator for figure-generation inputs and outputs.

Strengths:

- Refuses non-committed or malformed analytical inputs
- Re-renders figures twice and compares fingerprints
- Validates output sets, metadata, and figure-to-claim links
- Separates primary figures from supplementary figures cleanly

Research relevance:

- This materially improves defensibility for a journal submission because figure provenance is explicit

Technical caution:

- The script depends on fonts and rendering environment details
- Public readers may need a short note explaining why exact rendering checks can fail across incomplete environments

### `run_experiment.ps1`

This is the repository’s operational backbone.

Strengths:

- Enforces interpreter version and 64-bit CPython expectations
- Verifies exact dependency closure rather than approximate installation
- Makes release production safer for readers reproducing on Windows
- Binds figure publication to the exact analytical run

Research relevance:

- This is consistent with a serious reproducibility posture rather than a best-effort artifact bundle

Technical caution:

- The runner is Windows-first, so Linux/macOS users will rely on direct Python entry points unless you later add a cross-platform wrapper

## Paper analysis

The manuscript’s main contribution is not just the workload model. Its stronger methodological contribution is the insistence on a claim boundary:

- supported analytical demand estimates
- prohibited implementation-performance interpretations
- prohibited security, erasure, and legal-compliance inferences

That boundary is credible because the code mirrors it.

## Alignment between code and paper

The alignment is strong in four places:

1. Scope language
   Both the paper and the scripts repeatedly state that the outputs are offered-workload estimates, not achieved throughput.

2. Headline quantities
   The paper reports the same categories the code generates: analytical workload, component decomposition, sensitivity, temporal envelopes, logical metadata volume, and stochastic verification.

3. Reproducibility posture
   The manuscript claims deterministic, fail-closed reproducibility, and the code actually implements manifests, locks, checksums, release commits, and exact inventories.

4. Claim discipline
   The paper describes a machine-checkable claim matrix, and `simUp.py` contains explicit claim-matrix generation and validation logic.

## Duplicate PDF note

`Paper1V2_Transaction.pdf` and `Paper1V2_Transaction (1).pdf` are byte-identical duplicates. Only one needs to be referenced in the repository documentation.

## Suggested public-facing positioning

Present this repository as:

- the analytical workload model
- the reproducibility package
- the figure-generation pipeline

Do not present it as:

- the HTGL implementation
- a blockchain benchmark suite
- an empirical trace study

## Suggested next repository improvements

1. Add a short `LICENSE` file.
2. Replace placeholder GitHub and publication metadata in `CITATION.cff`.
3. Consider adding a minimal cross-platform `Makefile` or `justfile`.
4. Optionally add a small `sample_release/` pointer or Zenodo link after artifact archival.
