# Reproducibility Notes

## Intended execution path

The safest release path is:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_experiment.ps1
```

That runner is stricter than invoking the Python scripts directly. It validates:

- Supported 64-bit CPython version
- Exact package closure from `requirements.txt`
- Script self-tests
- Committed analytical and figure release integrity

## What the scripts guarantee

### `simUp.py`

- Strict configuration validation
- Deterministic regeneration of release frames
- Immutable committed output directories
- Cryptographic file inventories and manifests
- Generated claim matrices and verification checks

### `plots.py`

- Input must be a committed analytical release
- Figure renders are generated twice and fingerprint-compared
- Figure metadata and input references are validated
- Output inventories and release manifests are published

## What I verified during analysis

- The two PDF files are byte-identical duplicates
- `simUp.py`, `plots.py`, and `run_experiment.ps1` are internally consistent in scope and contract language
- The paper’s stated scope boundary matches the code comments and generated artifact intent

## What I could not fully execute in this workspace

I did not run the full script self-tests in this workspace because the available bundled Python runtime did not include all pinned runtime packages:

- `simUp.py` requires `scipy`
- `plots.py` requires `matplotlib`

This is an environment-availability issue, not a code-logic inconsistency.

## Recommended public release checklist

1. Create a clean virtual environment with CPython 3.12 or 3.13.
2. Install the exact lock from `requirements.txt`.
3. Run `run_experiment.ps1`.
4. Confirm that both analytical and figure releases are published successfully.
5. Upload only source files and documentation to GitHub.
6. Publish generated releases separately if you want readers to download exact artifacts.
