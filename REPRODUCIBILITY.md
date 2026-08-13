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
