#!/usr/bin/env python3
# -*- coding: utf-8 -*-
Default outputs
---------------
* Input:  ``htgl_analytical_data/``
* Output: ``htgl_analytical_figures/``
* Formats: vector PDF and high-resolution PNG
* Reproducibility supplement: verification and boundary diagnostics.

Examples
--------
    python plots.py
    python plots.py --data-dir ./results --figure-dir ./figures
    python plots.py --formats pdf png --dpi 600 --clean
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unicodedata
import uuid
import warnings
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


CODE_VERSION = "plotsV8-production-analytical-workload-characterization-2026-07-23"
CONTRACT_VERSION = "htgl-analytical-release-contract-v1"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "htgl_analytical_data"
DEFAULT_FIGURE_DIR = BASE_DIR / "htgl_analytical_figures"
DEFAULT_CACHE_DIR = BASE_DIR / ".matplotlib_cache"
REQUIRED_POPULATIONS = (50_000_000, 150_000_000, 450_000_000)
MINIMUM_PUBLICATION_DPI = 300
PRIMARY_FIGURE_NUMBERS = frozenset({2, 3, 8, 11, 12})

SCOPE_NOTE = (
    "Analytical offered-workload characterization only; not an HTGL "
    "implementation or platform-performance measurement."
)

# Set a writable, local cache and a non-interactive renderer before importing
# pyplot.  This keeps command-line and headless executions reproducible.
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_CACHE_DIR))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402


SCENARIO_COLORS = {
    "baseline": "#0072B2",
    "high_authorization": "#D55E00",
    "high_der_adoption_metadata_activity": "#009E73",
    "combined_high_activity": "#CC79A7",
}
COMPONENT_COLORS = {
    "consent_events_day": "#56B4E9",
    "authorization_events_day": "#E69F00",
    "der_metadata_events_day": "#009E73",
    "integrity_events_day": "#6B7280",
}
PROFILE_COLORS = {
    "uniform": "#6B7280",
    "diurnal_30pct": "#0072B2",
    "diurnal_60pct": "#D55E00",
}
STORAGE_COLORS = {
    "nominal": "#56B4E9",
    "conservative": "#D55E00",
}
POPULATION_MARKERS = {50.0: "o", 150.0: "s", 450.0: "^"}

COMPONENT_ORDER = (
    "consent_events_day",
    "authorization_events_day",
    "der_metadata_events_day",
    "integrity_events_day",
)
COMPONENT_LABELS = {
    "consent_events_day": "Consent-state changes",
    "authorization_events_day": "Authorization decisions",
    "der_metadata_events_day": "DER metadata changes",
    "integrity_events_day": "Integrity/audit anchors",
}
PARAMETER_ORDER = (
    "authorization_events_per_consumer_day",
    "der_adoption_fraction",
    "metadata_events_per_der_day",
    "consent_changes_per_consumer_year",
    "integrity_events_per_actor_day",
    "institutional_actors",
)
PARAMETER_LABELS = {
    "consumer_population": "Consumer population",
    "authorization_events_per_consumer_day": "Authorization rate",
    "der_adoption_fraction": "DER adoption",
    "metadata_events_per_der_day": "DER metadata rate",
    "consent_changes_per_consumer_year": "Consent-change rate",
    "integrity_events_per_actor_day": "Integrity-event rate",
    "institutional_actors": "Institutional actors",
}


@dataclass(frozen=True)
class FigureSpec:
    number: int
    basename: str
    title: str
    caption: str
    sources: tuple[str, ...]


FIGURE_SPECS = (
    FigureSpec(
        1,
        "fig_01_baseline_component_decomposition",
        "Baseline workload decomposition across populations",
        (
            "Expected daily HTGL coordination-event counts under the baseline "
            "assumptions. Bars are decomposed into consent-state changes, "
            "authorization decisions, DER metadata changes, and integrity/audit "
            "anchors. The populations denote consumers or endpoints, not "
            "consensus nodes."
        ),
        ("component_decomposition.csv",),
    ),
    FigureSpec(
        2,
        "fig_02_scenario_composition_50m",
        "Workload composition by scenario at 50 million consumers",
        (
            "Percentage composition of the expected coordination-event workload "
            "for the four analytical scenarios at 50 million consumers. The "
            "scenario inputs are explicit assumptions rather than measured "
            "deployment traces."
        ),
        ("component_decomposition.csv", "scenario_definitions.csv"),
    ),
    FigureSpec(
        3,
        "fig_03_population_scenario_scaling",
        "Analytical workload scaling",
        (
            "Expected average offered coordination TPS for 50 million, 150 "
            "million, and 450 million consumers under each analytical workload "
            "scenario. These are workload quantities, not achieved system "
            "throughput measurements."
        ),
        ("analytical_workload_results.csv",),
    ),
    FigureSpec(
        4,
        "fig_04_analytical_stochastic_alignment",
        "Closed-form and stochastic-model alignment",
        (
            "Comparison of the closed-form expected average TPS with the pooled "
            "stochastic mean for all population-scenario cases, with residuals "
            "shown in parts per million. Stochastic sampling verifies the "
            "daily-count equations; it is not an HTGL performance experiment."
        ),
        ("analytical_vs_stochastic.csv",),
    ),
    FigureSpec(
        5,
        "fig_05_baseline_daily_variability",
        "Modeled daily-count variability",
        (
            "Distribution of synthetic daily event-count deviations after division "
            "by 86,400 for the baseline case. These daily-count-equivalent rates "
            "are not instantaneous throughput requirements. Curves show the "
            "large-lambda Poisson-normal reference implied by the model."
        ),
        (
            "daily_stochastic_samples.csv",
            "analytical_workload_results.csv",
        ),
    ),
    FigureSpec(
        6,
        "fig_06_replication_stability",
        "Independent-replication stability",
        (
            "Distribution of replication-level mean errors across the configured "
            "independent replications for every workload scenario and population. "
            "Errors are expressed in parts per million relative to the "
            "closed-form mean."
        ),
        ("replication_summary.csv",),
    ),
    FigureSpec(
        7,
        "fig_07_convergence_diagnostics",
        "Stochastic mean convergence",
        (
            "Absolute stochastic-mean error as independent replications are "
            "accumulated, together with the model-implied 95% mean-error envelope. "
            "Both axes are logarithmic; this verifies numerical implementation."
        ),
        ("convergence_diagnostics.csv",),
    ),
    FigureSpec(
        8,
        "fig_08_oat_sensitivity",
        "One-at-a-time assumption sensitivity",
        (
            "Range of workload change obtained by varying one architectural "
            "assumption at a time over the predefined analytical bounds while "
            "holding all other baseline inputs constant."
        ),
        ("oat_sensitivity.csv",),
    ),
    FigureSpec(
        9,
        "fig_09_authorization_der_assumption_grid",
        "Authorization-rate and DER-adoption grid",
        (
            "Expected average coordination TPS over the bounded two-parameter "
            "grid of authorization-event rate and DER adoption using one shared "
            "color scale across populations. The white cross "
            "marks the baseline assumption (0.5 events/consumer/day and 10% DER "
            "adoption)."
        ),
        ("authorization_der_assumption_grid.csv",),
    ),
    FigureSpec(
        10,
        "fig_10_local_elasticities",
        "Analytical workload elasticities",
        (
            "Exact local elasticities of the linear workload equations. Each "
            "cell approximates the proportional workload change caused by a 1% "
            "change in the indicated input, holding other inputs constant."
        ),
        ("local_elasticities.csv",),
    ),
    FigureSpec(
        11,
        "fig_11_synthetic_temporal_profiles",
        "Synthetic within-day workload concentration",
        (
            "The left panel shows the three normalized synthetic within-day shapes; "
            "the right panel shows the high-concentration peak expected offered rate "
            "by scenario and population. These are analytical envelopes, not "
            "measured arrival traces or instantaneous p99 requirements."
        ),
        ("temporal_profile_detail.csv",),
    ),
    FigureSpec(
        12,
        "fig_12_logical_metadata_volume",
        "Annual logical coordination-metadata volume",
        (
            "Annual logical record volume under component-specific nominal and "
            "conservative payload-size assumptions. Values exclude block, index, "
            "replication, networking, and storage-engine overhead."
        ),
        ("logical_metadata_volume.csv",),
    ),
    FigureSpec(
        13,
        "fig_13_event_count_boundary",
        "Operational-to-coordination event-count boundary",
        (
            "Ratio of modeled operational meter-reading events to HTGL "
            "coordination events. This is an event-count comparison only and "
            "does not imply an equivalent byte, network-traffic, or transaction-"
            "throughput reduction."
        ),
        ("workload_boundary_event_counts.csv",),
    ),
    FigureSpec(
        14,
        "fig_14_reference_workload_envelopes",
        "Inverse analytical workload envelopes",
        (
            "Consumer population obtained by inverting each scenario workload "
            "equation for generic offered-load budgets. The budgets are "
            "hypothetical analytical reference values, not measured HTGL or "
            "ledger capacities."
        ),
        ("reference_workload_envelopes.csv",),
    ),
)

SPEC_BY_NUMBER = {spec.number: spec for spec in FIGURE_SPECS}


REQUIRED_COLUMNS: Mapping[str, set[str]] = {
    "analytical_workload_results.csv": {
        "population",
        "population_millions",
        "scenario",
        "coordination_events_day",
        "expected_tps",
        "daily_count_sd_equivalent_tps",
        "scope",
    },
    "component_decomposition.csv": {
        "population",
        "population_millions",
        "scenario",
        "component",
        "expected_events_day",
        "share_pct",
    },
    "daily_stochastic_samples.csv": {
        "population",
        "population_millions",
        "scenario",
        "replication",
        "day",
        "daily_count_equivalent_tps",
    },
    "analytical_vs_stochastic.csv": {
        "population",
        "population_millions",
        "scenario",
        "expected_tps",
        "daily_count_equivalent_tps_mean",
        "mean_relative_error_pct",
    },
    "replication_summary.csv": {
        "population",
        "population_millions",
        "scenario",
        "replication",
        "mean_relative_error_pct",
    },
    "convergence_diagnostics.csv": {
        "population",
        "population_millions",
        "scenario",
        "replications_included",
        "observations_included",
        "relative_error_pct",
    },
    "oat_sensitivity.csv": {
        "population",
        "population_millions",
        "varied_parameter",
        "workload_change_pct",
    },
    "local_elasticities.csv": {
        "population",
        "population_millions",
        "scenario",
        "parameter",
        "local_elasticity",
    },
    "authorization_der_assumption_grid.csv": {
        "population",
        "population_millions",
        "authorization_events_per_consumer_day",
        "der_adoption_fraction",
        "expected_tps",
    },
    "temporal_profile_detail.csv": {
        "population",
        "population_millions",
        "scenario",
        "temporal_profile",
        "profile_label",
        "hour_start",
        "expected_tps",
    },
    "logical_metadata_volume.csv": {
        "population",
        "population_millions",
        "scenario",
        "record_size_assumption",
        "horizon_days",
        "logical_tb_decimal",
    },
    "workload_boundary_event_counts.csv": {
        "population",
        "population_millions",
        "scenario",
        "operational_to_coordination_event_count_ratio",
    },
    "reference_workload_envelopes.csv": {
        "scenario",
        "reference_offered_load_budget_tps",
        "analytical_population_at_budget_millions",
    },
    "scenario_definitions.csv": {
        "name",
        "label",
        "description",
        "der_adoption_fraction",
        "evidence_class",
        "rationale",
    },
    "verification_checks.csv": {"check", "passed", "criterion"},
    "statistical_validation_metrics.csv": {
        "population",
        "scenario",
        "standardized_mean_error_z",
        "mean_z_critical",
        "mean_test_passed",
        "poisson_dispersion_index",
        "dispersion_acceptance_lower",
        "dispersion_acceptance_upper",
        "dispersion_test_passed",
    },
    "claim_matrix.csv": {
        "claim_id",
        "status",
        "publication_wording",
        "prohibited_interpretation",
        "evidence_files",
        "required_checks",
        "assumption_dependency",
    },
    "claim_matrix_validation.csv": {"validation", "passed"},
    "reproducibility_check.csv": {"check", "passed"},
}


@dataclass(frozen=True)
class InputRelease:
    root: Path
    directory: Path
    manifest: dict[str, Any]
    commit: dict[str, Any]
    config: dict[str, Any]
    inventory: dict[str, dict[str, Any]]
    manifest_sha256: str
    commit_sha256: str

    def assert_unchanged(self) -> None:
        """Revalidate the immutable input immediately before publication."""

        manifest_path = self.directory / "release_manifest.json"
        commit_path = self.directory / "RELEASE_COMMIT.json"
        if sha256(manifest_path) != self.manifest_sha256:
            raise RuntimeError("input release manifest changed during rendering")
        if sha256(commit_path) != self.commit_sha256:
            raise RuntimeError("input release commit changed during rendering")
        for relative, entry in self.inventory.items():
            path = validated_child_path(self.directory, relative)
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"input release file disappeared: {relative}")
            if path.stat().st_size != int(entry["size_bytes"]):
                raise RuntimeError(f"input release file size changed: {relative}")
            if sha256(path) != str(entry["sha256"]):
                raise RuntimeError(f"input release file checksum changed: {relative}")


@dataclass
class PlotContext:
    release: InputRelease
    data_root: Path
    data_dir: Path
    figure_dir: Path
    formats: tuple[str, ...]
    dpi: int
    frames: dict[str, pd.DataFrame]
    scenario_order: tuple[str, ...]
    scenario_labels: dict[str, str]
    populations: tuple[int, ...]
    input_manifest: dict[str, Any]
    input_commit: dict[str, Any]
    input_config: dict[str, Any]
    active_specs: tuple[FigureSpec, ...]
    created_paths: list[Path] = field(default_factory=list)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot analytical HTGL coordination-workload results."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory created by simUp.py (default: %(default)s).",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Destination for figures and caption metadata (default: %(default)s).",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "png"),
        default=("pdf", "png"),
        help="Output formats (default: pdf png).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=450,
        help="PNG resolution in dots per inch (default: %(default)s).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="After a successful immutable publication, remove legacy flat figure files.",
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Publish only the five primary-paper figures; omit supplement figures.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run release-contract and Unicode validation tests and exit.",
    )
    args = parser.parse_args(argv)
    if args.dpi < MINIMUM_PUBLICATION_DPI:
        parser.error(
            f"--dpi must be at least {MINIMUM_PUBLICATION_DPI}; lower values "
            "cannot satisfy the publication pixel-dimension gate"
        )
    args.formats = tuple(dict.fromkeys(args.formats))
    return args


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 9.0,
            "axes.titleweight": "semibold",
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.5,
            "grid.color": "#D1D5DB",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.75,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def forbidden_encoding_fragments() -> tuple[str, ...]:
    return (
        "\ufffd",
        "\u00c3",
        "\u00c2",
        "\u00e2\u20ac",
        "\u00e2\u20ac\u201d",
        "\u00e2\u20ac\u201c",
        "\u00e2\u20ac\u00a6",
    )


def validate_unicode_bytes(data: bytes, label: str) -> None:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"{label} is not valid UTF-8: {exc}") from exc
    if unicodedata.normalize("NFC", text) != text:
        raise RuntimeError(f"{label} is not NFC-normalized Unicode")
    bad = [value for value in forbidden_encoding_fragments() if value in text]
    if bad:
        raise RuntimeError(f"{label} contains encoding-corruption markers: {bad}")


def load_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required JSON object is absent or unsafe: {path}")
    try:
        payload = path.read_bytes()
        validate_unicode_bytes(payload, path.name)
        value = json.loads(payload.decode("utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise RuntimeError(f"cannot load required JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain one JSON object")
    return value


def validated_child_path(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if (
        not relative
        or relative_path.is_absolute()
        or bool(relative_path.drive)
        or any(part in {"", ".", ".."} for part in relative_path.parts)
    ):
        raise RuntimeError(f"release path must be non-empty and relative: {relative!r}")
    root_resolved = root.resolve()
    cursor = root_resolved
    for part in relative_path.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeError(f"release path contains a symbolic link: {relative!r}")
    candidate = (root_resolved / relative_path).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise RuntimeError(f"release path escapes its root: {relative!r}") from exc
    return candidate


def parse_boolean_series(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
    normalized = frame[column].astype(str).str.strip().str.lower()
    invalid = sorted(set(normalized) - {"true", "false"})
    if invalid:
        raise RuntimeError(f"{label} contains invalid booleans: {invalid}")
    return normalized.eq("true")


def text_columns(frame: pd.DataFrame) -> list[Any]:
    """Identify object-backed and extension-backed text without deprecated APIs."""

    return [
        column
        for column, dtype in frame.dtypes.items()
        if pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
    ]


class ReleaseContractReader:
    """Load exactly one committed simUp release and reject every partial state."""

    POINTER_FIELDS = {
        "contract_version",
        "run_id",
        "relative_run_directory",
        "release_commit_sha256",
        "configuration_sha256",
    }
    INVENTORY_FIELDS = {"path", "sha256", "size_bytes", "row_count", "columns"}

    def __init__(self, data_argument: Path) -> None:
        self.root = data_argument.expanduser().resolve()

    def read(self) -> tuple[InputRelease, dict[str, pd.DataFrame]]:
        release = self._open_release()
        frames = self._load_frames(release)
        self._validate_scientific_contract(release, frames)
        return release, frames

    def _open_release(self) -> InputRelease:
        if self.root.is_symlink() or not self.root.is_dir():
            raise RuntimeError(
                f"analytical data root is absent or unsafe: {self.root}. "
                "Run simUp.py or run_experiment.ps1 first."
            )

        pointer_path = self.root / "latest.json"
        pointer: dict[str, Any] | None
        if pointer_path.is_file() and not pointer_path.is_symlink():
            pointer = load_json_object(pointer_path)
            if set(pointer) != self.POINTER_FIELDS:
                raise RuntimeError(
                    "latest.json has an invalid schema; expected "
                    f"{sorted(self.POINTER_FIELDS)}, observed {sorted(pointer)}"
                )
            if pointer["contract_version"] != CONTRACT_VERSION:
                raise RuntimeError("latest.json uses an unsupported release contract")
            run_directory = validated_child_path(
                self.root, str(pointer["relative_run_directory"])
            )
            if run_directory.name != str(pointer["run_id"]):
                raise RuntimeError("latest.json run ID does not match its directory")
            if not run_directory.is_dir():
                raise RuntimeError(
                    f"latest.json is broken: selected run directory is missing: "
                    f"{run_directory}. Do not edit the pointer; rerun simUp.py or "
                    "run_experiment.ps1 to publish a complete run."
                )
        elif (self.root / "release_manifest.json").is_file() and (
            self.root / "RELEASE_COMMIT.json"
        ).is_file():
            pointer = None
            run_directory = self.root
        else:
            raise RuntimeError(
                "refusing an uncommitted or legacy analytical data directory; "
                "expected latest.json or a release_manifest.json/RELEASE_COMMIT.json pair"
            )

        manifest_path = run_directory / "release_manifest.json"
        commit_path = run_directory / "RELEASE_COMMIT.json"
        missing_control_files = [
            path.name for path in (manifest_path, commit_path) if not path.is_file()
        ]
        if missing_control_files:
            raise RuntimeError(
                f"selected analytical run {run_directory.name} is incomplete; missing "
                f"{missing_control_files}. Do not reconstruct release metadata manually; "
                "rerun simUp.py or run_experiment.ps1."
            )
        manifest = load_json_object(manifest_path)
        commit = load_json_object(commit_path)
        manifest_hash = sha256(manifest_path)
        commit_hash = sha256(commit_path)

        if manifest.get("contract_version") != CONTRACT_VERSION:
            raise RuntimeError("input manifest uses an unsupported release contract")
        if commit.get("contract_version") != CONTRACT_VERSION:
            raise RuntimeError("input release commit uses an unsupported contract")
        if commit.get("complete") is not True:
            raise RuntimeError("input RELEASE_COMMIT.json is not complete")
        run_id = str(manifest.get("run_id", ""))
        if not run_id or run_id != str(commit.get("run_id", "")):
            raise RuntimeError("manifest and commit disagree on the run ID")
        if run_directory != self.root and run_directory.name != run_id:
            raise RuntimeError("immutable run directory and manifest run ID disagree")
        if commit.get("release_manifest_sha256") != manifest_hash:
            raise RuntimeError("RELEASE_COMMIT.json does not bind release_manifest.json")
        if pointer is not None:
            if pointer["release_commit_sha256"] != commit_hash:
                raise RuntimeError("latest.json does not bind RELEASE_COMMIT.json")
            if pointer["configuration_sha256"] != manifest.get("configuration_sha256"):
                raise RuntimeError("latest.json and manifest disagree on configuration")

        gates = manifest.get("publication_gates")
        if not isinstance(gates, dict) or not gates:
            raise RuntimeError("input manifest contains no publication gates")
        failed_gates = sorted(name for name, value in gates.items() if value is not True)
        if failed_gates:
            raise RuntimeError(f"input publication gates failed: {failed_gates}")

        config = load_json_object(run_directory / "effective_config.json")
        config_hash = digest_bytes(canonical_json_bytes(config))
        if config_hash != manifest.get("configuration_sha256"):
            raise RuntimeError("effective configuration does not match the manifest")
        if config_hash != commit.get("configuration_sha256"):
            raise RuntimeError("effective configuration does not match the commit")

        inventory = self._validate_inventory(run_directory, manifest)
        self._validate_checksum_index(run_directory, inventory)
        self._validate_daily_contract(manifest, inventory)
        self._validate_reproducibility(run_directory, manifest, commit)
        return InputRelease(
            root=self.root,
            directory=run_directory,
            manifest=manifest,
            commit=commit,
            config=config,
            inventory=inventory,
            manifest_sha256=manifest_hash,
            commit_sha256=commit_hash,
        )

    def _validate_inventory(
        self,
        run_directory: Path,
        manifest: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        raw_entries = manifest.get("files")
        if not isinstance(raw_entries, list) or not raw_entries:
            raise RuntimeError("input manifest contains no file inventory")
        inventory: dict[str, dict[str, Any]] = {}
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict) or set(raw_entry) != self.INVENTORY_FIELDS:
                raise RuntimeError(f"invalid file-inventory entry: {raw_entry!r}")
            relative = str(raw_entry["path"])
            if relative in inventory:
                raise RuntimeError(f"duplicate file-inventory path: {relative}")
            path = validated_child_path(run_directory, relative)
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"manifested input is absent or unsafe: {relative}")
            if path.stat().st_size != int(raw_entry["size_bytes"]):
                raise RuntimeError(f"manifested size mismatch: {relative}")
            if sha256(path) != str(raw_entry["sha256"]):
                raise RuntimeError(f"manifested checksum mismatch: {relative}")
            inventory[relative] = dict(raw_entry)

        observed = {
            path.relative_to(run_directory).as_posix()
            for path in run_directory.rglob("*")
            if path.is_file()
        }
        expected = set(inventory) | {"release_manifest.json", "RELEASE_COMMIT.json"}
        if observed != expected:
            raise RuntimeError(
                f"input release file set is not exact; missing={sorted(expected-observed)}, "
                f"unexpected={sorted(observed-expected)}"
            )
        return inventory

    @staticmethod
    def _validate_checksum_index(
        run_directory: Path,
        inventory: Mapping[str, Mapping[str, Any]],
    ) -> None:
        path = run_directory / "output_checksums.csv"
        frame = pd.read_csv(path, keep_default_na=False)
        expected_columns = ["file", "sha256", "size_bytes", "row_count"]
        if list(frame.columns) != expected_columns:
            raise RuntimeError("output_checksums.csv has an invalid exact schema")
        expected_files = set(inventory) - {"output_checksums.csv"}
        if set(frame["file"].astype(str)) != expected_files:
            raise RuntimeError("output_checksums.csv does not cover the exact release")
        if frame["file"].astype(str).duplicated().any():
            raise RuntimeError("output_checksums.csv contains duplicate paths")
        for row in frame.to_dict(orient="records"):
            entry = inventory[str(row["file"])]
            if str(row["sha256"]) != str(entry["sha256"]):
                raise RuntimeError(f"checksum-index hash mismatch: {row['file']}")
            if int(row["size_bytes"]) != int(entry["size_bytes"]):
                raise RuntimeError(f"checksum-index size mismatch: {row['file']}")

    @staticmethod
    def _validate_daily_contract(
        manifest: Mapping[str, Any],
        inventory: Mapping[str, Mapping[str, Any]],
    ) -> None:
        daily_written = manifest.get("daily_samples_written")
        expected_rows = manifest.get("daily_samples_expected_rows")
        daily_entry = inventory.get("daily_stochastic_samples.csv")
        if daily_written is True:
            if daily_entry is None or int(daily_entry["row_count"]) != int(expected_rows):
                raise RuntimeError("daily-sample inventory disagrees with the manifest")
        elif daily_written is False:
            if daily_entry is not None or int(expected_rows) != 0:
                raise RuntimeError("manifest declares omitted daily samples but data remain")
        else:
            raise RuntimeError("daily_samples_written must be a JSON boolean")

    @staticmethod
    def _validate_reproducibility(
        run_directory: Path,
        manifest: Mapping[str, Any],
        commit: Mapping[str, Any],
    ) -> None:
        report = load_json_object(run_directory / "reproducibility_report.json")
        if report.get("passed") is not True:
            raise RuntimeError("input reproducibility report did not pass")
        fingerprint = report.get("final_release_frame_sha256")
        if fingerprint != manifest.get("deterministic_release_frame_sha256"):
            raise RuntimeError("reproducibility report and manifest disagree")
        if fingerprint != commit.get("deterministic_release_frame_sha256"):
            raise RuntimeError("reproducibility report and commit disagree")

    def _load_frames(self, release: InputRelease) -> dict[str, pd.DataFrame]:
        contract = dict(REQUIRED_COLUMNS)
        if release.manifest["daily_samples_written"] is not True:
            contract.pop("daily_stochastic_samples.csv")
        frames: dict[str, pd.DataFrame] = {}
        failures: list[str] = []
        for filename, required_columns in contract.items():
            entry = release.inventory.get(filename)
            if entry is None:
                failures.append(f"manifest omits {filename}")
                continue
            frame = pd.read_csv(release.directory / filename, keep_default_na=False)
            if len(frame) != int(entry["row_count"]):
                failures.append(
                    f"{filename} has {len(frame)} rows; manifest requires {entry['row_count']}"
                )
            if list(frame.columns) != list(entry["columns"]):
                failures.append(f"{filename} columns disagree with the manifest")
            missing = sorted(required_columns - set(frame.columns))
            if missing:
                failures.append(f"{filename} lacks required columns {missing}")
            frames[filename] = frame
        if failures:
            raise RuntimeError("invalid analytical table contract: " + "; ".join(failures))
        return frames

    def _validate_scientific_contract(
        self,
        release: InputRelease,
        frames: Mapping[str, pd.DataFrame],
    ) -> None:
        config = release.config
        analytical = frames["analytical_workload_results.csv"]
        configured_populations = tuple(int(value) for value in config["populations"])
        observed_populations = tuple(
            sorted(int(value) for value in analytical["population"].unique())
        )
        if configured_populations != REQUIRED_POPULATIONS:
            raise RuntimeError("configuration does not contain the required populations")
        if observed_populations != REQUIRED_POPULATIONS:
            raise RuntimeError("analytical results do not contain the required populations")

        definitions = frames["scenario_definitions.csv"]
        if not pd.api.types.is_numeric_dtype(definitions["der_adoption_fraction"].dtype):
            raise RuntimeError("scenario DER adoption must be numeric")
        scenario_order = tuple(definitions["name"].astype(str))
        configured_scenarios = tuple(str(item["name"]) for item in config["scenarios"])
        if scenario_order != configured_scenarios:
            raise RuntimeError("scenario definitions do not preserve configuration order")
        if set(analytical["scenario"].astype(str)) != set(scenario_order):
            raise RuntimeError("analytical scenarios and definitions disagree")
        expected_cases = len(REQUIRED_POPULATIONS) * len(scenario_order)
        if len(analytical) != expected_cases or analytical.duplicated(
            ["population", "scenario"]
        ).any():
            raise RuntimeError("analytical results are not one-to-one by population/scenario")
        if not analytical["scope"].astype(str).eq(SCOPE_NOTE).all():
            raise RuntimeError("analytical scope boundary is missing or altered")

        component_columns = list(COMPONENT_ORDER)
        component_error = (
            analytical[component_columns].sum(axis=1)
            - analytical["coordination_events_day"]
        ).abs()
        relative_error = float(component_error.max()) / float(
            analytical["coordination_events_day"].max()
        )
        if relative_error >= 5e-12:
            raise RuntimeError("recomputed component-sum identity failed")

        checks = frames["verification_checks.csv"]
        passed_checks = parse_boolean_series(checks, "passed", "verification checks")
        if checks["check"].astype(str).duplicated().any() or not passed_checks.all():
            failed = list(checks.loc[~passed_checks, "check"].astype(str))
            raise RuntimeError(f"simulation verification checks failed: {failed}")
        check_lookup = dict(zip(checks["check"].astype(str), passed_checks))

        statistical = frames["statistical_validation_metrics.csv"]
        mean_pass = statistical["standardized_mean_error_z"].abs().le(
            statistical["mean_z_critical"]
        )
        dispersion_pass = statistical["poisson_dispersion_index"].between(
            statistical["dispersion_acceptance_lower"],
            statistical["dispersion_acceptance_upper"],
            inclusive="both",
        )
        stored_mean = parse_boolean_series(statistical, "mean_test_passed", "mean tests")
        stored_dispersion = parse_boolean_series(
            statistical, "dispersion_test_passed", "dispersion tests"
        )
        if not (
            mean_pass.all()
            and dispersion_pass.all()
            and np.array_equal(mean_pass.to_numpy(), stored_mean.to_numpy())
            and np.array_equal(dispersion_pass.to_numpy(), stored_dispersion.to_numpy())
        ):
            raise RuntimeError("recomputed mean or dispersion validation failed")

        claim_validation = frames["claim_matrix_validation.csv"]
        if not parse_boolean_series(
            claim_validation, "passed", "claim-matrix validation"
        ).all():
            raise RuntimeError("claim matrix validation is false")
        claims = frames["claim_matrix.csv"].fillna("")
        if claims["claim_id"].astype(str).duplicated().any():
            raise RuntimeError("claim matrix contains duplicate claim IDs")
        for row in claims.to_dict(orient="records"):
            claim_id = str(row["claim_id"])
            if row["status"] == "prohibited":
                if not str(row["prohibited_interpretation"]).strip():
                    raise RuntimeError(f"prohibited claim {claim_id} lacks a boundary")
                continue
            for evidence in filter(None, str(row["evidence_files"]).split(";")):
                if evidence not in release.inventory:
                    raise RuntimeError(f"claim {claim_id} references missing {evidence}")
            for check_name in filter(None, str(row["required_checks"]).split(";")):
                if check_lookup.get(check_name) is not True:
                    raise RuntimeError(f"claim {claim_id} depends on failed {check_name}")

        reproduction = frames["reproducibility_check.csv"]
        if not parse_boolean_series(
            reproduction, "passed", "reproducibility check"
        ).all():
            raise RuntimeError("input reproducibility check is false")

        if release.manifest["daily_samples_written"] is True:
            self._validate_daily_rows(release, frames["daily_stochastic_samples.csv"], scenario_order)

        for filename, frame in frames.items():
            for column in text_columns(frame):
                for value in frame[column].dropna().astype(str):
                    validate_unicode_bytes(
                        value.encode("utf-8"), f"{filename}:{column}"
                    )
            numeric = frame.select_dtypes(include=[np.number])
            if numeric.empty:
                continue
            finite_by_column = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=0)
            invalid_columns = set(numeric.columns[~finite_by_column])
            allowed = {"oat_sensitivity.csv": {"secant_elasticity"}}.get(
                filename, set()
            )
            if not invalid_columns <= allowed:
                raise RuntimeError(
                    f"{filename} has non-finite values in {sorted(invalid_columns)}"
                )

    @staticmethod
    def _validate_daily_rows(
        release: InputRelease,
        daily: pd.DataFrame,
        scenario_order: Sequence[str],
    ) -> None:
        replications = int(release.config["replications"])
        days = int(release.config["days_per_replication"])
        expected_per_case = replications * days
        expected_total = len(REQUIRED_POPULATIONS) * len(scenario_order) * expected_per_case
        if len(daily) != expected_total:
            raise RuntimeError("daily sample total does not match the configuration")
        if len(daily) != int(release.manifest["daily_samples_expected_rows"]):
            raise RuntimeError("daily sample total does not match the manifest")
        duplicated = daily.duplicated(["population", "scenario", "replication", "day"])
        if duplicated.any():
            raise RuntimeError("daily samples contain duplicate case/replication/day keys")
        expected_replications = set(range(1, replications + 1))
        expected_days = set(range(1, days + 1))
        for (population, scenario), case in daily.groupby(
            ["population", "scenario"], sort=False
        ):
            if len(case) != expected_per_case:
                raise RuntimeError(
                    f"daily case {population}/{scenario} has an invalid row count"
                )
            if set(case["replication"].astype(int)) != expected_replications:
                raise RuntimeError(f"daily case {population}/{scenario} misses replications")
            if set(case["day"].astype(int)) != expected_days:
                raise RuntimeError(f"daily case {population}/{scenario} misses days")


def build_context(
    args: argparse.Namespace,
    figure_stage: Path,
    release: InputRelease,
    frames: dict[str, pd.DataFrame],
) -> PlotContext:
    manifest = release.manifest
    definitions = frames["scenario_definitions.csv"]
    scenario_order = tuple(definitions["name"].astype(str))
    scenario_labels = dict(
        zip(definitions["name"].astype(str), definitions["label"].astype(str))
    )
    populations = tuple(
        sorted(
            int(value)
            for value in frames["analytical_workload_results.csv"]["population"].unique()
        )
    )
    active_specs = tuple(
        spec
        for spec in FIGURE_SPECS
        if (not args.primary_only or spec.number in PRIMARY_FIGURE_NUMBERS)
        and (
            spec.number != 5
            or manifest["daily_samples_written"] is True
        )
    )
    if not active_specs:
        raise RuntimeError("figure selection produced an empty release")
    return PlotContext(
        release=release,
        data_root=release.root,
        data_dir=release.directory,
        figure_dir=figure_stage,
        formats=args.formats,
        dpi=args.dpi,
        frames=frames,
        scenario_order=scenario_order,
        scenario_labels=scenario_labels,
        populations=populations,
        input_manifest=release.manifest,
        input_commit=release.commit,
        input_config=release.config,
        active_specs=active_specs,
    )

def scenario_color(scenario: str) -> str:
    return SCENARIO_COLORS.get(scenario, "#4B5563")


def wrapped_scenario_label(context: PlotContext, scenario: str) -> str:
    label = context.scenario_labels.get(scenario, scenario.replace("_", " ").title())
    return "\n".join(textwrap.wrap(label, width=14))


def population_label(population: int | float) -> str:
    return f"{float(population) / 1_000_000:.0f}M"


def beautify_axis(axis: plt.Axes, grid_axis: str = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, axis=grid_axis)
    axis.set_axisbelow(True)


def add_panel_labels(axes: Sequence[plt.Axes]) -> None:
    for index, axis in enumerate(axes):
        axis.text(
            -0.12,
            1.06,
            f"({chr(ord('a') + index)})",
            transform=axis.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
        )


def figure_section(spec: FigureSpec) -> str:
    return "primary" if spec.number in PRIMARY_FIGURE_NUMBERS else "supplement"


def save_figure(
    figure: plt.Figure,
    spec: FigureSpec,
    context: PlotContext,
) -> None:
    figure.suptitle(spec.title, fontsize=10, fontweight="semibold")
    section_dir = context.figure_dir / figure_section(spec)
    section_dir.mkdir(parents=True, exist_ok=True)
    for output_format in context.formats:
        path = section_dir / f"{spec.basename}.{output_format}"
        rendered = BytesIO()
        if output_format == "pdf":
            metadata = {
                "Title": spec.title,
                "Author": "HTGL analytical workload study",
                "Subject": SCOPE_NOTE,
                "Keywords": "HTGL, analytical workload characterization",
                "Creator": CODE_VERSION,
                "CreationDate": None,
                "ModDate": None,
            }
            figure.savefig(
                rendered,
                format="pdf",
                bbox_inches="tight",
                pad_inches=0.06,
                metadata=metadata,
            )
        else:
            metadata = {
                "Title": spec.title,
                "Description": spec.caption,
                "Software": CODE_VERSION,
            }
            figure.savefig(
                rendered,
                format="png",
                dpi=context.dpi,
                bbox_inches="tight",
                pad_inches=0.06,
                metadata=metadata,
            )
        payload = rendered.getvalue()
        rendered.close()
        # Validate the in-memory render before publishing it. This catches a
        # renderer failure without creating a partial filesystem artifact.
        if output_format == "pdf":
            if not payload.startswith(b"%PDF"):
                raise RuntimeError(f"Invalid rendered PDF for {path.name}")
        else:
            with Image.open(BytesIO(payload)) as rendered_image:
                rendered_image.verify()
        # Write only validated bytes, then synchronize the file before making
        # it available to the paper build or post-render validator.
        with path.open("wb") as output_stream:
            output_stream.write(payload)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        context.created_paths.append(path)
    plt.close(figure)


def plot_01_baseline_component_decomposition(context: PlotContext) -> None:
    spec = SPEC_BY_NUMBER[1]
    data = context.frames["component_decomposition.csv"]
    data = data[data["scenario"] == "baseline"].copy()
    pivot = data.pivot(
        index="population", columns="component", values="expected_events_day"
    ).reindex(index=context.populations, columns=COMPONENT_ORDER, fill_value=0.0)

    figure, axis = plt.subplots(figsize=(7.2, 3.45), constrained_layout=True)
    x = np.arange(len(pivot.index))
    bottom = np.zeros(len(pivot))
    for component in COMPONENT_ORDER:
        values = pivot[component].to_numpy(dtype=float) / 1e6
        axis.bar(
            x,
            values,
            bottom=bottom,
            width=0.62,
            color=COMPONENT_COLORS[component],
            label=COMPONENT_LABELS[component],
            edgecolor="white",
            linewidth=0.45,
        )
        bottom += values
    for position, total in zip(x, bottom):
        axis.text(position, total * 1.015, f"{total:.1f}M", ha="center", va="bottom")
    axis.set_xticks(x, [population_label(value) for value in pivot.index])
    axis.set_xlabel("Consumer population")
    axis.set_ylabel("Expected coordination events/day (millions)")
    axis.set_ylim(0, bottom.max() * 1.15)
    axis.legend(ncol=2, frameon=False, loc="upper left")
    beautify_axis(axis)
    save_figure(figure, spec, context)


def plot_02_scenario_composition(context: PlotContext) -> None:
    spec = SPEC_BY_NUMBER[2]
    data = context.frames["component_decomposition.csv"]
    data = data[data["population"] == REQUIRED_POPULATIONS[0]].copy()
    pivot = data.pivot(
        index="scenario", columns="component", values="share_pct"
    ).reindex(index=context.scenario_order, columns=COMPONENT_ORDER, fill_value=0.0)

    figure, axis = plt.subplots(figsize=(7.2, 3.55), constrained_layout=True)
    x = np.arange(len(pivot.index))
    bottom = np.zeros(len(pivot))
    for component in COMPONENT_ORDER:
        values = pivot[component].to_numpy(dtype=float)
        axis.bar(
            x,
            values,
            bottom=bottom,
            width=0.68,
            color=COMPONENT_COLORS[component],
            label=COMPONENT_LABELS[component],
            edgecolor="white",
            linewidth=0.45,
        )
        bottom += values
    axis.set_xticks(
        x, [wrapped_scenario_label(context, value) for value in pivot.index]
    )
    axis.set_ylabel("Share of expected coordination events (%)")
    axis.set_ylim(0, 103)
    axis.yaxis.set_major_locator(mticker.MultipleLocator(20))
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=4,
        frameon=False,
        loc="outside lower center",
    )
    beautify_axis(axis)
    save_figure(figure, spec, context)


def plot_03_population_scenario_scaling(context: PlotContext) -> None:
    spec = SPEC_BY_NUMBER[3]
    data = context.frames["analytical_workload_results.csv"]
    figure, axis = plt.subplots(figsize=(7.2, 3.55), constrained_layout=True)
    for scenario in context.scenario_order:
        subset = data[data["scenario"] == scenario].sort_values("population")
        axis.plot(
            subset["population_millions"],
            subset["expected_tps"],
            marker="o",
            color=scenario_color(scenario),
            label=context.scenario_labels[scenario],
        )
    axis.set_xticks([50, 150, 450], ["50M", "150M", "450M"])
    axis.set_xlabel("Consumer population")
    axis.set_ylabel("Expected average offered TPS")
    axis.set_xlim(35, 465)
    axis.set_ylim(bottom=0)
    axis.legend(ncol=2, frameon=False, loc="upper left")
    beautify_axis(axis)
    save_figure(figure, spec, context)


def plot_04_analytical_stochastic_alignment(context: PlotContext) -> None:
    spec = SPEC_BY_NUMBER[4]
    data = context.frames["analytical_vs_stochastic.csv"].copy()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.35, 3.35),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.05, 1.0)},
    )
    left, right = axes
    for _, row in data.iterrows():
        population_millions = float(row["population_millions"])
        left.scatter(
            row["expected_tps"],
            row["daily_count_equivalent_tps_mean"],
            color=scenario_color(str(row["scenario"])),
            marker=POPULATION_MARKERS[population_millions],
            s=34,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        right.scatter(
            row["expected_tps"],
            row["mean_relative_error_pct"] * 10_000.0,
            color=scenario_color(str(row["scenario"])),
            marker=POPULATION_MARKERS[population_millions],
            s=34,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
    minimum = min(
        data["expected_tps"].min(),
        data["daily_count_equivalent_tps_mean"].min(),
    ) * 0.9
    maximum = max(
        data["expected_tps"].max(),
        data["daily_count_equivalent_tps_mean"].max(),
    ) * 1.1
    left.plot([minimum, maximum], [minimum, maximum], "--", color="#374151", lw=1.0)
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlim(minimum, maximum)
    left.set_ylim(minimum, maximum)
    left.set_xlabel("Closed-form expected TPS")
    left.set_ylabel("Pooled daily-count-equivalent mean TPS")
    beautify_axis(left, grid_axis="both")

    right.axhline(0.0, color="#374151", lw=0.8)
    right.set_xscale("log")
    right.set_xlabel("Closed-form expected TPS")
    right.set_ylabel("Stochastic mean error (ppm)")
    beautify_axis(right, grid_axis="both")

    scenario_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=scenario_color(scenario),
            markeredgecolor="none",
            label=context.scenario_labels[scenario],
        )
        for scenario in context.scenario_order
    ]
    population_handles = [
        Line2D(
            [0],
            [0],
            marker=POPULATION_MARKERS[float(population / 1e6)],
            linestyle="none",
            color="#374151",
            label=population_label(population),
        )
        for population in context.populations
    ]
    figure.legend(
        handles=scenario_handles + population_handles,
        ncol=4,
        frameon=False,
        loc="outside lower center",
    )
    add_panel_labels(axes)
    save_figure(figure, spec, context)


def plot_05_baseline_daily_variability(context: PlotContext) -> None:
    spec = SPEC_BY_NUMBER[5]
    samples = context.frames["daily_stochastic_samples.csv"]
    analytical = context.frames["analytical_workload_results.csv"]
    analytical = analytical[analytical["scenario"] == "baseline"].set_index("population")

    figure, axes = plt.subplots(
        1, 3, figsize=(7.45, 2.95), constrained_layout=True, sharey=False
    )
    for axis, population in zip(axes, context.populations):
        subset = samples[
            (samples["population"] == population)
            & (samples["scenario"] == "baseline")
        ]
        expected = float(analytical.loc[population, "expected_tps"])
        theoretical_sd = float(
            analytical.loc[population, "daily_count_sd_equivalent_tps"]
        )
        deviations = (
            subset["daily_count_equivalent_tps"].to_numpy(dtype=float) - expected
        )
        axis.hist(
            deviations,
            bins=36,
            density=True,
            color="#B9DDF1",
            edgecolor="#0072B2",
            linewidth=0.4,
            label="Modeled days",
        )
        x = np.linspace(deviations.min(), deviations.max(), 350)
        density = np.exp(-0.5 * (x / theoretical_sd) ** 2) / (
            theoretical_sd * math.sqrt(2.0 * math.pi)
        )
        axis.plot(x, density, color="#D55E00", label="Poisson-normal reference")
        axis.axvline(0.0, color="#374151", lw=0.8)
        axis.set_title(f"{population_label(population)} consumers")
        axis.set_xlabel("Daily-count-equivalent TPS deviation")
        beautify_axis(axis)
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False, loc="best")
    add_panel_labels(axes)
    save_figure(figure, spec, context)


def plot_06_replication_stability(context: PlotContext) -> None:
    spec = SPEC_BY_NUMBER[6]
    data = context.frames["replication_summary.csv"].copy()
    data["error_ppm"] = data["mean_relative_error_pct"] * 10_000.0

    figure, axes = plt.subplots(
        1, 3, figsize=(7.5, 3.25), constrained_layout=True, sharey=True
    )
    for axis, population in zip(axes, context.populations):
        subset = data[data["population"] == population]
        distributions = [
            subset.loc[subset["scenario"] == scenario, "error_ppm"].to_numpy()
            for scenario in context.scenario_order
        ]
        boxplot = axis.boxplot(
            distributions,
            tick_labels=[str(index + 1) for index in range(len(distributions))],
            patch_artist=True,
            widths=0.64,
            showfliers=False,
            medianprops={"color": "#111827", "linewidth": 1.0},
            whiskerprops={"color": "#4B5563", "linewidth": 0.8},
            capprops={"color": "#4B5563", "linewidth": 0.8},
        )
        for patch, scenario in zip(boxplot["boxes"], context.scenario_order):
            patch.set_facecolor(scenario_color(scenario))
            patch.set_alpha(0.72)
            patch.set_edgecolor("#374151")
        axis.axhline(0.0, color="#374151", lw=0.8)
        axis.set_title(f"{population_label(population)} consumers")
        axis.set_xlabel("Scenario index")
        beautify_axis(axis)
    axes[0].set_ylabel("Replication-mean error (ppm)")
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor=scenario_color(scenario),
            markeredgecolor="none",
            label=f"{index + 1}: {context.scenario_labels[scenario]}",
        )
        for index, scenario in enumerate(context.scenario_order)
    ]
    figure.legend(
        handles=legend_handles,
        ncol=2,
        frameon=False,
        loc="outside lower center",
    )
    add_panel_labels(axes)
    save_figure(figure, spec, context)


def plot_07_convergence_diagnostics(context: PlotContext) -> None:
    spec = SPEC_BY_NUMBER[7]
    data = context.frames["convergence_diagnostics.csv"].copy()
    data["absolute_error_ppm"] = np.maximum(
        data["relative_error_pct"].abs() * 10_000.0, 1e-4
    )

    figure, axes = plt.subplots(
        1, 3, figsize=(7.5, 3.15), constrained_layout=True, sharey=True
    )
    for axis, population in zip(axes, context.populations):
        subset = data[data["population"] == population]
        for scenario_index, scenario in enumerate(context.scenario_order):
            series = subset[subset["scenario"] == scenario].sort_values(
                "observations_included"
            )
            axis.plot(
                series["observations_included"],
                series["absolute_error_ppm"],
                marker="o",
                color=scenario_color(scenario),
                label=context.scenario_labels[scenario],
            )
            theoretical_95_ppm = (
                1.95996398454
                * 1_000_000.0
                / np.sqrt(
                    series["analytical_events_day"].to_numpy(dtype=float)
                    * series["observations_included"].to_numpy(dtype=float)
                )
            )
            axis.plot(
                series["observations_included"],
                theoretical_95_ppm,
                linestyle="--",
                linewidth=0.9,
                alpha=0.75,
                color=scenario_color(scenario),
                label=(
                    "Poisson 95% mean-error envelope"
                    if scenario_index == 0
                    else "_nolegend_"
                ),
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(f"{population_label(population)} consumers")
        axis.set_xlabel("Pooled daily observations")
        beautify_axis(axis, grid_axis="both")
    axes[0].set_ylabel("Absolute stochastic-mean error (ppm)")
    axes[-1].legend(frameon=False, loc="best", fontsize=5.8)
    add_panel_labels(axes)
    save_figure(figure, spec, context)


def plot_08_oat_sensitivity(context: PlotContext) -> None:
    spec = SPEC_BY_NUMBER[8]
    data = context.frames["oat_sensitivity.csv"]
    figure, axis = plt.subplots(
        figsize=(7.35, 3.85), constrained_layout=True
    )
    positions = np.arange(len(PARAMETER_ORDER), dtype=float)
    offsets = np.linspace(-0.22, 0.22, len(context.populations))
    population_colors = ("#0072B2", "#009E73", "#D55E00")
    for offset, color, population in zip(
        offsets, population_colors, context.populations
    ):
        subset = data[
            (data["population"] == population)
            & data["varied_parameter"].isin(PARAMETER_ORDER)
        ]
        bounds = (
            subset.groupby("varied_parameter")["workload_change_pct"]
            .agg(["min", "max"])
            .reindex(PARAMETER_ORDER)
        )
        y = positions + offset
        axis.hlines(
            y,
            bounds["min"],
            bounds["max"],
            color=color,
            linewidth=1.9,
            alpha=0.88,
            zorder=1,
        )
        axis.scatter(
            bounds["min"],
            y,
            color=color,
            marker="|",
            s=55,
            zorder=2,
        )
        axis.scatter(
            bounds["max"],
            y,
            color=color,
            marker="|",
            s=55,
            zorder=2,
            label=f"{population_label(population)} consumers",
        )
    axis.axvline(0.0, color="#111827", lw=0.8)
    axis.set_xlabel("Modeled workload change over configured input range (%)")
    axis.set_yticks(
        positions,
        [PARAMETER_LABELS[parameter] for parameter in PARAMETER_ORDER],
    )
    axis.invert_yaxis()
    axis.legend(frameon=False, loc="lower right", ncol=3)
    beautify_axis(axis, grid_axis="x")
    save_figure(figure, spec, context)


def plot_09_authorization_der_grid(context: PlotContext) -> None:
    spec = SPEC_BY_NUMBER[9]
    data = context.frames["authorization_der_assumption_grid.csv"]
    figure, axes = plt.subplots(
        1, 3, figsize=(7.55, 3.3), constrained_layout=True, sharex=True, sharey=True
    )
    vmin = float(data["expected_tps"].min())
    vmax = float(data["expected_tps"].max())
    image = None
    for axis, population in zip(axes, context.populations):
        subset = data[data["population"] == population]
        pivot = subset.pivot(
            index="der_adoption_fraction",
            columns="authorization_events_per_consumer_day",
            values="expected_tps",
        ).sort_index().sort_index(axis=1)
        extent = [
            float(pivot.columns.min()),
            float(pivot.columns.max()),
            float(pivot.index.min()),
            float(pivot.index.max()),
        ]
        image = axis.imshow(
            pivot.to_numpy(dtype=float),
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        axis.scatter(
            [0.5],
            [0.10],
            marker="x",
            s=42,
            linewidth=1.5,
            color="white",
            zorder=3,
        )
        axis.set_title(f"{population_label(population)} consumers")
        axis.set_xlabel("Authorization events/consumer/day")
    axes[0].set_ylabel("DER adoption fraction")
    if image is not None:
        colorbar = figure.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
        colorbar.set_label("Expected average offered TPS")
        colorbar.ax.tick_params(labelsize=6)
    add_panel_labels(axes)
    save_figure(figure, spec, context)


def plot_10_local_elasticities(context: PlotContext) -> None:
    spec = SPEC_BY_NUMBER[10]
    data = context.frames["local_elasticities.csv"]
    elasticity_parameters = (
        "consumer_population",
        "authorization_events_per_consumer_day",
        "der_adoption_fraction",
        "metadata_events_per_der_day",
        "consent_changes_per_consumer_year",
        "institutional_actors",
        "integrity_events_per_actor_day",
    )
    figure, axes = plt.subplots(
        1, 3, figsize=(7.65, 4.25), constrained_layout=True, sharey=True
    )
    image = None
    for axis, population in zip(axes, context.populations):
        subset = data[data["population"] == population]
        pivot = subset.pivot(
            index="scenario", columns="parameter", values="local_elasticity"
        ).reindex(index=context.scenario_order, columns=elasticity_parameters)
        values = pivot.to_numpy(dtype=float)
        image = axis.imshow(values, vmin=0.0, vmax=1.0, cmap="YlGnBu", aspect="auto")
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                annotation = (
                    f"{value:.1e}"
                    if 0.0 < abs(value) < 0.005
                    else f"{value:.3f}"
                )
                axis.text(
                    column_index,
                    row_index,
                    annotation,
                    ha="center",
                    va="center",
                    fontsize=5.6,
                    color="white" if value > 0.58 else "#111827",
                )
        axis.set_title(f"{population_label(population)} consumers")
        axis.set_xticks(
            np.arange(len(elasticity_parameters)),
            [PARAMETER_LABELS[value] for value in elasticity_parameters],
            rotation=55,
            ha="right",
        )
        axis.set_yticks(
            np.arange(len(context.scenario_order)),
            [wrapped_scenario_label(context, value) for value in context.scenario_order],
        )
    if image is not None:
        colorbar = figure.colorbar(image, ax=axes, shrink=0.78, pad=0.02)
        colorbar.set_label("Local elasticity")
    add_panel_labels(axes)
    save_figure(figure, spec, context)


def plot_11_temporal_profiles(context: PlotContext) -> None:
    spec = SPEC_BY_NUMBER[11]
    data = context.frames["temporal_profile_detail.csv"]
    profile_order = ("uniform", "diurnal_30pct", "diurnal_60pct")
    reference_population = context.populations[0]
    baseline = data[
        (data["scenario"] == "baseline")
        & (data["population"] == reference_population)
    ]

    figure, axes = plt.subplots(
        1, 2, figsize=(7.5, 3.2), constrained_layout=True
    )
    shape_axis, peak_axis = axes
    for profile in profile_order:
        series = baseline[baseline["temporal_profile"] == profile].sort_values(
            "hour_start"
        )
        if series.empty:
            continue
        expected = series["expected_tps"].to_numpy(dtype=float)
        shape_axis.plot(
            series["hour_start"] + 0.5,
            expected / expected.mean(),
            color=PROFILE_COLORS[profile],
            label=str(series["profile_label"].iloc[0]),
        )
    shape_axis.axhline(1.0, color="#6B7280", linewidth=0.7, linestyle=":")
    shape_axis.set_xlabel("Hour of day")
    shape_axis.set_ylabel("Normalized expected intensity")
    shape_axis.set_xlim(0, 24)
    shape_axis.set_xticks([0, 6, 12, 18, 24])
    shape_axis.legend(frameon=False, loc="best")
    beautify_axis(shape_axis)

    high_profile = data[data["temporal_profile"] == "diurnal_60pct"]
    peaks = (
        high_profile.groupby(
            ["population", "population_millions", "scenario"],
            as_index=False,
        )["expected_tps"]
        .max()
        .sort_values("population")
    )
    for scenario in context.scenario_order:
        series = peaks[peaks["scenario"] == scenario]
        peak_axis.plot(
            series["population_millions"],
            series["expected_tps"],
            marker="o",
            color=scenario_color(scenario),
            label=context.scenario_labels[scenario],
        )
    peak_axis.set_xlabel("Consumer population (millions)")
    peak_axis.set_ylabel("High-envelope peak expected TPS")
    peak_axis.legend(frameon=False, fontsize=5.8, loc="upper left")
    beautify_axis(peak_axis)
    add_panel_labels(axes)
    save_figure(figure, spec, context)


def plot_12_logical_metadata_volume(context: PlotContext) -> None:
    spec = SPEC_BY_NUMBER[12]
    data = context.frames["logical_metadata_volume.csv"]
    data = data[data["horizon_days"] == 365]
    size_order = ("nominal", "conservative")

    figure, axes = plt.subplots(
        1, 3, figsize=(7.55, 3.35), constrained_layout=True
    )
    x = np.arange(len(context.scenario_order))
    width = 0.36
    for axis, population in zip(axes, context.populations):
        subset = data[data["population"] == population]
        pivot = subset.pivot(
            index="scenario",
            columns="record_size_assumption",
            values="logical_tb_decimal",
        ).reindex(index=context.scenario_order, columns=size_order)
        for offset_index, size_name in enumerate(size_order):
            offset = (offset_index - 0.5) * width
            axis.bar(
                x + offset,
                pivot[size_name],
                width=width,
                color=STORAGE_COLORS[size_name],
                label=(
                    "Component-specific nominal"
                    if size_name == "nominal"
                    else "Component-specific conservative"
                ),
                edgecolor="white",
                linewidth=0.4,
            )
        axis.set_title(f"{population_label(population)} consumers")
        axis.set_xticks(
            x,
            [str(index + 1) for index in range(len(context.scenario_order))],
        )
        axis.set_xlabel("Scenario index")
        axis.set_ylim(bottom=0)
        beautify_axis(axis)
    axes[0].set_ylabel("Logical metadata volume (TB/year)")
    axes[-1].legend(frameon=False, loc="upper left")
    scenario_note = "   ".join(
        f"{index + 1}: {context.scenario_labels[scenario]}"
        for index, scenario in enumerate(context.scenario_order)
    )
    figure.text(0.5, -0.03, scenario_note, ha="center", fontsize=6.5)
    add_panel_labels(axes)
    save_figure(figure, spec, context)


def plot_13_event_count_boundary(context: PlotContext) -> None:
    spec = SPEC_BY_NUMBER[13]
    data = context.frames["workload_boundary_event_counts.csv"]
    pivot = data.pivot(
        index="population",
        columns="scenario",
        values="operational_to_coordination_event_count_ratio",
    ).reindex(index=context.populations, columns=context.scenario_order)
    values = pivot.to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(7.2, 3.25), constrained_layout=True)
    image = axis.imshow(values, cmap="YlGnBu", aspect="auto")
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.1f}×",
                ha="center",
                va="center",
                fontweight="semibold",
                color="white" if value > np.nanmedian(values) else "#111827",
            )
    axis.set_xticks(
        np.arange(len(context.scenario_order)),
        [wrapped_scenario_label(context, value) for value in context.scenario_order],
    )
    axis.set_yticks(
        np.arange(len(context.populations)),
        [population_label(value) for value in context.populations],
    )
    axis.set_xlabel("Analytical workload scenario")
    axis.set_ylabel("Consumer population")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.82, pad=0.02)
    colorbar.set_label("Operational/coordination event-count ratio")
    save_figure(figure, spec, context)


def plot_14_reference_workload_envelopes(context: PlotContext) -> None:
    spec = SPEC_BY_NUMBER[14]
    data = context.frames["reference_workload_envelopes.csv"]
    figure, axis = plt.subplots(figsize=(7.2, 3.65), constrained_layout=True)
    for scenario in context.scenario_order:
        subset = data[data["scenario"] == scenario].sort_values(
            "reference_offered_load_budget_tps"
        )
        axis.plot(
            subset["reference_offered_load_budget_tps"],
            subset["analytical_population_at_budget_millions"],
            marker="o",
            color=scenario_color(scenario),
            label=context.scenario_labels[scenario],
        )
    axis.set_xscale("log")
    axis.set_xlabel("Generic offered-load budget (TPS)")
    axis.set_ylabel("Population at analytical workload boundary (millions)")
    axis.xaxis.set_major_formatter(mticker.ScalarFormatter())
    axis.xaxis.set_minor_formatter(mticker.NullFormatter())
    axis.set_ylim(bottom=0)
    axis.legend(ncol=2, frameon=False, loc="upper left")
    beautify_axis(axis, grid_axis="both")
    axis.text(
        0.99,
        0.02,
        "Generic budgets only—no HTGL or ledger benchmark",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="#4B5563",
    )
    save_figure(figure, spec, context)


PLOT_FUNCTIONS: tuple[Callable[[PlotContext], None], ...] = (
    plot_01_baseline_component_decomposition,
    plot_02_scenario_composition,
    plot_03_population_scenario_scaling,
    plot_04_analytical_stochastic_alignment,
    plot_05_baseline_daily_variability,
    plot_06_replication_stability,
    plot_07_convergence_diagnostics,
    plot_08_oat_sensitivity,
    plot_09_authorization_der_grid,
    plot_10_local_elasticities,
    plot_11_temporal_profiles,
    plot_12_logical_metadata_volume,
    plot_13_event_count_boundary,
    plot_14_reference_workload_envelopes,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


FIGURE_CLAIMS: Mapping[int, str] = {
    1: "C03",
    2: "C03",
    3: "C01;C02",
    4: "C07",
    5: "C07",
    6: "C07",
    7: "C07",
    8: "C04",
    9: "C04",
    10: "C04",
    11: "C05",
    12: "C06",
    13: "C01",
    14: "C01",
}


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary_path.unlink()
        raise


def write_json(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    validate_unicode_bytes(payload, path.name)
    atomic_write_bytes(path, payload)


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("numpy", "pandas", "matplotlib", "Pillow"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"required plotting dependency {distribution!r} is not installed"
            ) from exc
    return versions


def source_control_snapshot() -> dict[str, object]:
    script_directory = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=script_directory,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=script_directory,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        return {
            "available": True,
            "commit": commit,
            "tracked_tree_dirty": bool(status.strip()),
        }
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "commit": None, "tracked_tree_dirty": None}


def font_fingerprints() -> list[dict[str, object]]:
    requested = ("DejaVu Serif", "Times New Roman", "Times")
    path = Path(
        font_manager.findfont(
            font_manager.FontProperties(family=list(requested)),
            fallback_to_default=True,
        )
    ).resolve()
    resolved_name = font_manager.FontProperties(fname=str(path)).get_name()
    return [
        {
            "requested_family_stack": list(requested),
            "resolved_family": resolved_name,
            "font_file": path.name,
            "sha256": sha256(path),
        }
    ]


def validate_required_glyphs(fonts: Sequence[Mapping[str, object]]) -> None:
    if not fonts:
        raise RuntimeError("no publication font was resolved")
    resolved_path = Path(
        font_manager.findfont(
            font_manager.FontProperties(
                family=[str(fonts[0]["resolved_family"])]
            ),
            fallback_to_default=True,
        )
    )
    character_map = font_manager.get_font(str(resolved_path)).get_charmap()
    required = {0x00D7: "multiplication sign", 0x2014: "em dash"}
    missing = [name for codepoint, name in required.items() if codepoint not in character_map]
    if missing:
        raise RuntimeError(
            f"resolved publication font lacks required glyphs: {missing}"
        )


def environment_snapshot() -> dict[str, object]:
    fonts = font_fingerprints()
    validate_required_glyphs(fonts)
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "packages": package_versions(),
        "matplotlib_backend": matplotlib.get_backend(),
        "platform": platform.platform(),
        "operating_system": platform.system(),
        "machine": platform.machine(),
        "fonts": fonts,
        "source_control": source_control_snapshot(),
    }


def manifest_command_line() -> list[str]:
    result = [Path(sys.argv[0]).name]
    for argument in sys.argv[1:]:
        option, separator, value = argument.partition("=")
        if separator and Path(value).is_absolute():
            result.append(f"{option}=<absolute>/{Path(value).name}")
        elif Path(argument).is_absolute():
            result.append(f"<absolute>/{Path(argument).name}")
        else:
            result.append(argument)
    return result


def create_staging_directory(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise RuntimeError(f"figure publication root must not be a symbolic link: {root}")
    return Path(tempfile.mkdtemp(prefix=".staging-", dir=root))


@contextlib.contextmanager
def publication_lock(root: Path) -> Iterable[None]:
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".publication.lock"
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f"publication lock exists at {lock_path}; inspect it before retrying"
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            lock_path.unlink()


def publish_immutable_run(stage: Path, root: Path, run_id: str) -> Path:
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    destination = runs / run_id
    if destination.exists():
        raise RuntimeError(f"immutable figure run already exists: {destination}")
    os.replace(stage, destination)
    return destination


def write_latest_pointer(
    root: Path,
    run_id: str,
    commit_path: Path,
    input_run_id: str,
    plot_configuration_sha256: str,
) -> None:
    write_json(
        root / "latest.json",
        {
            "contract_version": CONTRACT_VERSION,
            "run_id": run_id,
            "relative_run_directory": f"runs/{run_id}",
            "release_commit_sha256": sha256(commit_path),
            "input_data_run_id": input_run_id,
            "plot_configuration_sha256": plot_configuration_sha256,
        },
    )


def rendered_fingerprints(context: PlotContext) -> dict[str, str]:
    return {
        path.relative_to(context.figure_dir).as_posix(): sha256(path)
        for path in sorted(context.created_paths)
    }


def render_active_figures(context: PlotContext) -> None:
    active_numbers = {spec.number for spec in context.active_specs}
    for spec, function in zip(FIGURE_SPECS, PLOT_FUNCTIONS):
        if spec.number in active_numbers:
            for text_value in (
                spec.basename,
                spec.title,
                spec.caption,
                *spec.sources,
            ):
                validate_unicode_bytes(
                    text_value.encode("utf-8"),
                    f"figure {spec.number} metadata",
                )
            function(context)


def validate_rendered_figures(context: PlotContext) -> None:
    expected_paths = {
        context.figure_dir
        / figure_section(spec)
        / f"{spec.basename}.{output_format}"
        for spec in context.active_specs
        for output_format in context.formats
    }
    created_paths = set(context.created_paths)
    if created_paths != expected_paths:
        missing = sorted(
            path.relative_to(context.figure_dir).as_posix()
            for path in expected_paths - created_paths
        )
        unexpected = sorted(
            path.relative_to(context.figure_dir).as_posix()
            for path in created_paths - expected_paths
        )
        raise RuntimeError(
            f"Figure set mismatch; missing={missing}, unexpected={unexpected}"
        )

    for path in sorted(expected_paths):
        if path.is_symlink() or not path.exists() or path.stat().st_size < 5_000:
            raise RuntimeError(f"Figure is missing or unexpectedly small: {path}")
        if path.suffix == ".pdf":
            if not path.read_bytes().startswith(b"%PDF"):
                raise RuntimeError(f"Invalid PDF header: {path}")
        elif path.suffix == ".png":
            with Image.open(path) as rendered_image:
                width, height = rendered_image.size
                rendered_image.verify()
            if min(width, height) < 600:
                raise RuntimeError(
                    "PNG dimensions are unexpectedly small for publication: "
                    f"{path} -> {width}x{height}; dpi={context.dpi}"
                )


def output_file_inventory(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(value for value in root.rglob("*") if value.is_file())
    ]


def validate_output_inventory(
    root: Path,
    entries: Sequence[Mapping[str, object]],
) -> None:
    expected = {str(entry["path"]) for entry in entries}
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if observed != expected:
        raise RuntimeError(
            f"figure release file-set mismatch; missing={sorted(expected-observed)}, "
            f"unexpected={sorted(observed-expected)}"
        )
    for entry in entries:
        path = root / str(entry["path"])
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"invalid figure-release file: {entry['path']}")
        if path.stat().st_size != int(entry["size_bytes"]):
            raise RuntimeError(f"figure-release size mismatch: {entry['path']}")
        if sha256(path) != entry["sha256"]:
            raise RuntimeError(f"figure-release checksum mismatch: {entry['path']}")


def write_metadata(
    context: PlotContext,
    run_id: str,
    generated_at: datetime,
    reproducibility_fingerprints: Mapping[str, str],
) -> tuple[str, str]:
    active_numbers = {spec.number for spec in context.active_specs}
    manifest_rows: list[dict[str, object]] = []
    for spec in FIGURE_SPECS:
        published = spec.number in active_numbers
        if published:
            omission_reason = ""
        elif spec.number == 5 and context.input_manifest["daily_samples_written"] is not True:
            omission_reason = "daily samples were intentionally not published"
        elif spec.number not in PRIMARY_FIGURE_NUMBERS:
            omission_reason = "supplement omitted by --primary-only"
        else:
            omission_reason = "not selected"
        manifest_rows.append(
            {
                "figure_number": spec.number,
                "basename": spec.basename,
                "section": figure_section(spec),
                "published": published,
                "omission_reason": omission_reason,
                "title": spec.title,
                "caption": spec.caption,
                "data_sources": "; ".join(spec.sources),
                "formats": "; ".join(context.formats) if published else "",
                "claim_ids": FIGURE_CLAIMS[spec.number],
                "scope": SCOPE_NOTE,
            }
        )
    manifest_path = context.figure_dir / "figure_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(
        manifest_path, index=False, lineterminator="\n"
    )

    caption_lines = [
        "# Analytical workload figure captions",
        "",
        (
            "The primary directory contains the five non-redundant manuscript "
            "figures. Verification and boundary diagnostics are isolated in the "
            "supplement directory. All quantities are analytical or synthetic "
            "under explicit assumptions; none measures an HTGL implementation."
        ),
        "",
    ]
    for spec in context.active_specs:
        caption_lines.extend(
            [
                f"## Figure {spec.number}. {spec.title}",
                "",
                f"Section: {figure_section(spec)}.",
                "",
                spec.caption,
                "",
                f"Data: {', '.join(spec.sources)}.",
                "",
            ]
        )
    captions_path = context.figure_dir / "figure_captions.md"
    caption_payload = "\n".join(caption_lines).encode("utf-8")
    validate_unicode_bytes(caption_payload, captions_path.name)
    atomic_write_bytes(captions_path, caption_payload)

    input_claims = set(
        context.frames["claim_matrix.csv"]["claim_id"].astype(str)
    )
    claim_rows: list[dict[str, object]] = []
    for spec in context.active_specs:
        claim_ids = tuple(filter(None, FIGURE_CLAIMS[spec.number].split(";")))
        missing_claims = sorted(set(claim_ids) - input_claims)
        if missing_claims:
            raise RuntimeError(
                f"figure {spec.number} references missing claims {missing_claims}"
            )
        for source in spec.sources:
            if source not in {
                str(entry["path"]) for entry in context.input_manifest["files"]
            }:
                raise RuntimeError(
                    f"figure {spec.number} references unmanifested input {source}"
                )
        claim_rows.append(
            {
                "figure_number": spec.number,
                "section": figure_section(spec),
                "claim_ids": ";".join(claim_ids),
                "input_data_files": ";".join(spec.sources),
                "claim_matrix_sha256": sha256(
                    context.data_dir / "claim_matrix.csv"
                ),
                "validated": True,
            }
        )
    figure_claim_matrix = pd.DataFrame(claim_rows)
    if figure_claim_matrix.empty or not figure_claim_matrix["validated"].all():
        raise RuntimeError("figure claim matrix is empty or invalid")
    figure_claim_matrix.to_csv(
        context.figure_dir / "figure_claim_matrix.csv",
        index=False,
        lineterminator="\n",
    )

    reproducibility_frame = pd.DataFrame(
        [
            {
                "check": "independent_render_regeneration",
                "passed": True,
                "files_compared": len(reproducibility_fingerprints),
                "combined_sha256": digest_bytes(
                    canonical_json_bytes(dict(sorted(reproducibility_fingerprints.items())))
                ),
                "comparison": "byte-identical independently rendered PDF/PNG files",
            }
        ]
    )
    reproducibility_frame.to_csv(
        context.figure_dir / "render_reproducibility_check.csv",
        index=False,
        lineterminator="\n",
    )
    write_json(
        context.figure_dir / "render_reproducibility_report.json",
        {
            "check": "independent render regeneration",
            "passed": True,
            "files": dict(sorted(reproducibility_fingerprints.items())),
            "combined_sha256": digest_bytes(
                canonical_json_bytes(dict(sorted(reproducibility_fingerprints.items())))
            ),
        },
    )

    environment = environment_snapshot()
    write_json(context.figure_dir / "environment_snapshot.json", environment)
    plot_configuration = {
        "input_data_run_id": context.input_manifest["run_id"],
        "input_configuration_sha256": context.input_manifest["configuration_sha256"],
        "formats": list(context.formats),
        "dpi": context.dpi,
        "active_figure_numbers": [spec.number for spec in context.active_specs],
        "primary_figure_numbers": sorted(PRIMARY_FIGURE_NUMBERS),
    }
    plot_configuration_digest = digest_bytes(canonical_json_bytes(plot_configuration))
    write_json(
        context.figure_dir / "effective_plot_config.json",
        plot_configuration,
    )

    for path in sorted(
        value for value in context.figure_dir.rglob("*") if value.is_file()
    ):
        if path.suffix.lower() in {".csv", ".md", ".json", ".txt"}:
            validate_unicode_bytes(
                path.read_bytes(),
                path.relative_to(context.figure_dir).as_posix(),
            )

    pre_checksum_inventory = output_file_inventory(context.figure_dir)
    checksum_frame = pd.DataFrame(
        [
            {
                "file": entry["path"],
                "sha256": entry["sha256"],
                "size_bytes": entry["size_bytes"],
            }
            for entry in pre_checksum_inventory
        ]
    )
    checksum_frame.to_csv(
        context.figure_dir / "figure_checksums.csv",
        index=False,
        lineterminator="\n",
    )
    final_inventory = output_file_inventory(context.figure_dir)
    gates = {
        "input_release_manifest_and_commit_verified": True,
        "input_checksums_rows_and_schemas_verified": True,
        "input_statistical_tests_recomputed": True,
        "input_claim_matrix_resolved": True,
        "rendered_files_validated": True,
        "independent_render_reproducibility_passed": True,
        "unicode_and_required_glyph_validation_passed": True,
    }
    source_hash = sha256(Path(__file__).resolve())
    release_manifest = {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "generated_at_utc": generated_at.isoformat(),
        "code_version": CODE_VERSION,
        "source": {"file": "plots.py", "sha256": source_hash},
        "source_control": environment["source_control"],
        "command_line": manifest_command_line(),
        "scope": SCOPE_NOTE,
        "plot_configuration_sha256": plot_configuration_digest,
        "input_release": {
            "run_id": context.input_manifest["run_id"],
            "release_manifest_sha256": sha256(
                context.data_dir / "release_manifest.json"
            ),
            "release_commit_sha256": sha256(
                context.data_dir / "RELEASE_COMMIT.json"
            ),
            "configuration_sha256": context.input_manifest[
                "configuration_sha256"
            ],
            "deterministic_release_frame_sha256": context.input_manifest[
                "deterministic_release_frame_sha256"
            ],
        },
        "environment": environment,
        "formats": list(context.formats),
        "dpi": context.dpi,
        "primary_figures": sorted(
            spec.number
            for spec in context.active_specs
            if spec.number in PRIMARY_FIGURE_NUMBERS
        ),
        "supplement_figures": sorted(
            spec.number
            for spec in context.active_specs
            if spec.number not in PRIMARY_FIGURE_NUMBERS
        ),
        "daily_variability_figure_published": 5 in active_numbers,
        "publication_gates": gates,
        "files": final_inventory,
    }
    validate_output_inventory(context.figure_dir, final_inventory)
    write_json(
        context.figure_dir / "figure_release_manifest.json",
        release_manifest,
    )
    manifest_hash = sha256(
        context.figure_dir / "figure_release_manifest.json"
    )
    commit = {
        "complete": True,
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "figure_release_manifest_sha256": manifest_hash,
        "plot_configuration_sha256": plot_configuration_digest,
        "input_data_run_id": context.input_manifest["run_id"],
        "publication_gates": gates,
    }
    write_json(context.figure_dir / "RELEASE_COMMIT.json", commit)
    if sha256(context.figure_dir / "figure_release_manifest.json") != commit[
        "figure_release_manifest_sha256"
    ]:
        raise RuntimeError("figure release commit does not bind its manifest")
    return plot_configuration_digest, manifest_hash


def remove_legacy_flat_outputs(root: Path) -> None:
    names = {
        *(f"{spec.basename}.{extension}" for spec in FIGURE_SPECS for extension in ("pdf", "png")),
        "figure_manifest.csv",
        "figure_captions.md",
        "figure_checksums.csv",
        "plot_run_manifest.json",
    }
    for name in sorted(names):
        path = root / name
        if path.is_symlink():
            raise RuntimeError(f"refusing to delete symbolic-link legacy output: {path}")
        if path.is_file():
            path.unlink()


def run_self_tests() -> None:
    configure_style()
    if len(PRIMARY_FIGURE_NUMBERS) != 5:
        raise RuntimeError("self-test failed: primary figure set must contain five figures")
    if not PRIMARY_FIGURE_NUMBERS <= set(SPEC_BY_NUMBER):
        raise RuntimeError("self-test failed: primary figure selection is invalid")
    validate_unicode_bytes("158.5×; budgets only—no benchmark\n".encode("utf-8"), "unicode")
    fonts = font_fingerprints()
    validate_required_glyphs(fonts)
    corrupted = bytes([0xC3, 0x83, 0xC2, 0x97]).decode("utf-8")
    try:
        validate_unicode_bytes(corrupted.encode("utf-8"), "corruption-test")
    except RuntimeError:
        pass
    else:
        raise RuntimeError("self-test failed: mojibake marker was accepted")
    dtype_probe = pd.DataFrame(
        {
            "legacy_object": pd.Series(["alpha", None], dtype=object),
            "extension_string": pd.Series(["beta", pd.NA], dtype="string"),
            "numeric": pd.Series([1, 2], dtype="int64"),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        selected_columns = text_columns(dtype_probe)
    if selected_columns != ["legacy_object", "extension_string"]:
        raise RuntimeError(
            "self-test failed: text dtype selection is incomplete or overbroad"
        )
    with tempfile.TemporaryDirectory(prefix="plots-v8-self-test-") as temporary:
        root = Path(temporary)
        write_json(
            root / "latest.json",
            {
                "contract_version": CONTRACT_VERSION,
                "run_id": "missing-run",
                "relative_run_directory": "runs/missing-run",
                "release_commit_sha256": "0" * 64,
                "configuration_sha256": "0" * 64,
            },
        )
        try:
            ReleaseContractReader(root).read()
        except RuntimeError as exc:
            if "latest.json is broken" not in str(exc):
                raise RuntimeError(
                    "self-test failed: broken pointer diagnostic is not actionable"
                ) from exc
        else:
            raise RuntimeError("self-test failed: broken latest pointer was accepted")
        (root / "runs" / "missing-run").mkdir(parents=True)
        try:
            ReleaseContractReader(root).read()
        except RuntimeError as exc:
            if "is incomplete; missing" not in str(exc):
                raise RuntimeError(
                    "self-test failed: incomplete run diagnostic is not actionable"
                ) from exc
        else:
            raise RuntimeError("self-test failed: incomplete run directory was accepted")
    print("All plots.py self-tests passed.")


def execute(args: argparse.Namespace) -> None:
    configure_style()
    release, frames = ReleaseContractReader(args.data_dir).read()
    figure_root = args.figure_dir.expanduser().resolve()
    first_stage = create_staging_directory(figure_root)
    second_stage = create_staging_directory(figure_root)
    destination: Path | None = None
    try:
        first = build_context(args, first_stage, release, frames)
        second = replace(
            first,
            figure_dir=second_stage,
            created_paths=[],
        )
        environment = environment_snapshot()
        validate_required_glyphs(environment["fonts"])
        source_bytes = Path(__file__).resolve().read_bytes()
        validate_unicode_bytes(source_bytes, "plots.py")

        print(f"Running {CODE_VERSION}")
        print(f"Scope: {SCOPE_NOTE}")
        print(f"Committed analytical input: {first.data_dir}")
        print(f"Figure publication root: {figure_root}")
        print(
            "Populations: "
            + ", ".join(
                population_label(population) for population in first.populations
            )
        )
        print(
            "Primary figures: "
            + ", ".join(
                str(spec.number)
                for spec in first.active_specs
                if spec.number in PRIMARY_FIGURE_NUMBERS
            )
        )

        render_active_figures(first)
        validate_rendered_figures(first)
        render_active_figures(second)
        validate_rendered_figures(second)
        first_fingerprints = rendered_fingerprints(first)
        second_fingerprints = rendered_fingerprints(second)
        if first_fingerprints != second_fingerprints:
            differing = sorted(
                {
                    *(
                        name
                        for name in first_fingerprints
                        if first_fingerprints.get(name)
                        != second_fingerprints.get(name)
                    ),
                    *(
                        name
                        for name in second_fingerprints
                        if first_fingerprints.get(name)
                        != second_fingerprints.get(name)
                    ),
                }
            )
            raise RuntimeError(
                "independent render reproducibility failed for: "
                + ", ".join(differing)
            )
        shutil.rmtree(second_stage)
        release.assert_unchanged()

        generated_at = datetime.now(timezone.utc)
        source_hash = sha256(Path(__file__).resolve())
        run_id = (
            "plot-"
            + generated_at.strftime("%Y%m%dT%H%M%S%fZ")
            + f"-{str(first.input_manifest['run_id'])[-8:]}-{source_hash[:10]}-{uuid.uuid4().hex[:8]}"
        )
        plot_config_digest, _ = write_metadata(
            first,
            run_id,
            generated_at,
            first_fingerprints,
        )

        with publication_lock(figure_root):
            if args.clean:
                remove_legacy_flat_outputs(figure_root)
            destination = publish_immutable_run(first_stage, figure_root, run_id)
            write_latest_pointer(
                figure_root,
                run_id,
                destination / "RELEASE_COMMIT.json",
                str(first.input_manifest["run_id"]),
                plot_config_digest,
            )
    except BaseException:
        if first_stage.exists():
            shutil.rmtree(first_stage)
        if second_stage.exists():
            shutil.rmtree(second_stage)
        raise

    if destination is None:
        raise RuntimeError("figure release was not published")
    active_count = len(first.active_specs)
    print("\nAnalytical workload figures published successfully.")
    print(f"Figure run ID: {destination.name}")
    print(f"Figure release: {destination}")
    print(f"Figures: {active_count}")
    print(f"Rendered files: {len(first.created_paths)}")
    print(f"Formats: {', '.join(first.formats)}")
    if first.input_manifest["daily_samples_written"] is not True:
        print("Figure 5 skipped because daily samples were not published.")
    print("Every input, render, reproducibility, Unicode, manifest, and claim gate passed.")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.self_test:
        run_self_tests()
        return
    execute(args)


if __name__ == "__main__":
    main()
