#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analytical workload characterization for the proposed HTGL architecture.

This program characterizes the *offered coordination workload* implied by the
HTGL design.  It is deliberately not an HTGL implementation or a blockchain
benchmark.  In particular, it does not model consensus execution, validator
hardware, network propagation, transaction latency/finality, cryptographic
cost, Proof-after-Erasure, or legal compliance.

The study has two complementary layers:

1. Closed-form analysis of expected daily event counts and average offered TPS.
2. Reproducible component-wise Poisson sampling used only to verify the
   analytical calculations and characterize day-to-day count variation.

Default experiment matrix
-------------------------
* Consumer populations: 50 million, 150 million, and 450 million.
* Workload scenarios: baseline, high authorization, high DER adoption and
  metadata activity, and a combined high-activity scenario.
* Repeated study: 30 independent replications x 365 daily observations for
  every population-scenario pair (10,950 observations per pair).

All scenario inputs are explicit assumptions.  Synthetic temporal profiles and
reference workload envelopes are analytical what-if constructs, not measured
traces or measured platform capacities.

Dependencies: Python 3.10+, NumPy, pandas, and SciPy.

Example
-------
    python simUp.py
    python simUp.py --replications 50 --days-per-replication 365
    python simUp.py --output-dir ./results --artifact-dir ./paper_artifacts
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
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2


CODE_VERSION = "simUpV7-production-analytical-workload-characterization-2026-07-22"
CONTRACT_VERSION = "htgl-analytical-release-contract-v1"
DEFAULT_SEED = 20260722
DEFAULT_REPLICATIONS = 30
DEFAULT_DAYS_PER_REPLICATION = 365
POPULATIONS = (50_000_000, 150_000_000, 450_000_000)
DEFAULT_FAMILYWISE_ALPHA = 0.01

SECONDS_PER_DAY = 86_400
HOURS_PER_DAY = 24
SECONDS_PER_HOUR = 3_600
DAYS_PER_YEAR = 365.0
BASE_DIR = Path(__file__).resolve().parent

NORMAL = NormalDist()
Z_975 = NORMAL.inv_cdf(0.975)

COMPONENT_COLUMNS = (
    "consent_events_day",
    "authorization_events_day",
    "der_metadata_events_day",
    "integrity_events_day",
)

SCOPE_NOTE = (
    "Analytical offered-workload characterization only; not an HTGL "
    "implementation or platform-performance measurement."
)


@dataclass(frozen=True)
class ModelParameters:
    """Baseline architectural workload assumptions.

    Rates are deliberately kept in their natural units to avoid the previous
    consent-rate ambiguity: consent is annual and divided by 365 exactly once.
    Institutional actors are held fixed in the population experiment so that
    changing N_c isolates consumer-population scaling.  Actor-count sensitivity
    is evaluated separately.
    """

    institutional_actors: int = 500
    consent_changes_per_consumer_year: float = 2.0
    authorization_events_per_consumer_day: float = 0.5
    der_adoption_fraction: float = 0.10
    metadata_events_per_der_day: float = 1.0
    integrity_events_per_actor_day: float = 24.0
    meter_readings_per_consumer_day: float = 96.0
    consent_record_bytes_nominal: int = 320
    authorization_record_bytes_nominal: int = 256
    der_metadata_record_bytes_nominal: int = 384
    integrity_record_bytes_nominal: int = 128
    consent_record_bytes_conservative: int = 512
    authorization_record_bytes_conservative: int = 448
    der_metadata_record_bytes_conservative: int = 768
    integrity_record_bytes_conservative: int = 256

    def validate(self) -> None:
        if (
            isinstance(self.institutional_actors, bool)
            or not isinstance(self.institutional_actors, int)
            or self.institutional_actors <= 0
        ):
            raise ValueError("institutional_actors must be a positive integer")
        if (
            isinstance(self.der_adoption_fraction, bool)
            or not isinstance(self.der_adoption_fraction, (int, float))
            or not math.isfinite(float(self.der_adoption_fraction))
            or not 0.0 <= float(self.der_adoption_fraction) <= 1.0
        ):
            raise ValueError("der_adoption_fraction must be in [0, 1]")
        non_negative = {
            "consent_changes_per_consumer_year": self.consent_changes_per_consumer_year,
            "authorization_events_per_consumer_day": self.authorization_events_per_consumer_day,
            "metadata_events_per_der_day": self.metadata_events_per_der_day,
            "integrity_events_per_actor_day": self.integrity_events_per_actor_day,
            "meter_readings_per_consumer_day": self.meter_readings_per_consumer_day,
        }
        for name, value in non_negative.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite non-negative number")
        record_sizes = {
            name: value
            for name, value in asdict(self).items()
            if name.endswith("_record_bytes_nominal")
            or name.endswith("_record_bytes_conservative")
        }
        for name, value in record_sizes.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class WorkloadScenario:
    """Multipliers defining an analytical workload scenario."""

    name: str
    label: str
    consent_multiplier: float = 1.0
    authorization_multiplier: float = 1.0
    metadata_multiplier: float = 1.0
    integrity_multiplier: float = 1.0
    der_adoption_fraction: float = 0.10
    description: str = ""

    def validate(self) -> None:
        multipliers = (
            self.consent_multiplier,
            self.authorization_multiplier,
            self.metadata_multiplier,
            self.integrity_multiplier,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
            for value in multipliers
        ):
            raise ValueError(
                f"Scenario {self.name!r} has an invalid non-negative multiplier"
            )
        if (
            isinstance(self.der_adoption_fraction, bool)
            or not isinstance(self.der_adoption_fraction, (int, float))
            or not math.isfinite(float(self.der_adoption_fraction))
            or not 0.0 <= float(self.der_adoption_fraction) <= 1.0
        ):
            raise ValueError(
                f"Scenario {self.name!r} has DER adoption outside [0, 1]"
            )


SCENARIOS = (
    WorkloadScenario(
        name="baseline",
        label="Baseline",
        der_adoption_fraction=0.10,
        description="Reference HTGL coordination-event assumptions.",
    ),
    WorkloadScenario(
        name="high_authorization",
        label="High authorization",
        authorization_multiplier=2.0,
        description="Authorization-event frequency is twice the baseline assumption.",
    ),
    WorkloadScenario(
        name="high_der_adoption_metadata_activity",
        label="High DER adoption and metadata activity",
        metadata_multiplier=2.0,
        der_adoption_fraction=0.25,
        description=(
            "DER adoption is 25% and per-DER metadata-event frequency is twice "
            "the baseline assumption."
        ),
    ),
    WorkloadScenario(
        name="combined_high_activity",
        label="Combined high activity",
        authorization_multiplier=2.0,
        metadata_multiplier=2.0,
        der_adoption_fraction=0.25,
        description=(
            "High-authorization and high-DER-adoption/metadata-activity assumptions are applied "
            "together; this is a synthetic analytical scenario."
        ),
    ),
)


@dataclass(frozen=True)
class TemporalProfile:
    """Synthetic within-day concentration profile with mean weight equal to 1."""

    name: str
    label: str
    amplitude: float
    peak_hour: float = 18.0
    description: str = ""


TEMPORAL_PROFILES = (
    TemporalProfile(
        name="uniform",
        label="Uniform",
        amplitude=0.0,
        description="Expected coordination events are uniform across the day.",
    ),
    TemporalProfile(
        name="diurnal_30pct",
        label="Moderate concentration",
        amplitude=0.30,
        description=(
            "Synthetic sinusoidal profile with a 30% peak-above-mean amplitude."
        ),
    ),
    TemporalProfile(
        name="diurnal_60pct",
        label="High concentration",
        amplitude=0.60,
        description=(
            "Synthetic sinusoidal profile with a 60% peak-above-mean amplitude."
        ),
    ),
)


ALLOWED_EVIDENCE_CLASSES = {
    "explicit_hypothesis",
    "design_requirement",
    "empirical_trace",
    "literature",
    "policy_input",
}


def _hypothesis(rationale: str) -> dict[str, str | None]:
    return {
        "evidence_class": "explicit_hypothesis",
        "citation": None,
        "rationale": rationale,
    }


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "populations": list(POPULATIONS),
    "replications": DEFAULT_REPLICATIONS,
    "days_per_replication": DEFAULT_DAYS_PER_REPLICATION,
    "seed": DEFAULT_SEED,
    "familywise_alpha": DEFAULT_FAMILYWISE_ALPHA,
    "model_parameters": asdict(ModelParameters()),
    "parameter_evidence": {
        "consumer_populations": _hypothesis(
            "Required national-scale analytical population cases; not observed deployment sizes."
        ),
        "institutional_actors": _hypothesis(
            "Fixed actor-count case chosen to isolate consumer-population scaling."
        ),
        "consent_changes_per_consumer_year": _hypothesis(
            "Scenario input for annual consent-state change frequency; not an empirical estimate."
        ),
        "authorization_events_per_consumer_day": _hypothesis(
            "Scenario input whose dominance is explicitly assessed by sensitivity analysis."
        ),
        "der_adoption_fraction": _hypothesis(
            "Baseline DER-adoption scenario input; not a forecast."
        ),
        "metadata_events_per_der_day": _hypothesis(
            "Scenario input for DER metadata activity; not a measured trace rate."
        ),
        "integrity_events_per_actor_day": _hypothesis(
            "Scenario input for integrity-anchor activity; not a measured implementation rate."
        ),
        "meter_readings_per_consumer_day": _hypothesis(
            "Event-count boundary input used only for a dimensional comparison."
        ),
        "consent_record_bytes_nominal": _hypothesis(
            "Component-specific logical-payload budget for sensitivity; excludes protocol overhead."
        ),
        "authorization_record_bytes_nominal": _hypothesis(
            "Component-specific logical-payload budget for sensitivity; excludes protocol overhead."
        ),
        "der_metadata_record_bytes_nominal": _hypothesis(
            "Component-specific logical-payload budget for sensitivity; excludes protocol overhead."
        ),
        "integrity_record_bytes_nominal": _hypothesis(
            "Component-specific logical-payload budget for sensitivity; excludes protocol overhead."
        ),
        "consent_record_bytes_conservative": _hypothesis(
            "Upper logical-payload budget for the consent component; not a serialized measurement."
        ),
        "authorization_record_bytes_conservative": _hypothesis(
            "Upper logical-payload budget for authorization; not a serialized measurement."
        ),
        "der_metadata_record_bytes_conservative": _hypothesis(
            "Upper logical-payload budget for DER metadata; not a serialized measurement."
        ),
        "integrity_record_bytes_conservative": _hypothesis(
            "Upper logical-payload budget for integrity records; not a serialized measurement."
        ),
    },
    "scenarios": [asdict(value) for value in SCENARIOS],
    "scenario_evidence": {
        value.name: _hypothesis(value.description) for value in SCENARIOS
    },
    "temporal_profiles": [asdict(value) for value in TEMPORAL_PROFILES],
    "temporal_profile_evidence": {
        value.name: _hypothesis(value.description) for value in TEMPORAL_PROFILES
    },
    "sensitivity_ranges": {
        "consent_changes_per_consumer_year": [0.5, 1.0, 2.0, 4.0, 8.0],
        "authorization_events_per_consumer_day": [0.10, 0.25, 0.50, 1.0, 2.0],
        "der_adoption_fraction": [0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
        "metadata_events_per_der_day": [0.25, 0.50, 1.0, 2.0, 4.0],
        "integrity_events_per_actor_day": [1.0, 6.0, 12.0, 24.0, 48.0, 96.0],
        "institutional_actors": [100, 250, 500, 1000, 2000],
    },
    "authorization_grid": [float(value) for value in np.linspace(0.10, 2.00, 10)],
    "der_adoption_grid": [float(value) for value in np.linspace(0.05, 0.50, 10)],
    "population_der_grid": [
        int(value) for value in np.linspace(10_000_000, 450_000_000, 12, dtype=int)
    ],
    "storage_horizons_days": [1, 30, 365],
    "reference_budgets_tps": [500, 1000, 2000, 5000, 10000],
}


TOP_LEVEL_CONFIG_KEYS = frozenset(DEFAULT_CONFIG)
MODEL_PARAMETER_KEYS = frozenset(asdict(ModelParameters()))
EVIDENCE_KEYS = frozenset({"evidence_class", "citation", "rationale"})
SCENARIO_KEYS = frozenset(asdict(SCENARIOS[0]))
TEMPORAL_PROFILE_KEYS = frozenset(asdict(TEMPORAL_PROFILES[0]))


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_keys(value: Mapping[str, object], expected: frozenset[str], path: str) -> None:
    observed = frozenset(value)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing or unknown:
        raise ValueError(
            f"{path} schema mismatch; missing={missing or 'none'}, "
            f"unknown={unknown or 'none'}"
        )


def _finite_number(value: object, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{path} must be >= {minimum}")
    return result


def _positive_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _validate_evidence(value: object, path: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    _strict_keys(value, EVIDENCE_KEYS, path)
    evidence_class = value["evidence_class"]
    if evidence_class not in ALLOWED_EVIDENCE_CLASSES:
        raise ValueError(
            f"{path}.evidence_class must be one of {sorted(ALLOWED_EVIDENCE_CLASSES)}"
        )
    rationale = value["rationale"]
    if not isinstance(rationale, str) or len(rationale.strip()) < 20:
        raise ValueError(f"{path}.rationale must contain at least 20 non-blank characters")
    citation = value["citation"]
    if citation is not None and (not isinstance(citation, str) or not citation.strip()):
        raise ValueError(f"{path}.citation must be null or a non-empty string")
    if evidence_class != "explicit_hypothesis" and citation is None:
        raise ValueError(f"{path}.citation is required for {evidence_class!r}")


def validate_config(config: Mapping[str, object]) -> None:
    _strict_keys(config, TOP_LEVEL_CONFIG_KEYS, "config")
    if config["schema_version"] != 1:
        raise ValueError("config.schema_version must equal 1")

    populations = config["populations"]
    if not isinstance(populations, list) or len(populations) < 1:
        raise ValueError("config.populations must be a non-empty list")
    population_values = [_positive_int(value, "config.populations[]") for value in populations]
    if population_values != sorted(set(population_values)):
        raise ValueError("config.populations must contain unique values in increasing order")
    required = list(POPULATIONS)
    if population_values != required:
        raise ValueError(f"config.populations must equal {required}")

    _positive_int(config["replications"], "config.replications")
    if _positive_int(config["days_per_replication"], "config.days_per_replication") <= 1:
        raise ValueError("config.days_per_replication must be greater than one")
    seed = config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("config.seed must be a non-negative integer")
    alpha = _finite_number(config["familywise_alpha"], "config.familywise_alpha")
    if not 0.0 < alpha < 0.10:
        raise ValueError("config.familywise_alpha must be in (0, 0.10)")

    model = config["model_parameters"]
    if not isinstance(model, dict):
        raise ValueError("config.model_parameters must be an object")
    _strict_keys(model, MODEL_PARAMETER_KEYS, "config.model_parameters")
    try:
        parameters = ModelParameters(**model)
    except TypeError as exc:
        raise ValueError(f"invalid model_parameters: {exc}") from exc
    parameters.validate()

    parameter_evidence = config["parameter_evidence"]
    if not isinstance(parameter_evidence, dict):
        raise ValueError("config.parameter_evidence must be an object")
    expected_evidence = frozenset({"consumer_populations", *MODEL_PARAMETER_KEYS})
    _strict_keys(parameter_evidence, expected_evidence, "config.parameter_evidence")
    for name, evidence in parameter_evidence.items():
        _validate_evidence(evidence, f"config.parameter_evidence.{name}")

    scenarios = config["scenarios"]
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("config.scenarios must be a non-empty list")
    scenario_names: list[str] = []
    for index, value in enumerate(scenarios):
        if not isinstance(value, dict):
            raise ValueError(f"config.scenarios[{index}] must be an object")
        _strict_keys(value, SCENARIO_KEYS, f"config.scenarios[{index}]")
        scenario = WorkloadScenario(**value)
        scenario.validate()
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", scenario.name):
            raise ValueError(f"invalid scenario name {scenario.name!r}")
        if not scenario.label.strip() or not scenario.description.strip():
            raise ValueError(f"scenario {scenario.name!r} requires label and description")
        scenario_names.append(scenario.name)
    if len(scenario_names) != len(set(scenario_names)) or scenario_names[0] != "baseline":
        raise ValueError("scenario names must be unique and baseline must be first")
    baseline_adoption = float(scenarios[0]["der_adoption_fraction"])
    if not math.isclose(
        baseline_adoption,
        float(model["der_adoption_fraction"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(
            "baseline scenario der_adoption_fraction must equal the baseline model parameter"
        )
    scenario_evidence = config["scenario_evidence"]
    if not isinstance(scenario_evidence, dict):
        raise ValueError("config.scenario_evidence must be an object")
    _strict_keys(scenario_evidence, frozenset(scenario_names), "config.scenario_evidence")
    for name, evidence in scenario_evidence.items():
        _validate_evidence(evidence, f"config.scenario_evidence.{name}")

    profiles = config["temporal_profiles"]
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("config.temporal_profiles must be a non-empty list")
    profile_names: list[str] = []
    for index, value in enumerate(profiles):
        if not isinstance(value, dict):
            raise ValueError(f"config.temporal_profiles[{index}] must be an object")
        _strict_keys(value, TEMPORAL_PROFILE_KEYS, f"config.temporal_profiles[{index}]")
        profile = TemporalProfile(**value)
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", profile.name):
            raise ValueError(f"invalid temporal-profile name {profile.name!r}")
        if not 0.0 <= _finite_number(profile.amplitude, "profile.amplitude") < 1.0:
            raise ValueError("temporal profile amplitude must be in [0, 1)")
        if not 0.0 <= _finite_number(profile.peak_hour, "profile.peak_hour") < 24.0:
            raise ValueError("temporal profile peak_hour must be in [0, 24)")
        if not profile.label.strip() or not profile.description.strip():
            raise ValueError(f"temporal profile {profile.name!r} requires label and description")
        profile_names.append(profile.name)
    if len(profile_names) != len(set(profile_names)):
        raise ValueError("temporal profile names must be unique")
    temporal_evidence = config["temporal_profile_evidence"]
    if not isinstance(temporal_evidence, dict):
        raise ValueError("config.temporal_profile_evidence must be an object")
    _strict_keys(
        temporal_evidence,
        frozenset(profile_names),
        "config.temporal_profile_evidence",
    )
    for name, evidence in temporal_evidence.items():
        _validate_evidence(evidence, f"config.temporal_profile_evidence.{name}")

    ranges = config["sensitivity_ranges"]
    if not isinstance(ranges, dict):
        raise ValueError("config.sensitivity_ranges must be an object")
    allowed_ranges = frozenset(
        {
            "consent_changes_per_consumer_year",
            "authorization_events_per_consumer_day",
            "der_adoption_fraction",
            "metadata_events_per_der_day",
            "integrity_events_per_actor_day",
            "institutional_actors",
        }
    )
    _strict_keys(ranges, allowed_ranges, "config.sensitivity_ranges")
    for name, values in ranges.items():
        if not isinstance(values, list) or len(values) < 2:
            raise ValueError(f"config.sensitivity_ranges.{name} needs at least two values")
        numeric = [_finite_number(value, f"config.sensitivity_ranges.{name}[]", minimum=0) for value in values]
        if numeric != sorted(set(numeric)):
            raise ValueError(f"config.sensitivity_ranges.{name} must be unique and increasing")
        if name == "der_adoption_fraction" and any(value > 1.0 for value in numeric):
            raise ValueError("DER-adoption sensitivity values must not exceed 1")
        if name == "institutional_actors" and any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise ValueError("institutional-actor sensitivity values must be integers")

    for key in ("authorization_grid", "der_adoption_grid"):
        values = config[key]
        if not isinstance(values, list) or len(values) < 2:
            raise ValueError(f"config.{key} needs at least two values")
        numeric = [_finite_number(value, f"config.{key}[]", minimum=0) for value in values]
        if numeric != sorted(set(numeric)):
            raise ValueError(f"config.{key} must be unique and increasing")
    if any(float(value) > 1.0 for value in config["der_adoption_grid"]):
        raise ValueError("config.der_adoption_grid values must not exceed 1")

    population_grid = config["population_der_grid"]
    if not isinstance(population_grid, list) or len(population_grid) < 2:
        raise ValueError("config.population_der_grid needs at least two populations")
    population_grid_values = [_positive_int(value, "config.population_der_grid[]") for value in population_grid]
    if population_grid_values != sorted(set(population_grid_values)):
        raise ValueError("config.population_der_grid must be unique and increasing")

    for key in ("storage_horizons_days", "reference_budgets_tps"):
        values = config[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"config.{key} must be a non-empty list")
        checked = [_positive_int(value, f"config.{key}[]") for value in values]
        if checked != sorted(set(checked)):
            raise ValueError(f"config.{key} must be unique and increasing")


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        config = json.loads(json.dumps(DEFAULT_CONFIG))
    else:
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load experiment configuration {path}: {exc}") from exc
        if not isinstance(config, dict):
            raise ValueError("experiment configuration root must be an object")
    validate_config(config)
    return config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analytically characterize HTGL coordination workload."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Strict JSON experiment configuration. When omitted, the fully "
            "validated built-in configuration is used and published verbatim."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "htgl_analytical_data",
        help="Directory for CSV/JSON outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=BASE_DIR / "analytical_artifacts",
        help="Directory for manuscript-ready tables and notes (default: %(default)s).",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=None,
        help="Override config replications with a positive integer.",
    )
    parser.add_argument(
        "--days-per-replication",
        type=int,
        default=None,
        help="Override config days per replication with an integer greater than one.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override config master seed with a non-negative integer.",
    )
    parser.add_argument(
        "--no-daily-samples",
        action="store_true",
        help="Do not write the pooled daily sample file; summaries are still written.",
    )
    parser.add_argument(
        "--write-default-config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write the complete validated default JSON configuration and exit.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic unit and release-contract tests and exit.",
    )
    args = parser.parse_args(argv)
    if args.replications is not None and args.replications <= 0:
        parser.error("--replications must be positive")
    if args.days_per_replication is not None and args.days_per_replication <= 1:
        parser.error("--days-per-replication must be greater than one")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be non-negative")
    return args


def effective_parameters(
    parameters: ModelParameters,
    scenario: WorkloadScenario,
) -> dict[str, float]:
    """Return scenario-adjusted parameters without changing baseline inputs."""

    return {
        "consent_changes_per_consumer_year": (
            parameters.consent_changes_per_consumer_year
            * scenario.consent_multiplier
        ),
        "authorization_events_per_consumer_day": (
            parameters.authorization_events_per_consumer_day
            * scenario.authorization_multiplier
        ),
        "der_adoption_fraction": scenario.der_adoption_fraction,
        "metadata_events_per_der_day": (
            parameters.metadata_events_per_der_day * scenario.metadata_multiplier
        ),
        "integrity_events_per_actor_day": (
            parameters.integrity_events_per_actor_day
            * scenario.integrity_multiplier
        ),
    }


def expected_component_rates(
    population: int,
    parameters: ModelParameters,
    scenario: WorkloadScenario,
) -> dict[str, float]:
    """Compute the closed-form expected HTGL coordination events per day.

    Let N_c be consumers, N_a institutional actors, alpha_d DER adoption,
    lambda_c consent changes per consumer per year, and the remaining lambdas
    daily rates.  Then:

        R_c = N_c * lambda_c / 365
        R_a = N_c * lambda_a
        R_m = N_c * alpha_d * lambda_m
        R_i = N_a * lambda_i
        W   = R_c + R_a + R_m + R_i
    """

    if population <= 0:
        raise ValueError("population must be positive")
    effective = effective_parameters(parameters, scenario)
    consent = (
        population
        * effective["consent_changes_per_consumer_year"]
        / DAYS_PER_YEAR
    )
    authorization = (
        population * effective["authorization_events_per_consumer_day"]
    )
    der_metadata = (
        population
        * effective["der_adoption_fraction"]
        * effective["metadata_events_per_der_day"]
    )
    integrity = (
        parameters.institutional_actors
        * effective["integrity_events_per_actor_day"]
    )
    rates = {
        "consent_events_day": float(consent),
        "authorization_events_day": float(authorization),
        "der_metadata_events_day": float(der_metadata),
        "integrity_events_day": float(integrity),
    }
    rates["coordination_events_day"] = float(sum(rates.values()))
    return rates


def poisson_normal_quantile(mean_events: float, probability: float) -> float:
    """Large-lambda normal approximation to a Poisson event-count quantile."""

    if mean_events < 0:
        raise ValueError("Poisson mean cannot be negative")
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be in (0, 1)")
    return max(0.0, mean_events + NORMAL.inv_cdf(probability) * math.sqrt(mean_events))


def analytical_characterization(
    populations: Iterable[int],
    scenarios: Sequence[WorkloadScenario],
    parameters: ModelParameters,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for population in populations:
        for scenario in scenarios:
            rates = expected_component_rates(population, parameters, scenario)
            total = rates["coordination_events_day"]
            effective = effective_parameters(parameters, scenario)
            row: dict[str, object] = {
                "population": population,
                "population_millions": population / 1_000_000,
                "scenario": scenario.name,
                "scenario_label": scenario.label,
                "institutional_actors": parameters.institutional_actors,
                **effective,
                **rates,
                "expected_tps": total / SECONDS_PER_DAY,
                "poisson_variance_events_day": total,
                "poisson_sd_events_day": math.sqrt(total),
                "daily_count_sd_equivalent_tps": math.sqrt(total) / SECONDS_PER_DAY,
                "daily_count_normal_pi95_lower_equivalent_tps": poisson_normal_quantile(total, 0.025)
                / SECONDS_PER_DAY,
                "daily_count_normal_p50_equivalent_tps": poisson_normal_quantile(total, 0.50)
                / SECONDS_PER_DAY,
                "daily_count_normal_p95_equivalent_tps": poisson_normal_quantile(total, 0.95)
                / SECONDS_PER_DAY,
                "daily_count_normal_pi95_upper_equivalent_tps": poisson_normal_quantile(total, 0.975)
                / SECONDS_PER_DAY,
                "daily_count_normal_p99_equivalent_tps": poisson_normal_quantile(total, 0.99)
                / SECONDS_PER_DAY,
                "events_per_consumer_day": total / population,
                "quantile_approximation": (
                    "normal approximation to a synthetic daily Poisson count; "
                    "division by 86400 yields an equivalent daily-average rate, "
                    "not instantaneous throughput"
                ),
                "scope": SCOPE_NOTE,
            }
            for component in COMPONENT_COLUMNS:
                row[f"{component}_share_pct"] = 100.0 * rates[component] / total
            rows.append(row)
    return pd.DataFrame(rows)


def component_share_table(analytical: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "consent_events_day": "Consent-state changes",
        "authorization_events_day": "Authorization decisions",
        "der_metadata_events_day": "DER metadata changes",
        "integrity_events_day": "Integrity/audit anchors",
    }
    rows: list[dict[str, object]] = []
    for record in analytical.to_dict(orient="records"):
        total = float(record["coordination_events_day"])
        for component in COMPONENT_COLUMNS:
            events = float(record[component])
            rows.append(
                {
                    "population": record["population"],
                    "population_millions": record["population_millions"],
                    "scenario": record["scenario"],
                    "component": component,
                    "component_label": labels[component],
                    "expected_events_day": events,
                    "expected_tps": events / SECONDS_PER_DAY,
                    "share_pct": 100.0 * events / total,
                }
            )
    return pd.DataFrame(rows)


def generate_daily_samples(
    populations: Sequence[int],
    scenarios: Sequence[WorkloadScenario],
    parameters: ModelParameters,
    replications: int,
    days_per_replication: int,
    master_seed: int,
) -> pd.DataFrame:
    """Generate independent daily counts for stochastic model verification.

    Each component is sampled as an independent homogeneous Poisson daily
    count.  These samples do not represent ledger execution or instantaneous
    transaction arrival times.
    """

    frames: list[pd.DataFrame] = []
    for population_index, population in enumerate(populations):
        for scenario_index, scenario in enumerate(scenarios):
            rates = expected_component_rates(population, parameters, scenario)
            component_means = np.asarray(
                [rates[column] for column in COMPONENT_COLUMNS], dtype=float
            )
            for replication in range(1, replications + 1):
                # SeedSequence makes every case/replication an independent,
                # reproducible stream and prevents loop-order-dependent results.
                seed_sequence = np.random.SeedSequence(
                    [master_seed, population_index, scenario_index, replication]
                )
                rng = np.random.default_rng(seed_sequence)
                counts = np.column_stack(
                    [
                        rng.poisson(mean, size=days_per_replication)
                        for mean in component_means
                    ]
                )
                frame = pd.DataFrame(counts, columns=COMPONENT_COLUMNS)
                frame.insert(0, "day", np.arange(1, days_per_replication + 1))
                frame.insert(0, "replication", replication)
                frame.insert(0, "scenario", scenario.name)
                frame.insert(0, "population_millions", population / 1_000_000)
                frame.insert(0, "population", population)
                frame["coordination_events_day"] = frame[
                    list(COMPONENT_COLUMNS)
                ].sum(axis=1)
                frame["daily_count_equivalent_tps"] = (
                    frame["coordination_events_day"] / SECONDS_PER_DAY
                )
                frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def sample_statistics(values: pd.Series, prefix: str) -> dict[str, float]:
    values = pd.Series(values, dtype=float)
    count = int(values.size)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    mean_half_width = Z_975 * sd / math.sqrt(count)
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_sd": sd,
        f"{prefix}_mean_ci95_lower": mean - mean_half_width,
        f"{prefix}_mean_ci95_upper": mean + mean_half_width,
        f"{prefix}_pi95_lower": float(values.quantile(0.025)),
        f"{prefix}_p50": float(values.quantile(0.50)),
        f"{prefix}_p95": float(values.quantile(0.95)),
        f"{prefix}_pi95_upper": float(values.quantile(0.975)),
        f"{prefix}_p99": float(values.quantile(0.99)),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
    }


def summarize_stochastic_samples(
    samples: pd.DataFrame,
    analytical: pd.DataFrame,
    replications: int,
    days_per_replication: int,
) -> pd.DataFrame:
    expected_lookup = analytical.set_index(["population", "scenario"])[
        "coordination_events_day"
    ]
    rows: list[dict[str, object]] = []
    grouped = samples.groupby(["population", "population_millions", "scenario"], sort=True)
    for (population, population_millions, scenario), group in grouped:
        expected = float(expected_lookup.loc[(population, scenario)])
        event_stats = sample_statistics(group["coordination_events_day"], "events")
        tps_stats = sample_statistics(
            group["daily_count_equivalent_tps"], "daily_count_equivalent_tps"
        )
        variance = float(group["coordination_events_day"].var(ddof=1))
        observations = len(group)
        mean_standard_error = math.sqrt(expected / observations)
        standardized_mean_error = (
            float(event_stats["events_mean"]) - expected
        ) / mean_standard_error
        rows.append(
            {
                "population": population,
                "population_millions": population_millions,
                "scenario": scenario,
                "replications": replications,
                "days_per_replication": days_per_replication,
                "pooled_observations": len(group),
                "analytical_events_day": expected,
                "analytical_tps": expected / SECONDS_PER_DAY,
                **event_stats,
                **tps_stats,
                "poisson_dispersion_index": variance / expected,
                "standardized_mean_error_z": standardized_mean_error,
                "mean_relative_error_pct": (
                    100.0 * (event_stats["events_mean"] - expected) / expected
                ),
                "purpose": (
                    "Stochastic verification of the analytical daily-count model; "
                    "equivalent TPS columns are daily counts divided by 86400 and "
                    "are not instantaneous-throughput measurements."
                ),
            }
        )
    return pd.DataFrame(rows)


def replication_summary(
    samples: pd.DataFrame,
    analytical: pd.DataFrame,
) -> pd.DataFrame:
    expected = analytical[
        ["population", "scenario", "coordination_events_day", "expected_tps"]
    ].rename(
        columns={
            "coordination_events_day": "analytical_events_day",
            "expected_tps": "analytical_tps",
        }
    )
    summary = (
        samples.groupby(
            ["population", "population_millions", "scenario", "replication"],
            as_index=False,
            sort=True,
        )
        .agg(
            observed_days=("day", "count"),
            mean_events_day=("coordination_events_day", "mean"),
            sd_events_day=("coordination_events_day", "std"),
            mean_daily_count_equivalent_tps=("daily_count_equivalent_tps", "mean"),
            daily_count_p95_equivalent_tps=(
                "daily_count_equivalent_tps", lambda values: values.quantile(0.95)
            ),
            daily_count_p99_equivalent_tps=(
                "daily_count_equivalent_tps", lambda values: values.quantile(0.99)
            ),
        )
        .merge(expected, on=["population", "scenario"], how="left", validate="many_to_one")
    )
    summary["mean_relative_error_pct"] = 100.0 * (
        summary["mean_events_day"] - summary["analytical_events_day"]
    ) / summary["analytical_events_day"]
    return summary


def convergence_diagnostics(
    samples: pd.DataFrame,
    analytical: pd.DataFrame,
    replications: int,
) -> pd.DataFrame:
    checkpoints = sorted(
        {
            checkpoint
            for checkpoint in (1, 2, 5, 10, 20, replications)
            if checkpoint <= replications
        }
    )
    expected_lookup = analytical.set_index(["population", "scenario"])[
        "coordination_events_day"
    ]
    rows: list[dict[str, object]] = []
    for (population, population_millions, scenario), group in samples.groupby(
        ["population", "population_millions", "scenario"], sort=True
    ):
        expected = float(expected_lookup.loc[(population, scenario)])
        for checkpoint in checkpoints:
            subset = group[group["replication"] <= checkpoint]
            estimate = float(subset["coordination_events_day"].mean())
            rows.append(
                {
                    "population": population,
                    "population_millions": population_millions,
                    "scenario": scenario,
                    "replications_included": checkpoint,
                    "observations_included": len(subset),
                    "analytical_events_day": expected,
                    "estimated_events_day": estimate,
                    "estimated_average_offered_tps": estimate / SECONDS_PER_DAY,
                    "relative_error_pct": 100.0 * (estimate - expected) / expected,
                }
            )
    return pd.DataFrame(rows)


def temporal_characterization(
    analytical: pd.DataFrame,
    profiles: Sequence[TemporalProfile],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Characterize synthetic hourly concentration using piecewise Poisson means."""

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    hour_centres = np.arange(HOURS_PER_DAY, dtype=float) + 0.5

    for profile in profiles:
        weights = 1.0 + profile.amplitude * np.cos(
            2.0 * np.pi * (hour_centres - profile.peak_hour) / HOURS_PER_DAY
        )
        weights = weights / weights.mean()
        if np.any(weights <= 0.0):
            raise ValueError(f"Temporal profile {profile.name!r} has non-positive weights")

        for record in analytical.to_dict(orient="records"):
            daily_mean = float(record["coordination_events_day"])
            hourly_means = daily_mean * weights / HOURS_PER_DAY
            hourly_tps = hourly_means / SECONDS_PER_HOUR
            for hour, weight, mean_events, mean_tps in zip(
                range(HOURS_PER_DAY), weights, hourly_means, hourly_tps
            ):
                detail_rows.append(
                    {
                        "population": record["population"],
                        "population_millions": record["population_millions"],
                        "scenario": record["scenario"],
                        "temporal_profile": profile.name,
                        "profile_label": profile.label,
                        "hour_start": hour,
                        "normalized_intensity_weight": weight,
                        "expected_events_hour": mean_events,
                        "expected_tps": mean_tps,
                        "hourly_count_normal_p95_equivalent_tps": poisson_normal_quantile(mean_events, 0.95)
                        / SECONDS_PER_HOUR,
                        "hourly_count_normal_p99_equivalent_tps": poisson_normal_quantile(mean_events, 0.99)
                        / SECONDS_PER_HOUR,
                        "profile_provenance": (
                            "synthetic hourly count envelope; equivalent TPS is "
                            "an hourly count divided by 3600, not instantaneous throughput"
                        ),
                    }
                )
            summary_rows.append(
                {
                    "population": record["population"],
                    "population_millions": record["population_millions"],
                    "scenario": record["scenario"],
                    "temporal_profile": profile.name,
                    "profile_label": profile.label,
                    "average_expected_tps": daily_mean / SECONDS_PER_DAY,
                    "peak_expected_tps": float(hourly_tps.max()),
                    "peak_hour": (
                        np.nan if math.isclose(profile.amplitude, 0.0) else int(np.argmax(hourly_tps))
                    ),
                    "peak_to_average_ratio": float(hourly_tps.max() / hourly_tps.mean()),
                    "profile_provenance": (
                        "synthetic analytical assumption; no measured trace or "
                        "instantaneous-throughput claim"
                    ),
                }
            )
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def oat_sensitivity(
    populations: Sequence[int],
    parameters: ModelParameters,
    ranges: Mapping[str, Sequence[float]],
    baseline_scenario: WorkloadScenario,
) -> pd.DataFrame:
    """One-at-a-time sensitivity around the baseline architecture assumptions."""

    rows: list[dict[str, object]] = []
    for population in populations:
        baseline_total = expected_component_rates(
            population, parameters, baseline_scenario
        )["coordination_events_day"]
        for parameter_name, values in ranges.items():
            baseline_value = float(getattr(parameters, parameter_name))
            for value in values:
                adjusted_value: int | float = (
                    int(value) if parameter_name == "institutional_actors" else float(value)
                )
                adjusted = replace(parameters, **{parameter_name: adjusted_value})
                adjusted.validate()
                scenario_for_value = (
                    replace(
                        baseline_scenario,
                        der_adoption_fraction=float(adjusted_value),
                    )
                    if parameter_name == "der_adoption_fraction"
                    else baseline_scenario
                )
                total = expected_component_rates(
                    population, adjusted, scenario_for_value
                )["coordination_events_day"]
                workload_change = total / baseline_total - 1.0
                parameter_change = float(value) / baseline_value - 1.0
                rows.append(
                    {
                        "population": population,
                        "population_millions": population / 1_000_000,
                        "scenario": "baseline",
                        "varied_parameter": parameter_name,
                        "baseline_parameter_value": baseline_value,
                        "tested_value": value,
                        "expected_events_day": total,
                        "expected_tps": total / SECONDS_PER_DAY,
                        "workload_change_pct": 100.0 * workload_change,
                        "secant_elasticity": (
                            workload_change / parameter_change
                            if not math.isclose(parameter_change, 0.0)
                            else np.nan
                        ),
                        "method": "one-at-a-time analytical sensitivity",
                    }
                )
    return pd.DataFrame(rows)


def elasticity_table(analytical: pd.DataFrame) -> pd.DataFrame:
    """Return exact local elasticities of the linear workload equations."""

    rows: list[dict[str, object]] = []
    for record in analytical.to_dict(orient="records"):
        total = float(record["coordination_events_day"])
        consent_share = float(record["consent_events_day"]) / total
        authorization_share = float(record["authorization_events_day"]) / total
        metadata_share = float(record["der_metadata_events_day"]) / total
        integrity_share = float(record["integrity_events_day"]) / total
        elasticities = {
            "consumer_population": consent_share + authorization_share + metadata_share,
            "consent_changes_per_consumer_year": consent_share,
            "authorization_events_per_consumer_day": authorization_share,
            "der_adoption_fraction": metadata_share,
            "metadata_events_per_der_day": metadata_share,
            "institutional_actors": integrity_share,
            "integrity_events_per_actor_day": integrity_share,
        }
        for parameter_name, elasticity in elasticities.items():
            rows.append(
                {
                    "population": record["population"],
                    "population_millions": record["population_millions"],
                    "scenario": record["scenario"],
                    "parameter": parameter_name,
                    "local_elasticity": elasticity,
                    "interpretation": (
                        "Approximate proportional workload change for a 1% "
                        "parameter change, holding other inputs constant."
                    ),
                }
            )
    return pd.DataFrame(rows)


def assumption_grid(
    parameters: ModelParameters,
    populations: Sequence[int],
    authorization_rates: Sequence[float],
    der_adoptions: Sequence[float],
    baseline_scenario: WorkloadScenario,
) -> pd.DataFrame:
    """Two-way authorization/DER sweep for the three required populations."""

    rows: list[dict[str, object]] = []
    for population in populations:
        for authorization_rate in authorization_rates:
            for der_adoption in der_adoptions:
                adjusted = replace(
                    parameters,
                    authorization_events_per_consumer_day=float(authorization_rate),
                    der_adoption_fraction=float(der_adoption),
                )
                rates = expected_component_rates(
                    population,
                    adjusted,
                    replace(
                        baseline_scenario,
                        der_adoption_fraction=float(der_adoption),
                    ),
                )
                rows.append(
                    {
                        "population": population,
                        "population_millions": population / 1_000_000,
                        "authorization_events_per_consumer_day": authorization_rate,
                        "der_adoption_fraction": der_adoption,
                        **rates,
                        "expected_tps": rates["coordination_events_day"]
                        / SECONDS_PER_DAY,
                        "method": "bounded analytical assumption grid",
                    }
                )
    return pd.DataFrame(rows)


def population_der_sensitivity(
    parameters: ModelParameters,
    population_grid: Sequence[int],
    required_populations: Sequence[int],
    der_adoptions: Sequence[float],
    baseline_scenario: WorkloadScenario,
) -> pd.DataFrame:
    """Dense population/DER surface retained for manuscript visualization."""

    populations = sorted(set(population_grid) | set(required_populations))
    rows: list[dict[str, object]] = []
    for population in populations:
        for der_adoption in der_adoptions:
            adjusted = replace(parameters, der_adoption_fraction=float(der_adoption))
            rates = expected_component_rates(
                int(population),
                adjusted,
                replace(
                    baseline_scenario,
                    der_adoption_fraction=float(der_adoption),
                ),
            )
            rows.append(
                {
                    "population": int(population),
                    "population_millions": population / 1_000_000,
                    "der_adoption_fraction": der_adoption,
                    "expected_events_day": rates["coordination_events_day"],
                    "expected_tps": rates["coordination_events_day"]
                    / SECONDS_PER_DAY,
                }
            )
    return pd.DataFrame(rows)


def logical_storage_volume(
    analytical: pd.DataFrame,
    parameters: ModelParameters,
    horizons_days: Sequence[int],
) -> pd.DataFrame:
    """Logical payload volume only; excludes replication and protocol overhead."""

    sizes = {
        "nominal": {
            "consent_events_day": parameters.consent_record_bytes_nominal,
            "authorization_events_day": parameters.authorization_record_bytes_nominal,
            "der_metadata_events_day": parameters.der_metadata_record_bytes_nominal,
            "integrity_events_day": parameters.integrity_record_bytes_nominal,
        },
        "conservative": {
            "consent_events_day": parameters.consent_record_bytes_conservative,
            "authorization_events_day": parameters.authorization_record_bytes_conservative,
            "der_metadata_events_day": parameters.der_metadata_record_bytes_conservative,
            "integrity_events_day": parameters.integrity_record_bytes_conservative,
        },
    }
    rows: list[dict[str, object]] = []
    for record in analytical.to_dict(orient="records"):
        for size_name, component_sizes in sizes.items():
            daily_component_bytes = {
                component: float(record[component]) * component_sizes[component]
                for component in COMPONENT_COLUMNS
            }
            daily_bytes = float(sum(daily_component_bytes.values()))
            weighted_average_bytes = daily_bytes / float(record["coordination_events_day"])
            for horizon_days in horizons_days:
                logical_bytes = daily_bytes * horizon_days
                rows.append(
                    {
                        "population": record["population"],
                        "population_millions": record["population_millions"],
                        "scenario": record["scenario"],
                        "record_size_assumption": size_name,
                        "consent_record_bytes": component_sizes["consent_events_day"],
                        "authorization_record_bytes": component_sizes["authorization_events_day"],
                        "der_metadata_record_bytes": component_sizes["der_metadata_events_day"],
                        "integrity_record_bytes": component_sizes["integrity_events_day"],
                        "workload_weighted_average_record_bytes": weighted_average_bytes,
                        "horizon_days": horizon_days,
                        "coordination_records": float(record["coordination_events_day"]) * horizon_days,
                        "logical_bytes": logical_bytes,
                        "logical_gb_decimal": logical_bytes / 1e9,
                        "logical_tb_decimal": logical_bytes / 1e12,
                        "scope": (
                            "Logical record payload only; excludes block, index, "
                            "replication, networking, and storage-engine overhead."
                        ),
                    }
                )
    return pd.DataFrame(rows)


def workload_boundary(
    analytical: pd.DataFrame,
    parameters: ModelParameters,
) -> pd.DataFrame:
    """Compare event counts only, without interpreting them as byte traffic."""

    rows: list[dict[str, object]] = []
    for record in analytical.to_dict(orient="records"):
        operational_events = (
            int(record["population"])
            * parameters.meter_readings_per_consumer_day
        )
        coordination_events = float(record["coordination_events_day"])
        rows.append(
            {
                "population": record["population"],
                "population_millions": record["population_millions"],
                "scenario": record["scenario"],
                "operational_reading_events_day": operational_events,
                "coordination_events_day": coordination_events,
                "operational_to_coordination_event_count_ratio": (
                    operational_events / coordination_events
                ),
                "coordination_share_of_combined_event_count_pct": 100.0
                * coordination_events
                / (coordination_events + operational_events),
                "scope": (
                    "Event-count comparison only; assumes one modeled event is "
                    "one count and makes no byte, network, or throughput claim."
                ),
            }
        )
    return pd.DataFrame(rows)


def reference_workload_envelopes(
    parameters: ModelParameters,
    scenarios: Sequence[WorkloadScenario],
    budgets_tps: Sequence[int],
) -> pd.DataFrame:
    """Invert the workload equation for hypothetical offered-load budgets.

    The budgets are generic analytical reference values, not claimed HTGL or
    blockchain measurements.
    """

    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        effective = effective_parameters(parameters, scenario)
        consumer_events_per_day = (
            effective["consent_changes_per_consumer_year"] / DAYS_PER_YEAR
            + effective["authorization_events_per_consumer_day"]
            + effective["der_adoption_fraction"]
            * effective["metadata_events_per_der_day"]
        )
        actor_events_day = (
            parameters.institutional_actors
            * effective["integrity_events_per_actor_day"]
        )
        for budget_tps in budgets_tps:
            daily_budget = budget_tps * SECONDS_PER_DAY
            maximum_population = max(
                0.0, (daily_budget - actor_events_day) / consumer_events_per_day
            )
            rows.append(
                {
                    "scenario": scenario.name,
                    "reference_offered_load_budget_tps": budget_tps,
                    "reference_events_day": daily_budget,
                    "analytical_population_at_budget": math.floor(maximum_population),
                    "analytical_population_at_budget_millions": maximum_population
                    / 1_000_000,
                    "consumer_events_per_consumer_day": consumer_events_per_day,
                    "fixed_actor_events_day": actor_events_day,
                    "interpretation": (
                        "Inverse workload-equation result for a hypothetical "
                        "offered-load budget; not a platform benchmark or "
                        "supported-deployment claim."
                    ),
                }
            )
    return pd.DataFrame(rows)


def parameter_table(
    parameters: ModelParameters,
    populations: Sequence[int],
    replications: int,
    days_per_replication: int,
    seed: int,
    evidence: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame:
    metadata = {
        "consumer_populations": ("N_c", "consumers", "required population experiment"),
        "institutional_actors": ("N_a", "actors", "fixed during population scaling"),
        "consent_changes_per_consumer_year": ("lambda_c", "events/consumer/year", "divided by 365 exactly once"),
        "authorization_events_per_consumer_day": ("lambda_a", "events/consumer/day", "baseline workload input"),
        "der_adoption_fraction": ("alpha_d", "fraction", "baseline DER-adoption input"),
        "metadata_events_per_der_day": ("lambda_m", "events/DER/day", "baseline DER metadata input"),
        "integrity_events_per_actor_day": ("lambda_i", "events/actor/day", "baseline integrity input"),
        "meter_readings_per_consumer_day": ("f_m", "readings/consumer/day", "event-count boundary only"),
    }
    values: dict[str, object] = {
        "consumer_populations": json.dumps(list(populations)),
        **asdict(parameters),
    }
    rows: list[dict[str, object]] = []
    for name, value in values.items():
        if name in metadata:
            symbol, unit, role = metadata[name]
        elif name.endswith("_record_bytes_nominal"):
            symbol, unit, role = "B_component,nominal", "bytes/record", "component-specific logical-payload assumption"
        elif name.endswith("_record_bytes_conservative"):
            symbol, unit, role = "B_component,conservative", "bytes/record", "component-specific upper logical-payload assumption"
        else:
            raise RuntimeError(f"missing parameter metadata for {name}")
        source = evidence[name]
        rows.append(
            {
                "symbol": symbol,
                "parameter": name,
                "value": value,
                "unit": unit,
                "role": role,
                "evidence_class": source["evidence_class"],
                "citation": source["citation"],
                "rationale": source["rationale"],
            }
        )
    rows.extend(
        [
            {
                "symbol": "R",
                "parameter": "independent_replications",
                "value": replications,
                "unit": "replications/case",
                "role": "stochastic implementation verification",
                "evidence_class": "experiment_design",
                "citation": None,
                "rationale": "Repeated independent seed streams quantify implementation-verification error.",
            },
            {
                "symbol": "D",
                "parameter": "days_per_replication",
                "value": days_per_replication,
                "unit": "days/replication",
                "role": "stochastic implementation verification",
                "evidence_class": "experiment_design",
                "citation": None,
                "rationale": "A full synthetic year avoids a short-horizon verification result.",
            },
            {
                "symbol": "seed",
                "parameter": "master_seed",
                "value": seed,
                "unit": "integer",
                "role": "reproducibility",
                "evidence_class": "experiment_design",
                "citation": None,
                "rationale": "A fixed published master seed enables exact deterministic regeneration.",
            },
        ]
    )
    return pd.DataFrame(rows)


def scenario_table(
    scenarios: Sequence[WorkloadScenario],
    evidence: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        row = asdict(scenario)
        source = evidence[scenario.name]
        row.update(
            {
                "evidence_class": source["evidence_class"],
                "citation": source["citation"],
                "rationale": source["rationale"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def statistical_validation_metrics(
    stochastic: pd.DataFrame,
    familywise_alpha: float,
) -> pd.DataFrame:
    """Return calibrated, familywise-controlled implementation checks per case."""

    cases = len(stochastic)
    if cases <= 0:
        raise ValueError("stochastic summary must contain at least one case")
    per_tail_probability = familywise_alpha / (2.0 * cases)
    z_critical = NORMAL.inv_cdf(1.0 - per_tail_probability)
    result = stochastic[
        [
            "population",
            "population_millions",
            "scenario",
            "pooled_observations",
            "analytical_events_day",
            "events_mean",
            "standardized_mean_error_z",
            "poisson_dispersion_index",
        ]
    ].copy()
    result["familywise_alpha"] = familywise_alpha
    result["bonferroni_case_count"] = cases
    result["mean_z_critical"] = z_critical
    result["mean_test_passed"] = (
        result["standardized_mean_error_z"].abs() <= z_critical
    )
    degrees_of_freedom = result["pooled_observations"].astype(int) - 1
    result["dispersion_degrees_of_freedom"] = degrees_of_freedom
    result["dispersion_acceptance_lower"] = [
        float(chi2.ppf(per_tail_probability, int(df)) / df)
        for df in degrees_of_freedom
    ]
    result["dispersion_acceptance_upper"] = [
        float(chi2.ppf(1.0 - per_tail_probability, int(df)) / df)
        for df in degrees_of_freedom
    ]
    result["dispersion_test_passed"] = (
        result["poisson_dispersion_index"]
        >= result["dispersion_acceptance_lower"]
    ) & (
        result["poisson_dispersion_index"]
        <= result["dispersion_acceptance_upper"]
    )
    result["test_basis"] = (
        "Bonferroni familywise mean z-test and index-of-dispersion chi-square "
        "acceptance interval; verifies the synthetic Poisson implementation only"
    )
    return result


def verification_checks(
    analytical: pd.DataFrame,
    stochastic: pd.DataFrame,
    statistical_metrics: pd.DataFrame,
    required_populations: Sequence[int],
) -> pd.DataFrame:
    checks: list[dict[str, object]] = []

    expected_populations = set(required_populations)
    observed_populations = {
        int(value) for value in analytical["population"].astype(int).unique()
    }
    checks.append(
        {
            "check": "required_populations_present",
            "passed": observed_populations == expected_populations,
            "observed": json.dumps(sorted(observed_populations)),
            "criterion": json.dumps(sorted(expected_populations)),
        }
    )

    component_sum = analytical[list(COMPONENT_COLUMNS)].sum(axis=1)
    maximum_sum_error = float(
        np.max(np.abs(component_sum - analytical["coordination_events_day"]))
    )
    checks.append(
        {
            "check": "component_sum_identity",
            "passed": maximum_sum_error < 1e-6,
            "observed": maximum_sum_error,
            "criterion": "absolute error < 1e-6 events/day",
        }
    )

    maximum_abs_z = float(statistical_metrics["standardized_mean_error_z"].abs().max())
    z_critical = float(statistical_metrics["mean_z_critical"].iloc[0])
    checks.append(
        {
            "check": "familywise_standardized_mean_error",
            "passed": bool(statistical_metrics["mean_test_passed"].all()),
            "observed": maximum_abs_z,
            "criterion": f"every absolute standardized mean error <= {z_critical:.8g}",
        }
    )

    failing_dispersion = int((~statistical_metrics["dispersion_test_passed"]).sum())
    checks.append(
        {
            "check": "familywise_poisson_dispersion_interval",
            "passed": failing_dispersion == 0,
            "observed": failing_dispersion,
            "criterion": "zero cases outside Bonferroni chi-square dispersion intervals",
        }
    )

    monotonic = True
    for _, group in analytical.sort_values("population").groupby("scenario"):
        monotonic = monotonic and bool(np.all(np.diff(group["expected_tps"]) > 0))
    checks.append(
        {
            "check": "population_scaling_is_monotonic",
            "passed": monotonic,
            "observed": monotonic,
            "criterion": "expected TPS increases from 50M to 150M to 450M",
        }
    )

    all_finite = bool(
        np.isfinite(
            analytical[
                [
                    "coordination_events_day",
                    "expected_tps",
                    "daily_count_normal_p99_equivalent_tps",
                ]
            ].to_numpy(dtype=float)
        ).all()
    )
    checks.append(
        {
            "check": "analytical_outputs_are_finite",
            "passed": all_finite,
            "observed": all_finite,
            "criterion": "all principal analytical values are finite",
        }
    )
    return pd.DataFrame(checks)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manuscript_artifacts(
    artifact_dir: Path,
    analytical: pd.DataFrame,
    stochastic: pd.DataFrame,
    component_shares: pd.DataFrame,
    sensitivity: pd.DataFrame,
    parameters: ModelParameters,
    replications: int,
    days_per_replication: int,
    populations: Sequence[int],
) -> None:
    tables = artifact_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    summary = analytical.merge(
        stochastic[
            [
                "population",
                "scenario",
                "daily_count_equivalent_tps_mean",
                "daily_count_equivalent_tps_p95",
                "daily_count_equivalent_tps_p99",
                "mean_relative_error_pct",
            ]
        ],
        on=["population", "scenario"],
        how="left",
        validate="one_to_one",
    )
    write_csv(
        summary[
            [
                "population_millions",
                "scenario",
                "coordination_events_day",
                "expected_tps",
                "daily_count_normal_p99_equivalent_tps",
                "daily_count_equivalent_tps_mean",
                "daily_count_equivalent_tps_p95",
                "daily_count_equivalent_tps_p99",
                "mean_relative_error_pct",
            ]
        ],
        tables / "population_scenario_summary.csv",
    )
    write_csv(component_shares, tables / "component_decomposition.csv")

    sensitivity_extremes = (
        sensitivity.groupby(
            ["population_millions", "varied_parameter"], as_index=False
        )
        .agg(
            minimum_tps=("expected_tps", "min"),
            maximum_tps=("expected_tps", "max"),
        )
        .sort_values(["population_millions", "varied_parameter"])
    )
    write_csv(sensitivity_extremes, tables / "sensitivity_ranges.csv")

    baseline = analytical[
        (analytical["population"] == populations[0])
        & (analytical["scenario"] == "baseline")
    ].iloc[0]
    baseline_stochastic = stochastic[
        (stochastic["population"] == populations[0])
        & (stochastic["scenario"] == "baseline")
    ].iloc[0]
    authorization_share = baseline["authorization_events_day_share_pct"]

    (artifact_dir / "claim_boundary.md").write_text(
        "# Claim boundary\n\n"
        "This experiment characterizes the offered coordination workload implied "
        "by the proposed HTGL architecture. It does **not** implement HTGL and "
        "does not measure consensus throughput, validator utilization, network "
        "behavior, latency, finality, cryptographic cost, Proof-after-Erasure, "
        "security, or legal compliance. Stochastic sampling verifies the "
        "analytical daily-count equations; it is not an end-to-end system "
        "simulation.\n",
        encoding="utf-8",
    )
    (artifact_dir / "methodology_notes.md").write_text(
        "# Analytical workload methodology\n\n"
        "The coordination-event model is:\n\n"
        "- `R_c = N_c * lambda_c / 365` (consent-state changes/day)\n"
        "- `R_a = N_c * lambda_a` (authorization decisions/day)\n"
        "- `R_m = N_c * alpha_d * lambda_m` (DER metadata changes/day)\n"
        "- `R_i = N_a * lambda_i` (integrity/audit anchors/day)\n"
        "- `W = R_c + R_a + R_m + R_i` (coordination events/day)\n"
        "- `average offered TPS = W / 86,400`\n\n"
        f"The default stochastic-verification design uses {replications} independent "
        f"replications of {days_per_replication} days for every population-scenario "
        "pair. Each component is sampled as an independent daily Poisson count. "
        "The synthetic temporal profiles and reference workload budgets are "
        "what-if assumptions, not observations.\n",
        encoding="utf-8",
    )
    (artifact_dir / "paper_result_sentences.md").write_text(
        "# Draft result statements\n\n"
        f"- Under the baseline assumptions and a population of 50 million, the "
        f"analytical coordination workload is {baseline['coordination_events_day']:,.0f} "
        f"events/day ({baseline['expected_tps']:.2f} average offered TPS).\n"
        f"- Authorization decisions account for {authorization_share:.2f}% of the "
        "baseline modeled coordination-event count.\n"
        f"- Across {replications * days_per_replication:,} pooled daily observations "
        f"for the 50-million baseline case, the stochastic mean is "
        f"{baseline_stochastic['daily_count_equivalent_tps_mean']:.3f} daily-count-equivalent TPS and differs from the analytical "
        f"mean by {baseline_stochastic['mean_relative_error_pct']:.6f}%.\n"
        "- These quantities characterize offered workload under explicit "
        "assumptions; they do not establish the performance of an HTGL "
        "implementation.\n",
        encoding="utf-8",
    )


@dataclass
class ExperimentBundle:
    config: dict[str, Any]
    parameters: ModelParameters
    populations: tuple[int, ...]
    scenarios: tuple[WorkloadScenario, ...]
    temporal_profiles: tuple[TemporalProfile, ...]
    analytical: pd.DataFrame
    stochastic: pd.DataFrame
    samples: pd.DataFrame
    sensitivity: pd.DataFrame
    statistical_metrics: pd.DataFrame
    checks: pd.DataFrame
    frames: dict[str, pd.DataFrame]


def temporal_profile_table(
    profiles: Sequence[TemporalProfile],
    evidence: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for profile in profiles:
        source = evidence[profile.name]
        rows.append(
            {
                **asdict(profile),
                "evidence_class": source["evidence_class"],
                "citation": source["citation"],
                "rationale": source["rationale"],
            }
        )
    return pd.DataFrame(rows)


def claim_matrix_table() -> pd.DataFrame:
    """Machine-readable boundary between supported and prohibited claims."""

    rows = [
        {
            "claim_id": "C01",
            "status": "supported_under_stated_assumptions",
            "publication_wording": (
                "The equations characterize expected offered coordination-event "
                "counts and daily-average offered TPS under the published inputs."
            ),
            "prohibited_interpretation": "Measured HTGL or blockchain throughput.",
            "evidence_files": "analytical_workload_results.csv;component_decomposition.csv",
            "required_checks": "component_sum_identity;analytical_outputs_are_finite",
            "assumption_dependency": "all workload-rate inputs",
        },
        {
            "claim_id": "C02",
            "status": "supported_under_stated_assumptions",
            "publication_wording": "Modeled offered workload increases monotonically over the configured populations.",
            "prohibited_interpretation": "Demonstrated implementation scalability.",
            "evidence_files": "analytical_workload_results.csv",
            "required_checks": "population_scaling_is_monotonic",
            "assumption_dependency": "configured population cases and fixed actor count",
        },
        {
            "claim_id": "C03",
            "status": "supported_under_stated_assumptions",
            "publication_wording": "Authorization dominates the baseline event count for the configured default hypothesis.",
            "prohibited_interpretation": "An empirically universal workload composition.",
            "evidence_files": "component_decomposition.csv;oat_sensitivity.csv;model_parameters.csv",
            "required_checks": "component_sum_identity",
            "assumption_dependency": "especially authorization_events_per_consumer_day",
        },
        {
            "claim_id": "C04",
            "status": "supported_under_stated_assumptions",
            "publication_wording": "Sensitivity and elasticity results quantify dependence on explicit model inputs.",
            "prohibited_interpretation": "Empirical calibration or causal inference.",
            "evidence_files": "oat_sensitivity.csv;local_elasticities.csv;authorization_der_assumption_grid.csv",
            "required_checks": "analytical_outputs_are_finite",
            "assumption_dependency": "configured sensitivity ranges",
        },
        {
            "claim_id": "C05",
            "status": "supported_as_synthetic_envelope",
            "publication_wording": "Configured temporal profiles provide synthetic hourly offered-load envelopes.",
            "prohibited_interpretation": "Observed traces or instantaneous p99 throughput.",
            "evidence_files": "temporal_profile_definitions.csv;temporal_profile_summary.csv;temporal_profile_detail.csv",
            "required_checks": "analytical_outputs_are_finite",
            "assumption_dependency": "temporal amplitudes and peak hours",
        },
        {
            "claim_id": "C06",
            "status": "supported_as_logical_payload_estimate",
            "publication_wording": "Logical metadata volume follows from component-specific payload-size hypotheses.",
            "prohibited_interpretation": "Physical ledger, database, network, or replicated storage consumption.",
            "evidence_files": "logical_metadata_volume.csv;model_parameters.csv",
            "required_checks": "analytical_outputs_are_finite",
            "assumption_dependency": "component-specific record-size inputs",
        },
        {
            "claim_id": "C07",
            "status": "supported_as_implementation_verification",
            "publication_wording": "Independent deterministic Poisson streams agree with the analytical equations within calibrated familywise tests.",
            "prohibited_interpretation": "Validation of real arrivals, burstiness, or an HTGL implementation.",
            "evidence_files": "stochastic_verification_summary.csv;statistical_validation_metrics.csv;verification_checks.csv",
            "required_checks": "familywise_standardized_mean_error;familywise_poisson_dispersion_interval",
            "assumption_dependency": "independent homogeneous daily Poisson model",
        },
        {
            "claim_id": "P01",
            "status": "prohibited",
            "publication_wording": "No implementation-performance claim is supported.",
            "prohibited_interpretation": "Throughput capacity, latency, finality, validator utilization, or supported population.",
            "evidence_files": "",
            "required_checks": "",
            "assumption_dependency": "HTGL is not implemented",
        },
        {
            "claim_id": "P02",
            "status": "prohibited",
            "publication_wording": "Daily-count quantiles are labeled only as equivalent daily-average rates.",
            "prohibited_interpretation": "Instantaneous p95 or p99 throughput requirement.",
            "evidence_files": "",
            "required_checks": "",
            "assumption_dependency": "daily aggregation",
        },
        {
            "claim_id": "P03",
            "status": "prohibited",
            "publication_wording": "No security, privacy, erasure, or legal-compliance conclusion is supported.",
            "prohibited_interpretation": "Security proof, GDPR compliance, or Proof-after-Erasure validation.",
            "evidence_files": "",
            "required_checks": "",
            "assumption_dependency": "outside analytical workload scope",
        },
    ]
    return pd.DataFrame(rows)


def validate_claim_matrix(
    claim_matrix: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
    checks: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "claim_id",
        "status",
        "publication_wording",
        "prohibited_interpretation",
        "evidence_files",
        "required_checks",
        "assumption_dependency",
    }
    if set(claim_matrix.columns) != required_columns:
        raise RuntimeError("claim matrix schema is not exact")
    if claim_matrix["claim_id"].duplicated().any():
        raise RuntimeError("claim matrix contains duplicate claim IDs")
    check_lookup = checks.set_index("check")["passed"].astype(bool).to_dict()
    failures: list[str] = []
    for row in claim_matrix.to_dict(orient="records"):
        if row["status"] == "prohibited":
            if not str(row["prohibited_interpretation"]).strip():
                failures.append(f"{row['claim_id']}: prohibited boundary is blank")
            continue
        for filename in filter(None, str(row["evidence_files"]).split(";")):
            if filename not in frames and filename != "verification_checks.csv":
                failures.append(f"{row['claim_id']}: missing evidence file {filename}")
        for check_name in filter(None, str(row["required_checks"]).split(";")):
            if check_lookup.get(check_name) is not True:
                failures.append(f"{row['claim_id']}: required check {check_name} did not pass")
    if failures:
        raise RuntimeError("claim matrix validation failed: " + "; ".join(failures))
    return pd.DataFrame(
        [
            {
                "validation": "claim_matrix_evidence_and_gate_resolution",
                "passed": True,
                "claims_checked": len(claim_matrix),
                "supported_claims": int((claim_matrix["status"] != "prohibited").sum()),
                "prohibited_claims": int((claim_matrix["status"] == "prohibited").sum()),
            }
        ]
    )


def build_experiment(config: dict[str, Any], write_daily_samples: bool) -> ExperimentBundle:
    parameters = ModelParameters(**config["model_parameters"])
    parameters.validate()
    populations = tuple(int(value) for value in config["populations"])
    scenarios = tuple(WorkloadScenario(**value) for value in config["scenarios"])
    profiles = tuple(TemporalProfile(**value) for value in config["temporal_profiles"])
    for scenario in scenarios:
        scenario.validate()

    replications_count = int(config["replications"])
    days_count = int(config["days_per_replication"])
    seed = int(config["seed"])
    analytical = analytical_characterization(populations, scenarios, parameters)
    component_shares = component_share_table(analytical)
    samples = generate_daily_samples(
        populations,
        scenarios,
        parameters,
        replications_count,
        days_count,
        seed,
    )
    stochastic = summarize_stochastic_samples(
        samples, analytical, replications_count, days_count
    )
    replication_rows = replication_summary(samples, analytical)
    convergence = convergence_diagnostics(samples, analytical, replications_count)
    temporal_detail, temporal_summary = temporal_characterization(analytical, profiles)
    sensitivity = oat_sensitivity(
        populations,
        parameters,
        config["sensitivity_ranges"],
        scenarios[0],
    )
    elasticities = elasticity_table(analytical)
    grid = assumption_grid(
        parameters,
        populations,
        config["authorization_grid"],
        config["der_adoption_grid"],
        scenarios[0],
    )
    population_der = population_der_sensitivity(
        parameters,
        config["population_der_grid"],
        populations,
        config["der_adoption_grid"],
        scenarios[0],
    )
    storage = logical_storage_volume(
        analytical,
        parameters,
        config["storage_horizons_days"],
    )
    boundary = workload_boundary(analytical, parameters)
    envelopes = reference_workload_envelopes(
        parameters,
        scenarios,
        config["reference_budgets_tps"],
    )
    statistical_metrics = statistical_validation_metrics(
        stochastic,
        float(config["familywise_alpha"]),
    )
    checks = verification_checks(
        analytical,
        stochastic,
        statistical_metrics,
        populations,
    )

    analytical_vs_stochastic = analytical[
        [
            "population",
            "population_millions",
            "scenario",
            "coordination_events_day",
            "expected_tps",
            "daily_count_normal_p95_equivalent_tps",
            "daily_count_normal_p99_equivalent_tps",
        ]
    ].merge(
        stochastic[
            [
                "population",
                "scenario",
                "events_mean",
                "daily_count_equivalent_tps_mean",
                "daily_count_equivalent_tps_p95",
                "daily_count_equivalent_tps_p99",
                "poisson_dispersion_index",
                "standardized_mean_error_z",
                "mean_relative_error_pct",
            ]
        ],
        on=["population", "scenario"],
        how="left",
        validate="one_to_one",
    )

    frames: dict[str, pd.DataFrame] = {
        "model_parameters.csv": parameter_table(
            parameters,
            populations,
            replications_count,
            days_count,
            seed,
            config["parameter_evidence"],
        ),
        "scenario_definitions.csv": scenario_table(
            scenarios, config["scenario_evidence"]
        ),
        "temporal_profile_definitions.csv": temporal_profile_table(
            profiles, config["temporal_profile_evidence"]
        ),
        "analytical_workload_results.csv": analytical,
        "component_decomposition.csv": component_shares,
        "stochastic_verification_summary.csv": stochastic,
        "statistical_validation_metrics.csv": statistical_metrics,
        "replication_summary.csv": replication_rows,
        "convergence_diagnostics.csv": convergence,
        "analytical_vs_stochastic.csv": analytical_vs_stochastic,
        "temporal_profile_detail.csv": temporal_detail,
        "temporal_profile_summary.csv": temporal_summary,
        "oat_sensitivity.csv": sensitivity,
        "local_elasticities.csv": elasticities,
        "authorization_der_assumption_grid.csv": grid,
        "population_der_sensitivity.csv": population_der,
        "logical_metadata_volume.csv": storage,
        "workload_boundary_event_counts.csv": boundary,
        "reference_workload_envelopes.csv": envelopes,
        "verification_checks.csv": checks,
    }
    claim_matrix = claim_matrix_table()
    claim_validation = validate_claim_matrix(claim_matrix, frames, checks)
    frames["claim_matrix.csv"] = claim_matrix
    frames["claim_matrix_validation.csv"] = claim_validation
    if write_daily_samples:
        frames["daily_stochastic_samples.csv"] = samples

    failed_checks = checks[~checks["passed"].astype(bool)]
    if not failed_checks.empty:
        failed = ", ".join(failed_checks["check"].astype(str))
        raise RuntimeError(f"analytical verification failed: {failed}")
    return ExperimentBundle(
        config=config,
        parameters=parameters,
        populations=populations,
        scenarios=scenarios,
        temporal_profiles=profiles,
        analytical=analytical,
        stochastic=stochastic,
        samples=samples,
        sensitivity=sensitivity,
        statistical_metrics=statistical_metrics,
        checks=checks,
        frames=frames,
    )


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        None,
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    ).encode("utf-8")


def frame_fingerprints(frames: Mapping[str, pd.DataFrame]) -> dict[str, str]:
    return {
        filename: digest_bytes(csv_bytes(frame))
        for filename, frame in sorted(frames.items())
    }


def combined_fingerprint(fingerprints: Mapping[str, str]) -> str:
    return digest_bytes(canonical_json_bytes(dict(sorted(fingerprints.items()))))


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
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    validate_unicode_bytes(data, str(path))
    atomic_write_bytes(path, data)


def package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for distribution in ("numpy", "pandas", "scipy"):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"required dependency {distribution!r} is not installed") from exc
    return result


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
        return {"available": True, "commit": commit, "tracked_tree_dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "commit": None, "tracked_tree_dirty": None}


def environment_snapshot() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "packages": package_versions(),
        "platform": platform.platform(),
        "operating_system": platform.system(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "numpy_bit_generator": "PCG64 via numpy.random.default_rng",
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


def requirements_lock_text(environment: Mapping[str, object]) -> str:
    packages = environment["packages"]
    if not isinstance(packages, dict):
        raise RuntimeError("environment package inventory is malformed")
    return "".join(f"{name}=={version}\n" for name, version in sorted(packages.items()))


def file_inventory(
    root: Path,
    frame_schemas: Mapping[str, pd.DataFrame],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        entry: dict[str, object] = {
            "path": relative,
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
            "row_count": None,
            "columns": None,
        }
        if relative in frame_schemas:
            frame = frame_schemas[relative]
            entry["row_count"] = len(frame)
            entry["columns"] = list(frame.columns)
        rows.append(entry)
    return rows


def validate_inventory(root: Path, entries: Sequence[Mapping[str, object]]) -> None:
    expected = {str(entry["path"]) for entry in entries}
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if observed != expected:
        raise RuntimeError(
            f"staged file-set mismatch; missing={sorted(expected-observed)}, "
            f"unexpected={sorted(observed-expected)}"
        )
    for entry in entries:
        path = root / str(entry["path"])
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"staged release contains invalid file {entry['path']}")
        if path.stat().st_size != int(entry["size_bytes"]):
            raise RuntimeError(f"size mismatch for {entry['path']}")
        if sha256(path) != entry["sha256"]:
            raise RuntimeError(f"checksum mismatch for {entry['path']}")
        if entry.get("row_count") is not None:
            frame = pd.read_csv(path)
            if len(frame) != int(entry["row_count"]):
                raise RuntimeError(f"row-count mismatch for {entry['path']}")
            if list(frame.columns) != list(entry["columns"]):
                raise RuntimeError(f"column-schema mismatch for {entry['path']}")


@contextlib.contextmanager
def publication_lock(root: Path) -> Iterable[None]:
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".publication.lock"
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f"publication lock exists at {lock_path}; another run may be active. "
            "Inspect and remove only if the owning process is no longer running."
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


def create_staging_directory(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise RuntimeError(f"publication root must not be a symbolic link: {root}")
    return Path(tempfile.mkdtemp(prefix=".staging-", dir=root))


def publish_immutable_run(stage: Path, root: Path, run_id: str) -> Path:
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    destination = runs / run_id
    if destination.exists():
        raise RuntimeError(f"immutable run already exists: {destination}")
    os.replace(stage, destination)
    return destination


def write_latest_pointer(
    root: Path,
    run_id: str,
    commit_path: Path,
    config_digest: str,
) -> None:
    write_json(
        root / "latest.json",
        {
            "contract_version": CONTRACT_VERSION,
            "run_id": run_id,
            "relative_run_directory": f"runs/{run_id}",
            "release_commit_sha256": sha256(commit_path),
            "configuration_sha256": config_digest,
        },
    )


def resolve_effective_config(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.expanduser().resolve() if args.config is not None else None
    config = load_config(config_path)
    if args.replications is not None:
        config["replications"] = args.replications
    if args.days_per_replication is not None:
        config["days_per_replication"] = args.days_per_replication
    if args.seed is not None:
        config["seed"] = args.seed
    validate_config(config)
    return config


def run_self_tests() -> None:
    config = load_config(None)
    config["replications"] = 2
    config["days_per_replication"] = 3
    validate_config(config)
    first = build_experiment(config, write_daily_samples=True)
    second = build_experiment(config, write_daily_samples=True)
    if frame_fingerprints(first.frames) != frame_fingerprints(second.frames):
        raise RuntimeError("self-test failed: deterministic frame fingerprints differ")
    baseline = first.analytical[
        (first.analytical["population"] == 50_000_000)
        & (first.analytical["scenario"] == "baseline")
    ].iloc[0]
    if not math.isclose(float(baseline["coordination_events_day"]), 30_285_972.60273973):
        raise RuntimeError("self-test failed: baseline closed-form result changed")
    invalid = json.loads(json.dumps(config))
    invalid["unknown_key"] = 1
    try:
        validate_config(invalid)
    except ValueError:
        pass
    else:
        raise RuntimeError("self-test failed: unknown config key was accepted")
    validate_unicode_bytes("158.5×; only—no benchmark\n".encode("utf-8"), "unicode test")
    print("All simUp.py self-tests passed.")


def execute(args: argparse.Namespace) -> None:
    config = resolve_effective_config(args)
    write_daily_samples = not args.no_daily_samples
    output_root = args.output_dir.expanduser().resolve()
    artifact_root = args.artifact_dir.expanduser().resolve()
    if output_root == artifact_root:
        raise ValueError("--output-dir and --artifact-dir must be different directories")

    source_path = Path(__file__).resolve()
    source_hash = sha256(source_path)
    config_digest = digest_bytes(canonical_json_bytes(config))
    print(f"Running {CODE_VERSION}")
    print(f"Scope: {SCOPE_NOTE}")
    print(
        "Populations: "
        + ", ".join(f"{value / 1e6:.0f}M" for value in config["populations"])
    )
    print(
        "Repeated study: "
        f"{config['replications']} replications x "
        f"{config['days_per_replication']} days per population-scenario pair"
    )
    print("Publication policy: immutable staged run with fail-closed release gates")

    first = build_experiment(config, write_daily_samples)
    second = build_experiment(
        json.loads(json.dumps(config)),
        write_daily_samples,
    )
    first_core_fingerprints = frame_fingerprints(first.frames)
    second_core_fingerprints = frame_fingerprints(second.frames)
    reproduction_match = first_core_fingerprints == second_core_fingerprints
    if not reproduction_match:
        differing = sorted(
            {
                *(
                    name
                    for name in first_core_fingerprints
                    if first_core_fingerprints.get(name)
                    != second_core_fingerprints.get(name)
                ),
                *(
                    name
                    for name in second_core_fingerprints
                    if first_core_fingerprints.get(name)
                    != second_core_fingerprints.get(name)
                ),
            }
        )
        raise RuntimeError(
            "independent deterministic regeneration failed for: "
            + ", ".join(differing)
        )
    core_fingerprint = combined_fingerprint(first_core_fingerprints)
    reproducibility_row = pd.DataFrame(
        [
            {
                "check": "independent_deterministic_regeneration",
                "passed": reproduction_match,
                "first_combined_sha256": core_fingerprint,
                "second_combined_sha256": combined_fingerprint(
                    second_core_fingerprints
                ),
                "files_compared": len(first_core_fingerprints),
                "comparison": "byte-identical canonical UTF-8 CSV serialization",
            }
        ]
    )
    release_check = pd.DataFrame(
        [
            {
                "check": "independent_deterministic_regeneration",
                "passed": reproduction_match,
                "observed": core_fingerprint,
                "criterion": "independent complete frame fingerprints are identical",
            }
        ]
    )
    for bundle in (first, second):
        bundle.checks = pd.concat(
            [bundle.checks, release_check],
            ignore_index=True,
        )
        bundle.frames["verification_checks.csv"] = bundle.checks
        bundle.frames["reproducibility_check.csv"] = reproducibility_row
        bundle.frames["claim_matrix_validation.csv"] = validate_claim_matrix(
            bundle.frames["claim_matrix.csv"],
            bundle.frames,
            bundle.checks,
        )
    first_final_fingerprints = frame_fingerprints(first.frames)
    second_final_fingerprints = frame_fingerprints(second.frames)
    if first_final_fingerprints != second_final_fingerprints:
        raise RuntimeError(
            "final deterministic release frames differ after adding release gates"
        )
    final_fingerprint = combined_fingerprint(first_final_fingerprints)

    environment = environment_snapshot()
    generated_at = datetime.now(timezone.utc)
    run_id = (
        generated_at.strftime("%Y%m%dT%H%M%S%fZ")
        + f"-{config_digest[:10]}-{source_hash[:10]}-{uuid.uuid4().hex[:8]}"
    )
    data_stage = create_staging_directory(output_root)
    artifact_stage = create_staging_directory(artifact_root)
    artifact_destination: Path | None = None
    data_destination: Path | None = None

    try:
        for filename, frame in sorted(first.frames.items()):
            data = csv_bytes(frame)
            validate_unicode_bytes(data, filename)
            atomic_write_bytes(data_stage / filename, data)

        write_json(data_stage / "effective_config.json", config)
        write_json(data_stage / "environment_snapshot.json", environment)
        atomic_write_bytes(
            data_stage / "requirements-lock.txt",
            requirements_lock_text(environment).encode("utf-8"),
        )
        reproducibility_report = {
            "contract_version": CONTRACT_VERSION,
            "check": "independent deterministic regeneration",
            "passed": reproduction_match,
            "canonical_serialization": (
                "UTF-8 CSV, LF line endings, no index, %.12g float formatting"
            ),
            "files_compared": first_core_fingerprints,
            "first_combined_sha256": core_fingerprint,
            "second_combined_sha256": combined_fingerprint(
                second_core_fingerprints
            ),
            "final_release_frame_sha256": final_fingerprint,
        }
        write_json(
            data_stage / "reproducibility_report.json",
            reproducibility_report,
        )

        write_manuscript_artifacts(
            artifact_stage,
            first.analytical,
            first.stochastic,
            first.frames["component_decomposition.csv"],
            first.sensitivity,
            first.parameters,
            int(config["replications"]),
            int(config["days_per_replication"]),
            first.populations,
        )
        write_csv(
            first.frames["claim_matrix.csv"],
            artifact_stage / "tables" / "claim_matrix.csv",
        )
        for path in sorted(
            value for value in artifact_stage.rglob("*") if value.is_file()
        ):
            validate_unicode_bytes(path.read_bytes(), path.relative_to(artifact_stage).as_posix())

        artifact_pre_inventory = file_inventory(artifact_stage, {})
        artifact_checksums = pd.DataFrame(
            [
                {
                    "file": entry["path"],
                    "sha256": entry["sha256"],
                    "size_bytes": entry["size_bytes"],
                }
                for entry in artifact_pre_inventory
            ]
        )
        write_csv(
            artifact_checksums,
            artifact_stage / "artifact_checksums.csv",
        )
        artifact_inventory = file_inventory(artifact_stage, {})
        artifact_manifest = {
            "contract_version": CONTRACT_VERSION,
            "run_id": run_id,
            "generated_at_utc": generated_at.isoformat(),
            "code_version": CODE_VERSION,
            "source": {
                "file": "simUp.py",
                "sha256": source_hash,
            },
            "configuration_sha256": config_digest,
            "scope": SCOPE_NOTE,
            "files": artifact_inventory,
        }
        validate_inventory(artifact_stage, artifact_inventory)
        write_json(
            artifact_stage / "artifact_manifest.json",
            artifact_manifest,
        )
        artifact_manifest_hash = sha256(
            artifact_stage / "artifact_manifest.json"
        )
        artifact_commit = {
            "complete": True,
            "contract_version": CONTRACT_VERSION,
            "run_id": run_id,
            "artifact_manifest_sha256": artifact_manifest_hash,
            "configuration_sha256": config_digest,
        }
        write_json(artifact_stage / "RELEASE_COMMIT.json", artifact_commit)

        pre_checksum_inventory = file_inventory(data_stage, first.frames)
        output_checksums = pd.DataFrame(
            [
                {
                    "file": entry["path"],
                    "sha256": entry["sha256"],
                    "size_bytes": entry["size_bytes"],
                    "row_count": entry["row_count"],
                }
                for entry in pre_checksum_inventory
            ]
        )
        write_csv(output_checksums, data_stage / "output_checksums.csv")
        data_inventory = file_inventory(data_stage, first.frames)
        expected_daily_rows = (
            len(first.populations)
            * len(first.scenarios)
            * int(config["replications"])
            * int(config["days_per_replication"])
        )
        daily_contract_valid = (
            (
                write_daily_samples
                and "daily_stochastic_samples.csv" in first.frames
                and len(first.frames["daily_stochastic_samples.csv"])
                == expected_daily_rows
            )
            or (
                not write_daily_samples
                and "daily_stochastic_samples.csv" not in first.frames
                and not (data_stage / "daily_stochastic_samples.csv").exists()
            )
        )
        if not daily_contract_valid:
            raise RuntimeError("daily-sample publication contract failed")
        all_checks_passed = bool(first.checks["passed"].astype(bool).all())
        claim_matrix_passed = bool(
            first.frames["claim_matrix_validation.csv"]["passed"].astype(bool).all()
        )
        publication_gates = {
            "analytical_and_statistical_checks_passed": all_checks_passed,
            "claim_matrix_validated": claim_matrix_passed,
            "independent_reproducibility_check_passed": reproduction_match,
            "daily_sample_contract_passed": daily_contract_valid,
        }
        if not all(publication_gates.values()):
            raise RuntimeError(
                "publication gates failed: "
                + json.dumps(publication_gates, sort_keys=True)
            )
        manifest = {
            "contract_version": CONTRACT_VERSION,
            "run_id": run_id,
            "generated_at_utc": generated_at.isoformat(),
            "code_version": CODE_VERSION,
            "source": {
                "file": "simUp.py",
                "sha256": source_hash,
            },
            "source_control": environment["source_control"],
            "command_line": manifest_command_line(),
            "configuration_sha256": config_digest,
            "environment": environment,
            "scope": SCOPE_NOTE,
            "populations": list(first.populations),
            "scenarios": [scenario.name for scenario in first.scenarios],
            "replications_per_case": int(config["replications"]),
            "days_per_replication": int(config["days_per_replication"]),
            "observations_per_case": (
                int(config["replications"])
                * int(config["days_per_replication"])
            ),
            "total_daily_observations_evaluated": len(first.samples),
            "daily_samples_written": write_daily_samples,
            "daily_samples_expected_rows": (
                expected_daily_rows if write_daily_samples else 0
            ),
            "deterministic_release_frame_sha256": final_fingerprint,
            "artifact_release": {
                "run_id": run_id,
                "artifact_manifest_sha256": artifact_manifest_hash,
            },
            "publication_gates": publication_gates,
            "files": data_inventory,
            "checksum_policy": (
                "Every staged data/config/environment file is size- and SHA-256-bound "
                "by this manifest; RELEASE_COMMIT.json binds this manifest and is "
                "written only after schema, statistics, claims, and reproduction pass."
            ),
        }
        validate_inventory(data_stage, data_inventory)
        write_json(data_stage / "release_manifest.json", manifest)
        manifest_hash = sha256(data_stage / "release_manifest.json")
        commit = {
            "complete": True,
            "contract_version": CONTRACT_VERSION,
            "run_id": run_id,
            "release_manifest_sha256": manifest_hash,
            "artifact_manifest_sha256": artifact_manifest_hash,
            "configuration_sha256": config_digest,
            "deterministic_release_frame_sha256": final_fingerprint,
            "publication_gates": publication_gates,
        }
        write_json(data_stage / "RELEASE_COMMIT.json", commit)
        if sha256(data_stage / "release_manifest.json") != commit["release_manifest_sha256"]:
            raise RuntimeError("release commit does not bind the staged manifest")

        with publication_lock(artifact_root):
            with publication_lock(output_root):
                artifact_destination = publish_immutable_run(
                    artifact_stage, artifact_root, run_id
                )
                data_destination = publish_immutable_run(
                    data_stage, output_root, run_id
                )
                legacy_daily = output_root / "daily_stochastic_samples.csv"
                if legacy_daily.is_symlink():
                    raise RuntimeError(
                        f"refusing to delete symbolic-link legacy output: {legacy_daily}"
                    )
                if legacy_daily.is_file():
                    legacy_daily.unlink()
                write_latest_pointer(
                    artifact_root,
                    run_id,
                    artifact_destination / "RELEASE_COMMIT.json",
                    config_digest,
                )
                write_latest_pointer(
                    output_root,
                    run_id,
                    data_destination / "RELEASE_COMMIT.json",
                    config_digest,
                )
    except BaseException:
        if data_stage.exists():
            shutil.rmtree(data_stage)
        if artifact_stage.exists():
            shutil.rmtree(artifact_stage)
        raise

    if data_destination is None or artifact_destination is None:
        raise RuntimeError("release destinations were not published")
    baseline = first.analytical[
        (first.analytical["population"] == first.populations[0])
        & (first.analytical["scenario"] == "baseline")
    ].iloc[0]
    maximum_z = first.statistical_metrics["standardized_mean_error_z"].abs().max()
    print("\nAnalytical workload characterization published successfully.")
    print(f"Run ID: {run_id}")
    print(f"Data release: {data_destination}")
    print(f"Artifact release: {artifact_destination}")
    print(f"Population-scenario cases: {len(first.analytical)}")
    print(f"Daily stochastic observations evaluated: {len(first.samples):,}")
    print(
        f"{first.populations[0] / 1e6:.0f}M baseline: "
        f"{baseline['coordination_events_day']:,.2f} events/day, "
        f"{baseline['expected_tps']:.3f} average offered TPS"
    )
    print(f"Maximum absolute standardized mean error: {maximum_z:.6f}")
    print(f"Deterministic release fingerprint: {final_fingerprint}")
    print("Every publication gate passed; RELEASE_COMMIT.json is present.")

def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.write_default_config is not None:
        target = args.write_default_config.expanduser().resolve()
        validate_config(DEFAULT_CONFIG)
        write_json(target, DEFAULT_CONFIG)
        print(f"Validated default configuration written to {target}")
        return
    if args.self_test:
        run_self_tests()
        return
    execute(args)


if __name__ == "__main__":
    main()
