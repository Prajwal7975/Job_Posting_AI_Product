"""
src/configs/salary_predict/salary_experiment_config.py

Salary Experiment Configuration.

This module answers ONE question: "WHAT should a given salary-model
experiment be?" It does not answer "HOW do we run it?" — that is the job
of future components (SalaryPreprocessorBuilder, SalaryModelFactory,
SalaryModelTrainer, SalaryModelEvaluator, salary_experiment_runner.py).

This module intentionally has ZERO dependency on pandas, numpy, sklearn,
mlflow, or joblib. It is a pure, standard-library configuration contract:
dataclasses, enums, validation, deterministic serialization, and a
deterministic behavior-fingerprint ("config_signature").

Nothing in this file trains a model, reads a dataset, fits a transformer,
computes a metric, or writes an artifact.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple


# ======================================================================
# ENUMS
# ======================================================================


class ModelName(str, Enum):
    """
    Stable model identifiers. Adding a new algorithm (e.g. ElasticNet)
    means adding a member here AND teaching the future SalaryModelFactory
    how to build it — this enum alone does not instantiate anything.
    """

    DUMMY = "dummy"
    RIDGE = "ridge"


class SkillEncoding(str, Enum):
    """
    How `skill_list` should be represented, if at all. This is a switch,
    not a feature-list entry — `skill_list` must never appear literally
    inside `categorical_features`/`numeric_features` (see __post_init__).
    """

    NONE = "none"
    TFIDF = "tfidf"
    MULTIHOT = "multihot"


# ======================================================================
# NESTED CONFIG: TF-IDF SETTINGS
# ======================================================================


@dataclass(frozen=True)
class TfidfConfig:
    """
    Declarative TF-IDF settings. Never instantiates sklearn's
    TfidfVectorizer — the future SalaryPreprocessorBuilder reads these
    fields and builds the real transformer.
    """

    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 5
    max_df: float = 0.95
    sublinear_tf: bool = True
    max_features: Optional[int] = None
    strip_accents: Optional[str] = None
    lowercase: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.ngram_range, tuple)
            or len(self.ngram_range) != 2
            or not all(isinstance(n, int) and not isinstance(n, bool) for n in self.ngram_range)
        ):
            raise ValueError(
                "ngram_range must be a tuple containing exactly two integers."
            )

        n_min, n_max = self.ngram_range

        if n_min < 1 or n_max < 1:
            raise ValueError("ngram_range values must be >= 1.")

        if n_min > n_max:
            raise ValueError(
                "ngram_range lower bound must not exceed upper bound."
            )

        if not isinstance(self.min_df, int) or isinstance(self.min_df, bool):
            raise ValueError("min_df must be an integer.")

        if self.min_df < 1:
            raise ValueError("min_df must be >= 1.")

        if (
            not isinstance(self.max_df, (int, float))
            or isinstance(self.max_df, bool)
            or not (0 < self.max_df <= 1.0)
        ):
            raise ValueError(
                "max_df must be numeric and satisfy 0 < max_df <= 1.0."
            )

        if self.max_features is not None:
            if (
                not isinstance(self.max_features, int)
                or isinstance(self.max_features, bool)
                or self.max_features <= 0
            ):
                raise ValueError(
                    "max_features must be a positive integer if provided."
                )

        if self.strip_accents not in (None, "ascii", "unicode"):
            raise ValueError(
                "strip_accents must be None, 'ascii', or 'unicode'."
            )

        if not isinstance(self.sublinear_tf, bool):
            raise ValueError("sublinear_tf must be a boolean.")

        if not isinstance(self.lowercase, bool):
            raise ValueError("lowercase must be a boolean.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ngram_range": list(self.ngram_range),
            "min_df": self.min_df,
            "max_df": self.max_df,
            "sublinear_tf": self.sublinear_tf,
            "max_features": self.max_features,
            "strip_accents": self.strip_accents,
            "lowercase": self.lowercase,
        }


# ======================================================================
# FEATURE GROUP CONSTANTS
# (established by the preprocessing audit — see spec sections 7-18)
# ======================================================================

TEXT_FEATURE_TITLE = "title"
TEXT_FEATURE_SKILL_LIST = "skill_list"

CORE_CATEGORICAL_FEATURES: Tuple[str, ...] = (
    "formatted_experience_level",
    "company_state",
    "company_country",
    "top_industry",
)

OPTIONAL_CATEGORICAL_FEATURES: Tuple[str, ...] = (
    "formatted_work_type",
    "company_size",
)

# top_skill is neither "core" nor "optional" in the product-input sense —
# it is specifically an ablation feature tested by E5, kept in its own
# group so uses_optional_features doesn't accidentally fire for it.
ABLATION_CATEGORICAL_FEATURES: Tuple[str, ...] = ("top_skill",)

CORE_NUMERIC_FEATURES: Tuple[str, ...] = ("skill_count",)

OPTIONAL_NUMERIC_FEATURES: Tuple[str, ...] = ("log_company_employee_count",)

EXPERIMENTAL_NUMERIC_FEATURES: Tuple[str, ...] = ("log_company_follower_count",)

ALL_KNOWN_CATEGORICAL_FEATURES: FrozenSet[str] = frozenset(
    CORE_CATEGORICAL_FEATURES + OPTIONAL_CATEGORICAL_FEATURES + ABLATION_CATEGORICAL_FEATURES
)

ALL_KNOWN_NUMERIC_FEATURES: FrozenSet[str] = frozenset(
    CORE_NUMERIC_FEATURES + OPTIONAL_NUMERIC_FEATURES + EXPERIMENTAL_NUMERIC_FEATURES
)

# Columns that must NEVER become model predictors through this config,
# regardless of experiment. This includes the targets themselves, raw
# salary/target-construction leakage fields, split/lineage metadata, and
# free-text/date metadata. See spec section 31.
PROTECTED_NON_PREDICTOR_COLUMNS: FrozenSet[str] = frozenset(
    {
        # targets
        "target_log_salary",
        "target_annual_salary",
        # split / lineage metadata
        "posting_group_id",
        "dataset_version",
        "salary_target_source",
        "company_name",
        # raw salary / target-construction leakage
        "min_salary",
        "med_salary",
        "max_salary",
        "normalized_salary",
        "listed_min_salary",
        "listed_med_salary",
        "listed_max_salary",
        "salary_available",
        "pay_period",
        "currency",
        # metadata-only columns (never predictors)
        "location",
        "original_listed_time",
        "listed_time",
        # dropped/non-predictor duplicates of engineered numeric features
        "company_employee_count",
        "company_follower_count",
    }
)

CONFIG_VERSION = "1.0"


# ======================================================================
# MAIN EXPERIMENT CONFIG
# ======================================================================


@dataclass(frozen=True)
class SalaryExperimentConfig:
    """
    One complete, immutable definition of a salary-model experiment.
    """

    # ---- identity -----------------------------------------------------
    experiment_id: str
    experiment_name: str
    description: str

    # ---- model ----------------------------------------------------------
    model_name: str
    model_params: Mapping[str, Any] = field(default_factory=dict)

    # ---- targets --------------------------------------------------------
    training_target_col: str = "target_log_salary"
    annual_target_col: str = "target_annual_salary"

    # ---- text / NLP switches --------------------------------------------
    use_title: bool = False
    title_tfidf: Optional[TfidfConfig] = None

    skill_encoding: SkillEncoding = SkillEncoding.NONE
    skill_tfidf: Optional[TfidfConfig] = None

    # ---- structured feature groups (explicit per experiment) -----------
    categorical_features: Tuple[str, ...] = ()
    numeric_features: Tuple[str, ...] = ()

    # ---- schema version of this config object itself --------------------
    config_version: str = CONFIG_VERSION

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        try:
            model_params_copy = dict(self.model_params)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "model_params must be mapping-like and convertible to a dictionary."
            ) from exc

        if not all(isinstance(key, str) for key in model_params_copy):
            raise ValueError(
                "All model_params keys must be strings."
            )

        object.__setattr__(
            self,
            "model_params",
            MappingProxyType(model_params_copy),
        )

        for field_name, value in (
            ("experiment_id", self.experiment_id),
            ("experiment_name", self.experiment_name),
            ("description", self.description),
            ("model_name", self.model_name),
            ("training_target_col", self.training_target_col),
            ("annual_target_col", self.annual_target_col),
            ("config_version", self.config_version),
        ):
            if not isinstance(value, str):
                raise ValueError(
                    f"{field_name} must be a string, got {type(value).__name__}."
                )

        if not self.experiment_id.strip():
            raise ValueError("experiment_id must not be empty.")

        if not self.experiment_name.strip():
            raise ValueError("experiment_name must not be empty.")

        if not self.training_target_col.strip():
            raise ValueError("training_target_col must not be empty.")

        if not self.annual_target_col.strip():
            raise ValueError("annual_target_col must not be empty.")

        allowed_models = {m.value for m in ModelName}
        if self.model_name not in allowed_models:
            raise ValueError(
                f"Unsupported model_name '{self.model_name}'. "
                f"Allowed values: {sorted(allowed_models)}."
            )

        if not isinstance(self.skill_encoding, SkillEncoding):
            raise ValueError(
                f"skill_encoding must be a SkillEncoding member, got {self.skill_encoding!r}."
            )

        # E0 (dummy) must stay a pure baseline — no accidental features.
        if self.model_name == ModelName.DUMMY.value:
            if (
                self.use_title
                or self.skill_encoding != SkillEncoding.NONE
                or self.categorical_features
                or self.numeric_features
            ):
                raise ValueError(
                    "The dummy baseline must not declare any predictive features "
                    "(use_title, skill_encoding, categorical_features, "
                    "numeric_features must all be at their inactive defaults)."
                )

        # No duplicate feature names within a single group.
        if len(set(self.categorical_features)) != len(self.categorical_features):
            raise ValueError(f"Duplicate entries in categorical_features: {self.categorical_features}")
        if len(set(self.numeric_features)) != len(self.numeric_features):
            raise ValueError(f"Duplicate entries in numeric_features: {self.numeric_features}")

        # A feature cannot be both categorical and numeric.
        overlap = set(self.categorical_features) & set(self.numeric_features)
        if overlap:
            raise ValueError(
                f"Features cannot appear in both categorical_features and "
                f"numeric_features: {sorted(overlap)}"
            )

        combined = set(self.categorical_features) | set(self.numeric_features)

        # Text features are controlled ONLY via use_title / skill_encoding,
        # never smuggled into the structured feature lists.
        text_leak = combined & {TEXT_FEATURE_TITLE, TEXT_FEATURE_SKILL_LIST}
        if text_leak:
            raise ValueError(
                f"{sorted(text_leak)} must be controlled via use_title / "
                "skill_encoding, not listed in categorical_features or "
                "numeric_features."
            )

        # Every declared feature must come from a known, named feature group
        # — catches typos and prevents silently-inert features.
        unknown_categorical = set(self.categorical_features) - ALL_KNOWN_CATEGORICAL_FEATURES
        if unknown_categorical:
            raise ValueError(f"Unknown categorical_features: {sorted(unknown_categorical)}")
        unknown_numeric = set(self.numeric_features) - ALL_KNOWN_NUMERIC_FEATURES
        if unknown_numeric:
            raise ValueError(f"Unknown numeric_features: {sorted(unknown_numeric)}")

        # Leakage safety net — protected columns can never be predictors.
        leaked = combined & PROTECTED_NON_PREDICTOR_COLUMNS
        if leaked:
            raise ValueError(
                f"Protected/leakage columns must never be predictors: {sorted(leaked)}"
            )

        # Validate contradictory NLP configuration states
        if not self.use_title and self.title_tfidf is not None:
            raise ValueError("title_tfidf MUST be None when use_title is False.")
        if self.use_title and self.title_tfidf is None:
            object.__setattr__(self, "title_tfidf", TfidfConfig())

        if self.skill_encoding in (SkillEncoding.NONE, SkillEncoding.MULTIHOT):
            if self.skill_tfidf is not None:
                raise ValueError(f"skill_tfidf MUST be None when skill_encoding is {self.skill_encoding.value}.")
        elif self.skill_encoding == SkillEncoding.TFIDF:
            if self.skill_tfidf is None:
                object.__setattr__(self, "skill_tfidf", TfidfConfig())

    # ------------------------------------------------------------------
    # Derived, read-only feature resolution
    # ------------------------------------------------------------------
    @property
    def active_text_features(self) -> Tuple[str, ...]:
        features: List[str] = []
        if self.use_title:
            features.append(TEXT_FEATURE_TITLE)
        if self.skill_encoding != SkillEncoding.NONE:
            features.append(TEXT_FEATURE_SKILL_LIST)
        return tuple(features)

    @property
    def active_categorical_features(self) -> Tuple[str, ...]:
        return self.categorical_features

    @property
    def active_numeric_features(self) -> Tuple[str, ...]:
        return self.numeric_features

    @property
    def active_predictor_features(self) -> Tuple[str, ...]:
        """
        Stable-order union of every active predictor: text features first
        (title, then skill_list), then categorical, then numeric — in the
        order each group was declared. Never derived from an unordered set.
        """
        return self.active_text_features + self.active_categorical_features + self.active_numeric_features

    @property
    def has_text_features(self) -> bool:
        return bool(self.active_text_features)

    @property
    def uses_skills(self) -> bool:
        return self.skill_encoding != SkillEncoding.NONE

    @property
    def uses_optional_features(self) -> bool:
        optional_categorical = set(self.categorical_features) & set(OPTIONAL_CATEGORICAL_FEATURES)
        optional_numeric = set(self.numeric_features) & set(OPTIONAL_NUMERIC_FEATURES)
        return bool(optional_categorical or optional_numeric)

    @property
    def uses_experimental_features(self) -> bool:
        return bool(set(self.numeric_features) & set(EXPERIMENTAL_NUMERIC_FEATURES))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Full, JSON-serializable record of this experiment — identity + behavior."""
        return {
            "config_version": self.config_version,
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "description": self.description,
            "model_name": self.model_name,
            "model_params": dict(self.model_params),
            "training_target_col": self.training_target_col,
            "annual_target_col": self.annual_target_col,
            "use_title": self.use_title,
            "title_tfidf": self.title_tfidf.to_dict() if self.title_tfidf is not None else None,
            "skill_encoding": self.skill_encoding.value,
            "skill_tfidf": self.skill_tfidf.to_dict() if self.skill_tfidf is not None else None,
            "categorical_features": list(self.categorical_features),
            "numeric_features": list(self.numeric_features),
            "active_predictor_features": list(self.active_predictor_features),
        }

    def _signature_payload(self) -> Dict[str, Any]:
        """
        Behavior-only subset used for config_signature. Deliberately
        excludes experiment_id/experiment_name/description — renaming an
        experiment must not change its signature, only changing what it
        actually DOES should.
        """
        return {
            "config_version": self.config_version,
            "model_name": self.model_name,
            "model_params": dict(self.model_params),
            "training_target_col": self.training_target_col,
            "annual_target_col": self.annual_target_col,
            "use_title": self.use_title,
            "title_tfidf": self.title_tfidf.to_dict() if (self.use_title and self.title_tfidf) else None,
            "skill_encoding": self.skill_encoding.value,
            "skill_tfidf": (
                self.skill_tfidf.to_dict()
                if (self.skill_encoding == SkillEncoding.TFIDF and self.skill_tfidf)
                else None
            ),
            "categorical_features": list(self.categorical_features),
            "numeric_features": list(self.numeric_features),
        }

    @property
    def config_signature(self) -> str:
        """
        Deterministic SHA-256 fingerprint of everything that affects model
        behavior. No timestamps, no paths, no runtime-specific data —
        same behavior-affecting config always produces the same signature.
        """
        canonical = json.dumps(self._signature_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ======================================================================
# FEATURE-LIST HELPER (order-preserving, no accidental duplicates)
# ======================================================================


def _append_unique(base: Tuple[str, ...], additions: Iterable[str]) -> Tuple[str, ...]:
    """Append items from `additions` onto `base`, preserving order, skipping
    anything already present. Never uses set() for the output ordering."""
    result: List[str] = list(base)
    for item in additions:
        if item not in result:
            result.append(item)
    return tuple(result)


# ======================================================================
# PREDEFINED (INITIAL) EXPERIMENT BUILDERS — E0 through E3B
# ======================================================================
#
# These are the only experiments whose architecture is knowable up front.
# They do not depend on any prior experiment's results.


def build_e0_config() -> SalaryExperimentConfig:
    """Naive median baseline. Establishes the benchmark every later
    experiment must beat to justify its added complexity."""
    return SalaryExperimentConfig(
        experiment_id="E0",
        experiment_name="dummy_median_baseline",
        description="DummyRegressor(strategy='median') baseline with no predictive features.",
        model_name=ModelName.DUMMY.value,
        model_params={"strategy": "median"},
    )


def build_e1_config() -> SalaryExperimentConfig:
    """Structured features only, no text NLP at all."""
    return SalaryExperimentConfig(
        experiment_id="E1",
        experiment_name="structured_ridge",
        description="Core structured categorical/numeric features only, Ridge regression, no NLP.",
        model_name=ModelName.RIDGE.value,
        model_params={"alpha": 1.0},
        categorical_features=CORE_CATEGORICAL_FEATURES,
        numeric_features=CORE_NUMERIC_FEATURES,
    )


def build_e2_config() -> SalaryExperimentConfig:
    """E1 plus job-title TF-IDF."""
    return SalaryExperimentConfig(
        experiment_id="E2",
        experiment_name="title_tfidf_ridge",
        description="Structured core features plus TF-IDF job title using Ridge regression.",
        model_name=ModelName.RIDGE.value,
        model_params={"alpha": 1.0},
        use_title=True,
        categorical_features=CORE_CATEGORICAL_FEATURES,
        numeric_features=CORE_NUMERIC_FEATURES,
    )


def build_e3a_config() -> SalaryExperimentConfig:
    """E2 plus skill_list represented via TF-IDF."""
    return SalaryExperimentConfig(
        experiment_id="E3A",
        experiment_name="title_skills_tfidf_ridge",
        description="E2 plus skill_list represented as TF-IDF.",
        model_name=ModelName.RIDGE.value,
        model_params={"alpha": 1.0},
        use_title=True,
        skill_encoding=SkillEncoding.TFIDF,
        categorical_features=CORE_CATEGORICAL_FEATURES,
        numeric_features=CORE_NUMERIC_FEATURES,
    )


def build_e3b_config() -> SalaryExperimentConfig:
    """E2 plus skill_list represented via sparse multi-hot encoding."""
    return SalaryExperimentConfig(
        experiment_id="E3B",
        experiment_name="title_skills_multihot_ridge",
        description="E2 plus skill_list represented as sparse multi-hot/token encoding.",
        model_name=ModelName.RIDGE.value,
        model_params={"alpha": 1.0},
        use_title=True,
        skill_encoding=SkillEncoding.MULTIHOT,
        categorical_features=CORE_CATEGORICAL_FEATURES,
        numeric_features=CORE_NUMERIC_FEATURES,
    )


# ======================================================================
# DERIVED EXPERIMENT BUILDERS — E4, E5, E6
# ======================================================================
#
# E4/E5/E6 cannot be predefined: their correct base architecture is
# whichever of E3A/E3B (or later, E4) actually won on validation metrics,
# and that is only known AFTER those experiments run. These builders take
# the SELECTED prior config as input and derive a new, independent config
# from it via dataclasses.replace — the parent is never mutated, and
# nothing here hard-codes "E3B wins" or similar unearned conclusions.


def build_e4_config(
    selected_base: SalaryExperimentConfig,
    experiment_id: str = "E4",
    experiment_name: str = "optional_feature_expansion",
) -> SalaryExperimentConfig:
    """
    Add the optional feature expansion (work type, company size, log
    employee count) on top of whichever architecture won E3A vs E3B.
    Preserves the base's model family, model params, TF-IDF settings, and
    skill encoding untouched — only the feature set changes, so any
    metric delta can be attributed to these features alone.
    """
    return replace(
        selected_base,
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        description=(
            f"{selected_base.experiment_id} ({selected_base.experiment_name}) "
            "plus optional feature expansion (work type, company size, "
            "log employee count)."
        ),
        categorical_features=_append_unique(selected_base.categorical_features, OPTIONAL_CATEGORICAL_FEATURES),
        numeric_features=_append_unique(selected_base.numeric_features, OPTIONAL_NUMERIC_FEATURES),
    )


def build_e5_config(
    selected_base: SalaryExperimentConfig,
    experiment_id: str = "E5",
    experiment_name: str = "top_skill_ablation",
) -> SalaryExperimentConfig:
    """
    Answers exactly one question: does `top_skill` add value on top of
    the selected architecture? Adds only that single feature — nothing
    else changes, so the comparison stays interpretable.
    """
    return replace(
        selected_base,
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        description=(
            f"{selected_base.experiment_id} ({selected_base.experiment_name}) "
            "plus top_skill ablation."
        ),
        categorical_features=_append_unique(selected_base.categorical_features, ABLATION_CATEGORICAL_FEATURES),
    )


def build_e6_config(
    selected_base: SalaryExperimentConfig,
    experiment_id: str = "E6",
    experiment_name: str = "follower_count_ablation",
) -> SalaryExperimentConfig:
    """
    Answers exactly one question: does `log_company_follower_count` add
    value? This is EXPERIMENTAL, not optional — even a metric improvement
    may not justify requiring this as a product input, since it implies
    the caller must supply/look up a company's follower count at
    inference time.
    """
    return replace(
        selected_base,
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        description=(
            f"{selected_base.experiment_id} ({selected_base.experiment_name}) "
            "plus log_company_follower_count ablation."
        ),
        numeric_features=_append_unique(selected_base.numeric_features, EXPERIMENTAL_NUMERIC_FEATURES),
    )


# ======================================================================
# REGISTRY
# ======================================================================

_INITIAL_EXPERIMENT_BUILDERS: Dict[str, Callable[[], SalaryExperimentConfig]] = {
    "E0": build_e0_config,
    "E1": build_e1_config,
    "E2": build_e2_config,
    "E3A": build_e3a_config,
    "E3B": build_e3b_config,
}


def get_initial_experiment_configs() -> Tuple[SalaryExperimentConfig, ...]:
    """
    The full set of experiments whose architecture is knowable without
    any prior result: E0 through E3B. E4/E5/E6 are intentionally absent —
    build them with build_e4_config()/build_e5_config()/build_e6_config()
    once a winning base architecture has actually been selected.
    """
    return tuple(builder() for builder in _INITIAL_EXPERIMENT_BUILDERS.values())


def get_experiment_config(experiment_id: str) -> SalaryExperimentConfig:
    """Retrieve one of the initial (non-derived) experiment configs by id."""
    builder = _INITIAL_EXPERIMENT_BUILDERS.get(experiment_id)
    if builder is None:
        raise KeyError(
            f"Unknown initial experiment_id '{experiment_id}'. "
            f"Available: {sorted(_INITIAL_EXPERIMENT_BUILDERS)}. "
            "E4/E5/E6 are derived, not registered — use build_e4_config()/"
            "build_e5_config()/build_e6_config() with a selected base config."
        )
    return builder()