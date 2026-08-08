from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from src.configs.salary_predict.salary_ML_flow_config import (
    ENV_EXPERIMENT_NAME,
    ENV_TRACKING_ENABLED,
    ENV_TRACKING_URI,
    SalaryMLflowConfig,
)


# ---------------------------------------------------------------------------
# Environment isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_salary_mlflow_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Ensure every test starts without MLflow configuration overrides
    leaking in from the developer's (or CI runner's) actual environment.

    Individual environment-override tests explicitly set the variables
    they need after this fixture clears them.
    """
    monkeypatch.delenv(ENV_TRACKING_URI, raising=False)
    monkeypatch.delenv(ENV_EXPERIMENT_NAME, raising=False)
    monkeypatch.delenv(ENV_TRACKING_ENABLED, raising=False)


# ---------------------------------------------------------------------------
# Construction / defaults
# ---------------------------------------------------------------------------


def test_config_constructs_with_defaults() -> None:
    config = SalaryMLflowConfig()

    assert config.experiment_name == "salary_prediction"
    assert config.project_name == "linkedin-job-intelligence"
    assert config.pipeline_name == "salary_predict"
    assert config.run_name_prefix == "salary"
    assert config.registered_model_name == "salary_prediction_model"
    assert config.artifact_location is None
    assert config.max_parameter_length == 250
    assert config.tracking_enabled is True


def test_default_tracking_uri_is_local_file_uri_and_exists_logically() -> None:
    config = SalaryMLflowConfig()

    assert config.tracking_uri.startswith("file://")
    assert config.is_local_tracking is True
    assert isinstance(config.local_tracking_path, Path)
    # "Exists logically" -- it's a well-formed absolute path under the
    # project root, not that the directory has been created on disk.
    assert config.local_tracking_path.is_absolute()
    assert config.local_tracking_path.name == "mlruns"


def test_default_tracking_uri_resolves_under_project_root() -> None:
    config = SalaryMLflowConfig()

    # src/configs/salary_predict/ -> project root is 3 parents up from
    # this module's file, matching the config's own _project_root().
    project_root = Path(__file__).resolve().parents[2]
    assert config.local_tracking_path == (project_root / "mlruns").resolve()


# ---------------------------------------------------------------------------
# Tags / run naming
# ---------------------------------------------------------------------------


def test_default_tags_contain_project_and_pipeline() -> None:
    config = SalaryMLflowConfig(project_name="proj-x", pipeline_name="pipe-y")

    tags = config.default_tags()
    assert tags == {"project": "proj-x", "pipeline": "pipe-y"}


def test_default_tags_returns_a_fresh_dict_each_call() -> None:
    config = SalaryMLflowConfig()

    tags_a = config.default_tags()
    tags_b = config.default_tags()
    assert tags_a == tags_b
    assert tags_a is not tags_b  # mutating one must not affect the other


def test_build_run_name_uses_configured_prefix() -> None:
    config = SalaryMLflowConfig(run_name_prefix="salary")
    assert config.build_run_name("E3A") == "salary-E3A"


def test_build_run_name_rejects_blank_suffix() -> None:
    config = SalaryMLflowConfig()
    with pytest.raises(ValueError):
        config.build_run_name("   ")


def test_build_run_name_strips_suffix_whitespace() -> None:
    config = SalaryMLflowConfig()
    assert config.build_run_name("  E3A  ") == "salary-E3A"


# ---------------------------------------------------------------------------
# tracking_enabled helper
# ---------------------------------------------------------------------------


def test_tracking_enabled_helper() -> None:
    enabled = SalaryMLflowConfig(tracking_enabled=True)
    disabled = SalaryMLflowConfig(tracking_enabled=False)

    assert enabled.is_tracking_enabled is True
    assert disabled.is_tracking_enabled is False


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_config_is_frozen() -> None:
    config = SalaryMLflowConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.experiment_name = "something_else"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Validation: blank/empty required fields & types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field_name, bad_value",
    [
        ("experiment_name", ""),
        ("experiment_name", "   "),
        ("project_name", ""),
        ("pipeline_name", ""),
        ("run_name_prefix", ""),
        ("registered_model_name", ""),
    ],
)
def test_blank_required_fields_are_rejected(field_name: str, bad_value: str) -> None:
    with pytest.raises(ValueError):
        SalaryMLflowConfig(**{field_name: bad_value})


@pytest.mark.parametrize(
    "bad_uri",
    [
        "",
        "   ",
        "://missing-scheme",
        "http://",
        "https://",
        "file://",
    ],
)
def test_invalid_tracking_uris_are_rejected(bad_uri: str) -> None:
    with pytest.raises(ValueError):
        SalaryMLflowConfig(tracking_uri=bad_uri)


def test_tracking_uri_must_be_string() -> None:
    with pytest.raises(TypeError):
        SalaryMLflowConfig(tracking_uri=123)  # type: ignore[arg-type]


def test_blank_artifact_location_is_rejected_but_none_is_allowed() -> None:
    # None (the default) must be fine -- MLflow decides the default.
    SalaryMLflowConfig(artifact_location=None)

    with pytest.raises(ValueError):
        SalaryMLflowConfig(artifact_location="   ")


def test_artifact_location_must_be_string_or_none() -> None:
    with pytest.raises(TypeError):
        SalaryMLflowConfig(artifact_location=123)  # type: ignore[arg-type]


def test_non_bool_tracking_enabled_is_rejected() -> None:
    with pytest.raises(TypeError):
        SalaryMLflowConfig(tracking_enabled="yes")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_value", [0, -1, -100])
def test_invalid_max_parameter_length(bad_value: int) -> None:
    with pytest.raises(ValueError):
        SalaryMLflowConfig(max_parameter_length=bad_value)


def test_non_integer_max_parameter_length_is_rejected() -> None:
    with pytest.raises(TypeError):
        SalaryMLflowConfig(max_parameter_length="250")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Remote / non-local URIs
# ---------------------------------------------------------------------------


def test_remote_http_tracking_uri_is_accepted_and_not_local() -> None:
    config = SalaryMLflowConfig(tracking_uri="https://mlflow.example.com")

    assert config.tracking_uri == "https://mlflow.example.com"
    assert config.is_local_tracking is False
    assert config.local_tracking_path is None


def test_bare_local_tracking_path_is_recognized_as_local(tmp_path: Path) -> None:
    local_store = tmp_path / "mlruns"

    config = SalaryMLflowConfig(tracking_uri=str(local_store))

    assert config.is_local_tracking is True
    assert config.local_tracking_path is not None
    assert config.local_tracking_path.resolve() == local_store.resolve()


# ---------------------------------------------------------------------------
# Environment-variable overrides
# ---------------------------------------------------------------------------


def test_tracking_uri_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_TRACKING_URI, "http://localhost:5000")

    config = SalaryMLflowConfig()

    assert config.tracking_uri == "http://localhost:5000"
    assert config.is_local_tracking is False


def test_experiment_name_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_EXPERIMENT_NAME, "salary_prediction_ci")

    config = SalaryMLflowConfig()

    assert config.experiment_name == "salary_prediction_ci"


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        ("true", True),
        ("True", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
        ("off", False),
    ],
)
def test_tracking_enabled_env_override_accepts_common_boolean_spellings(
    monkeypatch: pytest.MonkeyPatch, raw_value: str, expected: bool
) -> None:
    monkeypatch.setenv(ENV_TRACKING_ENABLED, raw_value)

    config = SalaryMLflowConfig()

    assert config.tracking_enabled is expected


def test_tracking_enabled_env_override_rejects_unrecognized_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_TRACKING_ENABLED, "definitely-not-a-bool")

    with pytest.raises(ValueError):
        SalaryMLflowConfig()


def test_env_overrides_are_evaluated_at_construction_not_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Construct once with no override -> local default.
    config_before = SalaryMLflowConfig()
    assert config_before.is_local_tracking is True

    # Setting the env var AFTER import (module already loaded above) must
    # still affect a NEW instance, proving default_factory evaluates at
    # construction time, not at class-definition/import time.
    monkeypatch.setenv(ENV_TRACKING_URI, "http://localhost:5000")
    config_after = SalaryMLflowConfig()
    assert config_after.tracking_uri == "http://localhost:5000"
    assert config_after.is_local_tracking is False

    # And the earlier instance is untouched (frozen, constructed before
    # the override was set).
    assert config_before.is_local_tracking is True