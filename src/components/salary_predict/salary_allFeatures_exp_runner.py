from __future__ import annotations
import shutil
import csv
import dataclasses
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.components.salary_predict.salary_mlflow_tracker import SalaryMLflowTracker
from src.logger import logging
from src.exception import CustomException
from src.configs.salary_predict.salary_experiment_config import (
    SalaryExperimentConfig,
    get_experiment_config,
    get_initial_experiment_configs,
)
from src.components.salary_predict.salary_single_experiment_runner import (
    SalaryTrainingRunner,
    SalaryTrainingResult,
)

# Re-exported so downstream code (including the smoke-test blueprint that
# imports these two names FROM this module) resolves without needing to
# separately import salary_training_runner.
__all__ = [
    "SalaryTrainingRunner",
    "SalaryTrainingResult",
    "ReportArtifacts",
    "FeatureExperimentSummary",
    "SalaryFeatureExperimentReportWriter",
    "SalaryFeatureExperimentRunner",
]


# ======================================================================
# RANKING METRIC DEFINITIONS
# ======================================================================
#
# direction: +1 means "lower is better" (sort ascending as-is),
#            -1 means "higher is better" (negate before sorting ascending).
# key: which key inside SalaryTrainingResult.annual_metrics this reads.

_RANKING_METRICS: Dict[str, Tuple[str, int]] = {
    "annual_r2": ("r2", -1),
    "annual_mae": ("mae", 1),
    "annual_rmse": ("rmse", 1),
    "median_ape": ("median_ape", 1),
}

# Fixed tie-break chain per the approved default ranking strategy. Whatever
# the configured primary `ranking_metric` is, it leads; the remaining
# metrics below (minus the primary, to avoid comparing the same value
# twice) follow in this order, then training time, then experiment_id.
_DEFAULT_TIE_BREAK_ORDER: Tuple[str, ...] = (
    "annual_r2",
    "annual_mae",
    "annual_rmse",
    "median_ape",
)


# ======================================================================
# INTERNAL PER-EXPERIMENT OUTCOME (success or failure)
# ======================================================================


@dataclass(frozen=True)
class _ExperimentOutcome:
    experiment_id: str
    success: bool
    result: Optional[SalaryTrainingResult]
    error: Optional[str]
    elapsed_seconds: float


# ======================================================================
# REPORT ARTIFACTS CONTAINER
# ======================================================================


@dataclass(frozen=True)
class ReportArtifacts:
    summary_json_path: Path
    summary_csv_path: Path
    winner_config_path: Path
    metadata_path: Path
    report_dir: Path


# ======================================================================
# SUMMARY / REPORT
# ======================================================================


@dataclass(frozen=True)
class FeatureExperimentSummary:

    all_results: Tuple[SalaryTrainingResult, ...]
    ranked_results: Tuple[SalaryTrainingResult, ...]
    winner: SalaryTrainingResult
    winner_config: SalaryExperimentConfig
    winner_metrics: Dict[str, float]
    winner_score: float
    comparison_table: Tuple[Dict[str, Any], ...]
    ranking_metric: str
    experiment_count: int
    execution_seconds: float
    generated_at: datetime
    ranked_experiment_ids: Tuple[str, ...]
    report_artifacts: Optional[ReportArtifacts] = None
    failures: Dict[str, str] = field(default_factory=dict)
    winner_rank: int = 1  # winner is always rank 1 by construction; kept
    # explicit (rather than implied) so downstream consumers never need to
    # assume it, and so a future ranking strategy that overrides the
    # top-metric pick for business reasons has somewhere honest to record
    # that the "winner" wasn't necessarily rank 1.
    report_version: int = 1

    # ------------------------------------------------------------------
    # Convenience properties & Consumption helpers
    # ------------------------------------------------------------------
    @property
    def best_experiment_id(self) -> str:
        return self.winner.experiment_id

    @property
    def best_model_name(self) -> str:
        return self.winner.model_name

    @property
    def best_feature_count(self) -> int:
        return self.winner.raw_feature_count

    @property
    def best_raw_features(self) -> Tuple[str, ...]:
        return self.winner.raw_feature_columns

    @property
    def best_transformed_features(self) -> Optional[int]:
        return self.winner.transformed_feature_count

    @property
    def best_metrics(self) -> Dict[str, float]:
        return dict(self.winner_metrics)

    @property
    def best_features(self) -> Tuple[str, ...]:
        return self.winner.raw_feature_columns

    @property
    def best_model(self) -> str:
        return self.winner.model_name

    @property
    def best_config(self) -> SalaryExperimentConfig:
        return self.winner_config

    @property
    def failed_experiments(self) -> Dict[str, str]:
        return dict(self.failures)

    @property
    def successful_experiments(self) -> Tuple[str, ...]:
        return tuple(r.experiment_id for r in self.ranked_results)

    def to_markdown(self) -> str:
        lines = [
            "# Feature Experiment Summary",
            f"- **Generated At**: {self.generated_at.isoformat()}",
            f"- **Ranking Metric**: {self.ranking_metric}",
            f"- **Total Experiments**: {self.experiment_count}",
            f"- **Execution Time**: {self.execution_seconds:.2f} seconds",
            "",
            "## Winner",
            f"- **Experiment ID**: {self.best_experiment_id}",
            f"- **Model**: {self.best_model_name}",
            f"- **Score ({self.ranking_metric})**: {self.winner_score:.4f}",
            "",
            "## Leaderboard",
            "| Rank | Experiment ID | Model | Annual MAE | Annual RMSE | R² | Median APE |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for rank, row in enumerate(self.comparison_table, start=1):
            mae = row.get("annual_mae")
            rmse = row.get("annual_rmse")
            r2 = row.get("annual_r2")
            mape = row.get("median_ape")

            mae_str = f"{mae:.2f}" if mae is not None and math.isfinite(mae) else "n/a"
            rmse_str = (
                f"{rmse:.2f}" if rmse is not None and math.isfinite(rmse) else "n/a"
            )
            r2_str = f"{r2:.4f}" if r2 is not None and math.isfinite(r2) else "n/a"
            mape_str = (
                f"{mape:.2f}%" if mape is not None and math.isfinite(mape) else "n/a"
            )

            lines.append(
                f"| {rank} | {row.get('experiment_id')} | {row.get('model_name')} | "
                f"{mae_str} | {rmse_str} | {r2_str} | {mape_str} |"
            )

        if self.failures:
            lines.extend(["", "## Failures"])
            for eid, err in self.failures.items():
                lines.append(f"- **{eid}**: {err}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialization — explicit types only, no hidden default=str
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_version": self.report_version,
            "winner": str(self.winner.experiment_id),
            "winner_rank": int(self.winner_rank),
            "ranking_metric": str(self.ranking_metric),
            "winner_score": float(self.winner_score),
            "winner_metrics": {
                str(k): float(v) for k, v in self.winner_metrics.items()
            },
            "generated_at": self.generated_at.isoformat(),
            "experiment_count": int(self.experiment_count),
            "execution_seconds": round(float(self.execution_seconds), 4),
            "ranked_experiment_ids": [str(eid) for eid in self.ranked_experiment_ids],
            "experiments": [dict(row) for row in self.comparison_table],
            "failures": {str(k): str(v) for k, v in self.failures.items()},
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ======================================================================
# REPORT WRITER COMPONENT
# ======================================================================


class SalaryFeatureExperimentReportWriter:
    """
    Dedicated reporting component responsible for persisting reports,
    comparison tables, and winner configuration artifacts, and safely
    uploading them to MLflow without letting reporting failures crash
    the main training pipeline.
    """

    def __init__(
        self,
        report_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        
        
        default_report_dir = Path("artifacts") / "reports" / "feature_experiments"
        self.report_dir = (
            Path(report_dir) if report_dir is not None else default_report_dir
        )

    def write_reports(
        self,
        summary: FeatureExperimentSummary,
        winner_config: SalaryExperimentConfig,
    ) -> Optional[ReportArtifacts]:

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            base_report_dir = self.report_dir
            run_report_dir = base_report_dir / f"run_{timestamp}"
            latest_report_dir = base_report_dir / "latest"

        # Create versioned report directory
            run_report_dir.mkdir(parents=True, exist_ok=False)

            json_path = run_report_dir / "feature_summary.json"
            csv_path = run_report_dir / "feature_summary.csv"
            winner_config_path = run_report_dir / "winner_config.json"
            metadata_path = run_report_dir / "run_metadata.json"

        # ------------------------------------------------------------------
        # Write Feature Summary JSON
        # ------------------------------------------------------------------
            json_path.write_text(summary.to_json(), encoding="utf-8")

        # ------------------------------------------------------------------
        # Write Feature Comparison CSV
        # ------------------------------------------------------------------
            header, rows = self._build_csv_rows(summary.comparison_table)

            with csv_path.open("w", newline="", encoding="utf-8",) as f:
                writer = csv.writer(f)

                if header:
                    writer.writerow(header)
                    writer.writerows(rows)

        # ------------------------------------------------------------------
        # Write Winner Configuration
        # ------------------------------------------------------------------
            winner_config_path.write_text(
            json.dumps(
                winner_config.to_dict(),
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        # ------------------------------------------------------------------
        # Write Run Metadata
        # ------------------------------------------------------------------
            metadata = {
                "timestamp": timestamp,
                "winner_experiment": summary.best_experiment_id,
                "winner_model": summary.best_model_name,
                "winner_score": summary.winner_score,
                "ranking_metric": summary.ranking_metric,
                "total_experiments": summary.experiment_count,
                "successful_experiments": len(summary.ranked_results),
                "failed_experiments": len(summary.failures),
                }

            metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
            ),
            encoding="utf-8",
        )

        # ------------------------------------------------------------------
        # Update latest/
        # ------------------------------------------------------------------
            temp_latest = base_report_dir / "_latest_tmp"

            if temp_latest.exists():
                shutil.rmtree(temp_latest)

            shutil.copytree(run_report_dir, temp_latest)

            if latest_report_dir.exists():
                shutil.rmtree(latest_report_dir)

            temp_latest.rename(latest_report_dir)

            logging.info(
            "Reports successfully saved under:\n%s",
            run_report_dir.resolve(),
        )

            return ReportArtifacts(
                summary_json_path=json_path,
                summary_csv_path=csv_path,
                winner_config_path=winner_config_path,
                metadata_path=metadata_path,
                report_dir=run_report_dir,
                )

        except Exception as e:
            logging.error("Failed to write feature experiment reports.", exc_info=True,)
            
            return None

    @staticmethod
    def _build_csv_rows(
        comparison_table: Tuple[Dict[str, Any], ...],
    ) -> Tuple[Tuple[str, ...], Tuple[Tuple[Any, ...], ...]]:
        if not comparison_table:
            return (), ()
        header = tuple(comparison_table[0].keys())
        rows = tuple(tuple(row.get(col) for col in header) for row in comparison_table)
        return header, rows


# ======================================================================
# RUNNER ORCHESTRATION LAYER
# ======================================================================


class SalaryFeatureExperimentRunner:

    def __init__(
        self,
        training_runner: Optional[SalaryTrainingRunner] = None,
        mlflow_tracker: Optional[SalaryMLflowTracker] = None,
        report_writer: Optional[SalaryFeatureExperimentReportWriter] = None,
        experiment_ids: Optional[Sequence[str]] = None,
        ranking_metric: str = "annual_r2",
        auto_save_report: bool = False,
        report_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.training_runner = training_runner or SalaryTrainingRunner()
        self.mlflow_tracker = (
            mlflow_tracker if mlflow_tracker is not None else SalaryMLflowTracker()
        )

        self.report_writer = report_writer or SalaryFeatureExperimentReportWriter(report_dir=report_dir)
        
        self._explicit_experiment_ids = (
            tuple(experiment_ids) if experiment_ids is not None else None
        )
        self._validate_ranking_metric(ranking_metric)
        self.ranking_metric = ranking_metric

        self.auto_save_report = auto_save_report
        self.report_writer = report_writer or SalaryFeatureExperimentReportWriter(report_dir=report_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> FeatureExperimentSummary:
        try:
            start_time = time.perf_counter()
            configs = self._resolve_experiment_configs()

            logging.info("=" * 70)
            logging.info("FEATURE EXPERIMENT RUNNER STARTED")
            logging.info("=" * 70)
            logging.info(f"Experiments queued : {[c.experiment_id for c in configs]}")
            logging.info(f"Ranking metric     : {self.ranking_metric}")

            outcomes = [self._run_single_experiment(config) for config in configs]

            successes = [o for o in outcomes if o.success]
            failures: Dict[str, str] = {
                o.experiment_id: (o.error or "") for o in outcomes if not o.success
            }

            if not successes:
                raise RuntimeError(
                    f"All {len(outcomes)} feature experiments failed. "
                    f"Failures: {failures}"
                )

            if failures:
                logging.warning(
                    f"{len(failures)} experiment(s) failed and were excluded from ranking: {failures}"
                )

            all_results = tuple(o.result for o in successes if o.result is not None)
            ranked_results = self._rank_results(all_results)
            winner = ranked_results[0]
            winner_config = self._find_config(configs, winner.experiment_id)

            elapsed_map = {o.experiment_id: o.elapsed_seconds for o in outcomes}
            comparison_table = self._build_comparison_table(ranked_results, elapsed_map)
            ranked_experiment_ids = tuple(r.experiment_id for r in ranked_results)
            winner_score = self._raw_metric_value(winner, self.ranking_metric)

            execution_seconds = time.perf_counter() - start_time

            self._log_summary(ranked_results, winner, winner_config)

            summary = FeatureExperimentSummary(
                all_results=all_results,
                ranked_results=ranked_results,
                winner=winner,
                winner_config=winner_config,
                winner_metrics=dict(winner.annual_metrics),
                winner_score=winner_score,
                comparison_table=comparison_table,
                ranking_metric=self.ranking_metric,
                experiment_count=len(outcomes),
                execution_seconds=round(execution_seconds, 4),
                generated_at=datetime.now(timezone.utc),
                ranked_experiment_ids=ranked_experiment_ids,
                report_artifacts=None,
                failures=failures,
                winner_rank=1,
                report_version=1,
            )

            if self.auto_save_report:

                artifacts = self.report_writer.write_reports(
                    summary=summary,
                    winner_config=winner_config,
                )

                summary = dataclasses.replace(summary, report_artifacts=artifacts)

                if artifacts is not None:
                    try:
                        with self.mlflow_tracker.start_run(
                            run_name="feature-experiment-summary",
                        ):
                            self.mlflow_tracker.log_artifacts(
                                artifacts.report_dir, artifact_folder="feature_reports"
                            )

                            self.mlflow_tracker.log_tags(
                            {
                                "winner.experiment": summary.best_experiment_id,
                                "winner.model": summary.best_model_name,
                                "winner.score": str(summary.winner_score),
                                "ranking.metric": summary.ranking_metric,
                            }
                        )
                            self.mlflow_tracker.log_metrics(
                                {
                                    "winner.annual_r2": summary.winner_metrics.get("r2", float("nan")),
                                    "winner.annual_mae": summary.winner_metrics.get("mae", float("nan")),
                                    "winner.annual_rmse": summary.winner_metrics.get("rmse", float("nan")),
                                    "winner.median_ape": summary.winner_metrics.get("median_ape", float("nan")),
                                    "execution_seconds": summary.execution_seconds,
                                    "experiment_count": summary.experiment_count,
                                    }
                            )
                            
                            self.mlflow_tracker.log_params(
                                {
                                    "winner.experiment_id": winner_config.experiment_id,
                                    "winner.model_name": winner_config.model_name,
                                    "winner.feature_count": winner.raw_feature_count,
                                    "winner.config_signature": winner_config.config_signature,
                                    }
                                )
                            
                            self.mlflow_tracker.log_dict(
                                winner_config.to_dict(),
                                filename="winner_config.json",
                                artifact_folder="feature_reports",
                                )
                            self.mlflow_tracker.log_text(summary.to_markdown(), filename="summary.md", artifact_folder="feature_reports",)
                            
                    except Exception as e:
                        logging.warning("Failed to upload reports to MLflow: %s", e)

            logging.info("=" * 70)
            logging.info("FEATURE EXPERIMENT RUNNER COMPLETED")
            logging.info("=" * 70)

            return summary

        except CustomException:
            raise
        except Exception as e:
            logging.error(f"Feature experiment run failed: {e}", exc_info=True)
            raise CustomException(e, sys) from e

    # ------------------------------------------------------------------
    # Experiment resolution / validation
    # ------------------------------------------------------------------
    def _resolve_experiment_configs(self) -> Tuple[SalaryExperimentConfig, ...]:
        if self._explicit_experiment_ids is not None:
            configs = tuple(
                get_experiment_config(eid) for eid in self._explicit_experiment_ids
            )
        else:
            configs = tuple(get_initial_experiment_configs())

        if not configs:
            raise ValueError("No experiments configured to run.")

        self._validate_no_duplicate_ids(configs)
        return configs

    @staticmethod
    def _validate_no_duplicate_ids(configs: Tuple[SalaryExperimentConfig, ...]) -> None:
        seen: set = set()
        duplicates: set = set()
        for config in configs:
            if config.experiment_id in seen:
                duplicates.add(config.experiment_id)
            seen.add(config.experiment_id)
        if duplicates:
            raise ValueError(
                f"Duplicate experiment_id(s) in experiment set: {sorted(duplicates)}"
            )

    @staticmethod
    def _validate_ranking_metric(ranking_metric: str) -> None:
        if ranking_metric not in _RANKING_METRICS:
            raise ValueError(
                f"Unsupported ranking_metric '{ranking_metric}'. "
                f"Supported: {sorted(_RANKING_METRICS)}"
            )

    @staticmethod
    def _find_config(
        configs: Tuple[SalaryExperimentConfig, ...], experiment_id: str
    ) -> SalaryExperimentConfig:
        for config in configs:
            if config.experiment_id == experiment_id:
                return config
        raise ValueError(
            f"Could not find config for winning experiment_id='{experiment_id}'."
        )

    # ------------------------------------------------------------------
    # Single experiment execution — failures here never abort the batch
    # ------------------------------------------------------------------
    def _run_single_experiment(
        self, config: SalaryExperimentConfig
    ) -> _ExperimentOutcome:
        logging.info("")
        logging.info(f"Running Experiment {config.experiment_id}")
        start = time.perf_counter()
        try:
            result = self.training_runner.run(config)
            self._validate_result(config, result)
            self.mlflow_tracker.track_training_result(
                experiment_config=config, training_result=result
            )
            elapsed = time.perf_counter() - start
            logging.info("Completed")
            logging.info(
                "R²=%.4f | MAE=%.2f | RMSE=%.2f",
                result.annual_metrics["r2"],
                result.annual_metrics["mae"],
                result.annual_metrics["rmse"],
            )
            return _ExperimentOutcome(config.experiment_id, True, result, None, elapsed)
        except Exception as e:
            elapsed = time.perf_counter() - start
            logging.error(
                f"Experiment {config.experiment_id} FAILED: {e}", exc_info=True
            )
            try:
                self.mlflow_tracker.track_failed_run(experiment_config=config, error=e)
            except Exception as tracking_error:
                logging.warning(
                    f"Failed to log failed run status to MLflow: {tracking_error}"
                )
            return _ExperimentOutcome(
                config.experiment_id, False, None, str(e), elapsed
            )

    @staticmethod
    def _validate_result(
        config: SalaryExperimentConfig, result: SalaryTrainingResult
    ) -> None:
        if result is None:
            raise ValueError(f"{config.experiment_id}: training runner returned None.")
        if result.experiment_id != config.experiment_id:
            raise ValueError(
                f"Result experiment_id mismatch: expected '{config.experiment_id}', "
                f"got '{result.experiment_id}'."
            )

        required = ("mae", "rmse", "r2")
        for name in required:
            if name not in result.annual_metrics:
                raise ValueError(
                    f"{config.experiment_id}: annual_metrics missing '{name}'."
                )
            value = result.annual_metrics[name]
            if value is None or not math.isfinite(value):
                raise ValueError(
                    f"{config.experiment_id}: annual_metrics['{name}'] is not finite ({value!r})."
                )

        # median_ape is allowed to legitimately be NaN (all-zero-salary
        # edge case, defended against upstream) — warn, don't fail the
        # whole experiment over a diagnostic-only tie-breaker metric.
        median_ape = result.annual_metrics.get("median_ape")
        if median_ape is not None and not math.isfinite(median_ape):
            logging.warning(
                f"{config.experiment_id}: median_ape is not finite ({median_ape!r}); "
                "will be treated as worst-case for ranking purposes."
            )

    # ------------------------------------------------------------------
    # Ranking — single dedicated method, replaceable strategy
    # ------------------------------------------------------------------
    def _rank_results(
        self, results: Tuple[SalaryTrainingResult, ...]
    ) -> Tuple[SalaryTrainingResult, ...]:
        logging.info("Ranking metric : %s", self.ranking_metric)
        ordered_metrics = [self.ranking_metric] + [
            m for m in _DEFAULT_TIE_BREAK_ORDER if m != self.ranking_metric
        ]

        def sort_key(result: SalaryTrainingResult) -> Tuple[Any, ...]:
            metric_components = tuple(
                self._directional_metric_value(result, m) for m in ordered_metrics
            )
            return metric_components + (result.training_seconds, result.experiment_id)

        return tuple(sorted(results, key=sort_key))

    @staticmethod
    def _raw_metric_value(
        result: SalaryTrainingResult, ranking_metric_name: str
    ) -> float:
        """
        The actual metric value in its natural units/sign — e.g. a real
        R² of 0.52 stays 0.52, never the direction-flipped -0.52 used
        internally for sorting. This is what gets shown to a human or
        handed to a downstream stage, never the sort key itself.
        """
        annual_key, _direction = _RANKING_METRICS[ranking_metric_name]
        return result.annual_metrics.get(annual_key, float("nan"))

    @staticmethod
    def _directional_metric_value(
        result: SalaryTrainingResult, ranking_metric_name: str
    ) -> float:
        annual_key, direction = _RANKING_METRICS[ranking_metric_name]
        value = result.annual_metrics.get(annual_key)
        if value is None or not math.isfinite(value):
            # Worst-case sentinel: always sorts last regardless of direction.
            return math.inf
        return value * direction

    # ------------------------------------------------------------------
    # Comparison table / logging
    # ------------------------------------------------------------------
    @staticmethod
    def _build_comparison_table(
        ranked_results: Tuple[SalaryTrainingResult, ...],
        elapsed_map: Dict[str, float],
    ) -> Tuple[Dict[str, Any], ...]:
        table = []
        for rank, result in enumerate(ranked_results, start=1):
            table.append(
                {
                    "rank": rank,
                    "experiment_id": result.experiment_id,
                    "model_name": result.model_name,
                    "annual_mae": result.annual_metrics.get("mae"),
                    "annual_rmse": result.annual_metrics.get("rmse"),
                    "annual_r2": result.annual_metrics.get("r2"),
                    "median_ape": result.annual_metrics.get("median_ape"),
                    "training_seconds": result.training_seconds,
                    "elapsed_seconds": elapsed_map.get(
                        result.experiment_id, result.training_seconds
                    ),
                    "raw_feature_count": result.raw_feature_count,
                    "transformed_feature_count": result.transformed_feature_count,
                }
            )
        return tuple(table)

    @staticmethod
    def _log_summary(
        ranked_results: Tuple[SalaryTrainingResult, ...],
        winner: SalaryTrainingResult,
        winner_config: SalaryExperimentConfig,
    ) -> None:
        logging.info("")
        logging.info("=" * 70)
        logging.info("FEATURE EXPERIMENT SUMMARY")
        logging.info("=" * 70)
        logging.info(
            "%-6s %-12s %-10s %-14s %-14s %-12s",
            "Rank",
            "Experiment",
            "R2",
            "AnnualMAE",
            "AnnualRMSE",
            "MedianAPE",
        )
        logging.info("-" * 70)
        for rank, result in enumerate(ranked_results, start=1):
            median_ape = result.annual_metrics.get("median_ape")
            median_ape_str = (
                f"{median_ape:.2f}"
                if median_ape is not None and math.isfinite(median_ape)
                else "n/a"
            )
            logging.info(
                "%-6d %-12s %-10.4f %-14.2f %-14.2f %-12s",
                rank,
                result.experiment_id,
                result.annual_metrics.get("r2", float("nan")),
                result.annual_metrics.get("mae", float("nan")),
                result.annual_metrics.get("rmse", float("nan")),
                median_ape_str,
            )
        logging.info("=" * 70)
        logging.info("WINNER")
        logging.info(f"Experiment            : {winner.experiment_id} (rank 1)")
        logging.info(f"Model                 : {winner.model_name}")
        logging.info(f"Config Signature      : {winner_config.config_signature}")
        logging.info(f"Annual R2             : {winner.annual_metrics.get('r2'):.4f}")
        logging.info(f"Annual MAE            : {winner.annual_metrics.get('mae'):.2f}")
        logging.info(f"Annual RMSE           : {winner.annual_metrics.get('rmse'):.2f}")
        median_ape = winner.annual_metrics.get("median_ape")
        if median_ape is not None and math.isfinite(median_ape):
            logging.info(f"Median APE            : {median_ape:.2f}%")
        logging.info(f"Training Time         : {winner.training_seconds:.4f} sec")
        logging.info(f"Feature Count         : {winner.raw_feature_count}")
        logging.info(f"Feature Columns       : {list(winner.raw_feature_columns)}")
        logging.info(f"Transformation Count  : {winner.transformed_feature_count}")
        logging.info("=" * 70)
