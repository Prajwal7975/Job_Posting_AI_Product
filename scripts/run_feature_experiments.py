from src.components.salary_predict.salary_allFeatures_exp_runner import (
    SalaryFeatureExperimentRunner,
)

runner = SalaryFeatureExperimentRunner(
    auto_save_report=True,
)

summary = runner.run()

print("=" * 60)
print("Winner:", summary.best_experiment_id)
print("Model :", summary.best_model)
print("R2    :", summary.best_metrics["r2"])
print("=" * 60)
