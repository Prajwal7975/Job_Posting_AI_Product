You are a Senior MLOps / ML Platform Engineer.

You are working on my existing ML project:

Repository:
LinkedIn Job Posting Salary Prediction / Job Market Intelligence

The project is an end-to-end production-oriented salary prediction system built around an ML pipeline, MLflow Model Registry, FastAPI serving, Docker, frontend/Nginx, automated testing, and GitHub Actions.

Your job is to progressively transform the existing CI pipeline into a reliable, secure, production-ready CI/CD pipeline WITHOUT unnecessarily rewriting or breaking the existing ML pipeline, serving layer, Docker architecture, tests, or model registry workflow.

============================================================

1. # CURRENT PROJECT STATUS

The ML pipeline is already implemented.

The major ML pipeline stages are:

1. Salary Feature Engineering
2. Group-aware Dataset Splitting
3. Feature Experiments
4. Winning Feature Preparation
5. Model Family Comparison
6. Hyperparameter Tuning
7. Final Model Training
8. Validation Quality Gate
9. Final Holdout Test Evaluation
10. Final Model Verification
    10B. MLflow Model Registry Promotion

The final trainer logs the final model to MLflow with:

register_model=False

The orchestrator performs the actual registry registration only after validation, test evaluation, and artifact verification.

This avoids duplicate MLflow registry versions.

The current production model is:

Registered model:
salary_prediction_model

Production alias:
production

Current production model version:
V3

Model:
Ridge

The latest model achieved approximately:

Validation:
MAE = 0.2394
RMSE = 0.3293
R2 = 0.6598

Test:
MAE = 0.2335
RMSE = 0.3203
R2 = 0.6741

============================================================ 2. CURRENT MLflow ARCHITECTURE
============================================================

MLflow is responsible for:

- experiment tracking
- model artifacts
- model registry
- production alias
- model version metadata
- loading the production model for serving

The serving application loads:

models:/salary_prediction_model@production

The API does NOT hardcode a specific model version.

The production alias currently points to V3.

The local MLflow Docker server uses:

MLflow = 3.14.0

Backend store:

SQLite

Artifact storage:

filesystem mounted through Docker volume.

Current local MLflow architecture:

MLflow container
|
+-- /mlflow/mlflow.db
|
+-- /mlflow/mlartifacts

Docker host persistence:

./mlflow-data:/mlflow

IMPORTANT:
This local filesystem/SQLite architecture is acceptable for local development and demonstration.

It should NOT automatically be treated as a highly available production MLflow architecture.

When moving to cloud production, design proper persistent storage and backup strategy.

============================================================ 3. CURRENT SERVING ARCHITECTURE
============================================================

Current architecture:

Browser
|
v
Frontend / Nginx
|
v
FastAPI
|
v
MLflow Model Registry
|
v
salary_prediction_model@production
|
v
V3 model
|
v
Prediction

FastAPI endpoints include:

GET /
GET /health
GET /readiness
POST /api/v1/predict

Health endpoint:

/health

is a liveness check.

Readiness endpoint:

/readiness

checks whether the model is loaded and servable.

Current readiness response includes:

status
model_loaded
registered_model_name
model_alias
model_version

The API loads the model once during application startup.

It uses the MLflow production alias.

It does not continuously poll MLflow.

It does not recreate preprocessing on every request.

The serving layer has already been reviewed and should be considered frozen unless a concrete production issue requires modification.

Do NOT redesign the serving architecture unnecessarily.

============================================================ 4. CURRENT DOCKER ARCHITECTURE
============================================================

There are three Docker services:

1. MLflow
2. Salary FastAPI
3. Frontend/Nginx

Docker Compose services:

mlflow-server
salary-api
frontend

Ports:

MLflow:
5000

FastAPI:
8000

Frontend:
3000

The local Docker stack has already been successfully verified.

The containers successfully:

- build
- start
- become healthy
- load the production model
- expose readiness
- serve predictions
- serve the frontend

The actual end-to-end prediction has been tested successfully.

The frontend successfully sends a prediction request to FastAPI.

FastAPI successfully loads V3 through MLflow.

The API returns salary predictions successfully.

============================================================ 5. DEPENDENCY COMPATIBILITY ISSUE THAT WAS FIXED
============================================================

The model was trained using:

scikit-learn 1.9.0

The API container initially had:

scikit-learn 1.9.1

This caused:

InconsistentVersionWarning

The API container was corrected to:

scikit-learn==1.9.0

MLflow was also aligned.

Current API runtime:

scikit-learn==1.9.0
mlflow==3.14.0

After recreating the Docker containers:

- the warning disappeared
- model loaded successfully
- readiness returned 200
- prediction succeeded

DO NOT loosen these dependency pins again.

Production model-serving dependencies must remain reproducible.

============================================================ 6. CURRENT TESTING STATUS
============================================================

The project already contains:

tests/
conftest.py
unit/
integration/
api/
smoke/

pytest is configured with appropriate markers.

The existing test suite has passed successfully.

Latest known full suite:

259 passed

The API tests cover:

- health
- readiness
- prediction
- Pydantic validation
- domain validation
- dependency failures
- model metadata
- finite prediction output

The MLflow tracker has integration tests.

Training runner has integration tests.

Data validation and cleaning have dedicated tests.

DO NOT remove or weaken existing tests.

============================================================ 7. CURRENT GITHUB ACTIONS CI
============================================================

Current workflow is approximately:

name: CI

on:
push:
branches: - main - develop - "feature/\*\*"

pull_request:
branches: - main - develop

jobs:
test:

    name: Python Tests

    runs-on: ubuntu-latest

    timeout-minutes: 20

    steps:

      - Checkout repository

      - Setup Python 3.12

      - Install dependencies

      - Verify important project imports

      - Run pytest with coverage

      - Upload coverage reports

Current CI verifies:

- dependencies install
- project imports
- tests pass
- coverage is generated
- coverage artifacts are uploaded

The CI checks have already successfully run on GitHub.

============================================================ 8. CURRENT BRANCH / MERGE WORKFLOW
============================================================

Development is performed using feature branches.

Example:

feature/ci-pipeline

Pull requests are created toward:

main

GitHub Auto-merge has now been enabled.

The goal is:

feature branch
|
v
Pull Request
|
v
CI
|
+-- tests
+-- validation
+-- Docker checks
+-- security checks
+-- smoke checks
|
v
All required checks pass
|
v
Automatic merge
|
v
main
|
v
CD / deployment

Do not bypass required checks.

Do not use force push to main.

============================================================ 9. OBJECTIVE
============================================================

Transform the current basic CI into a complete production-quality CI/CD pipeline.

The final system should provide:

CI:

- code quality
- dependency verification
- unit tests
- integration tests
- API tests
- smoke tests
- coverage
- Docker build verification
- Docker Compose validation
- security scanning
- dependency vulnerability scanning
- artifact validation

CD:

- controlled deployment
- environment separation
- production configuration
- health checks
- readiness checks
- deployment verification
- rollback capability
- model version verification
- failure detection
- safe recovery

The pipeline must be:

- reproducible
- secure
- observable
- testable
- rollbackable
- fault tolerant
- maintainable
- cost conscious
- production oriented

============================================================ 10. IMPORTANT PRINCIPLE
============================================================

Do NOT attempt to make everything highly complicated immediately.

Build the pipeline incrementally.

Each stage must be:

1. implemented
2. tested
3. verified
4. committed
5. merged
6. only then proceed to the next stage

Do not make large uncontrolled changes across the repository.

============================================================ 11. PHASE 1 — IMPROVE CI
============================================================

Upgrade the existing GitHub Actions CI.

Current:

checkout
setup Python
install dependencies
imports
pytest
coverage

Add progressively:

A. dependency verification

Verify important runtime versions:

Python
scikit-learn
MLflow

The versions must match the production-serving requirements.

B. project structure/import verification

Keep existing import checks.

C. test matrix where useful

Consider testing supported Python versions.

Do not introduce a matrix if it causes excessive CI cost or conflicts with the project's pinned ML dependencies.

D. pytest

Run:

unit
integration
API
smoke

Use appropriate markers.

E. coverage

Keep XML and HTML coverage artifacts.

Do not make an arbitrary coverage threshold unless the existing project has a justified threshold.

If a threshold is introduced, make it explicit and maintainable.

============================================================ 12. PHASE 2 — DOCKER CI
============================================================

Add Docker validation to CI.

The CI should verify:

docker compose config

Then build:

MLflow image
FastAPI image
frontend image

Use:

docker compose build

or equivalent explicit image builds.

The purpose is to ensure that code that passes Python tests also produces valid deployment artifacts.

Do NOT deploy merely because Docker builds.

============================================================ 13. PHASE 3 — DOCKER SMOKE TEST
============================================================

After building Docker images, CI should optionally start the stack.

Expected:

MLflow:
healthy

FastAPI:
healthy

Frontend:
running

Then verify:

GET /health

GET /readiness

The readiness response must indicate:

model_loaded = true

However, be careful:

CI must NOT depend on a personal developer's local MLflow database or production model.

Create a deterministic CI-safe strategy.

Possible approach:

- use a temporary test MLflow backend
- use a small test model
- use mocked MLflow only where appropriate
- or separate deployment smoke tests from pure CI

Do NOT copy local production artifacts into GitHub Actions just to make the test pass.

The CI environment must be reproducible.

============================================================ 14. PHASE 4 — SECURITY
============================================================

Add security checks.

At minimum evaluate:

- dependency vulnerability scanning
- Python package security scanning
- Docker image scanning
- secret detection
- GitHub Actions permission hardening

Do not expose secrets in logs.

Use:

GitHub Actions Secrets
or
environment-specific secret management

Never commit:

API keys
passwords
tokens
credentials
production secrets
MLflow credentials

============================================================ 15. PHASE 5 — GITHUB ACTION SECURITY
============================================================

Harden workflow permissions.

Use least privilege.

Avoid:

permissions: write-all

Prefer explicit permissions.

Example concept:

contents: read

Only grant additional permissions when required.

Pin third-party GitHub Actions responsibly.

Do not blindly use arbitrary actions from unknown repositories.

============================================================ 16. PHASE 6 — BRANCH PROTECTION
============================================================

Production main should not accept unchecked code.

Recommended logical policy:

feature branch
|
v
PR
|
v
required CI checks
|
v
approval if configured
|
v
auto merge
|
v
main

Required CI checks should include the actual PR CI workflow.

Do not make unrelated duplicate push checks mandatory for PR merging.

Do not disable branch protection simply to make auto-merge work.

============================================================ 17. PHASE 7 — CONTINUOUS DEPLOYMENT
============================================================

After CI is stable, introduce CD.

Deployment target initially:

Render or another low-cost cloud platform.

Do NOT use AWS unless explicitly requested.

Separate environments:

development
staging
production

At minimum distinguish:

CI
staging deployment
production deployment

Do not immediately deploy every feature branch to production.

============================================================ 18. PRODUCTION ENVIRONMENT CONFIGURATION
============================================================

Environment variables must be injected by the deployment platform.

Never hardcode production values into source code.

Important configuration:

APP_ENV
MLFLOW_TRACKING_URI
SALARY_REGISTERED_MODEL_NAME
SALARY_MODEL_ALIAS
API_HOST
API_PORT
API_RELOAD
CORS_ALLOWED_ORIGINS

Production should use:

APP_ENV=production

MLflow URI should point to the deployed MLflow service.

The production model should remain:

salary_prediction_model@production

Do not hardcode:

model version = 3

============================================================ 19. MODEL DEPLOYMENT SAFETY
============================================================

Model promotion must remain separate from application deployment.

The existing workflow is:

train
↓
validate
↓
test
↓
artifact verification
↓
MLflow registration
↓
production alias

Do not automatically promote an unvalidated model merely because CI passed.

Application CI and model quality gates are different concerns.

The model quality gate must remain enforced.

============================================================ 20. PRODUCTION FAILURE HANDLING
============================================================

Design explicit failure scenarios.

At minimum consider:

A. FastAPI crashes

Expected behavior:

platform/container restarts the service.

Health/readiness checks detect failure.

B. MLflow becomes unavailable

FastAPI should fail readiness appropriately if it cannot load the model at startup.

A running API should continue serving already-loaded model predictions where the current architecture permits it.

Do not introduce continuous MLflow dependency for every prediction.

C. Model artifact download fails

Deployment should fail readiness rather than serving an unknown or partial model.

D. Invalid model version

Do not silently serve an invalid model.

E. Bad deployment

Deployment verification should fail.

The system must support rollback to the previous known-good deployment.

F. New model fails validation

Do not promote it.

Production alias remains unchanged.

G. Container crashes repeatedly

Platform-level restart policy should handle transient failures.

Persistent failures should trigger monitoring/alerting rather than endless silent restarts.

============================================================ 21. ROLLBACK STRATEGY
============================================================

Design two rollback concepts.

Application rollback:

previous known-good Docker/application version.

Model rollback:

previous known-good MLflow model version.

For model rollback:

production alias can be moved back to the previously verified model version.

Do not delete failed model versions.

Preserve lineage.

Every promoted model should retain:

run_id
model_version
validation metrics
test metrics
source commit
training metadata
feature configuration
model family

============================================================ 22. SERVER CRASH / BACKUP STRATEGY
============================================================

Do not claim that Docker alone provides disaster recovery.

For production, identify:

application recovery
model recovery
MLflow metadata recovery
artifact recovery

At minimum design:

1. persistent MLflow backend
2. persistent model artifact storage
3. backups
4. previous production model retained
5. deployment rollback
6. health checks
7. automatic service restart
8. monitoring/alerting

For local development:

Docker volumes are sufficient.

For cloud production:

use durable persistent storage appropriate to the selected platform.

Do not keep the only copy of production MLflow metadata inside an ephemeral container filesystem.

============================================================ 23. BACKUP DESIGN
============================================================

Design backup procedures for:

MLflow metadata/database
model artifacts
deployment configuration

Backups should have:

- retention
- versioning
- documented restore procedure

Do not implement fake backup logic.

If the selected cloud provider does not provide suitable persistent storage, explicitly identify the limitation and choose an appropriate external persistent storage strategy.

============================================================ 24. OBSERVABILITY
============================================================

Introduce production observability progressively.

At minimum capture:

application startup
model loading
prediction failures
validation failures
deployment failures
health status
readiness status
model version
request errors

Avoid logging sensitive user data.

Do not log full prediction payloads if they may contain sensitive information.

Later introduce:

metrics
error rates
latency
request counts
model version usage
drift metrics

============================================================ 25. MODEL DRIFT
============================================================

After CI/CD and deployment are stable, add model monitoring.

Monitor:

input feature drift
prediction drift
data quality
missing values
categorical distribution changes
salary distribution changes

Do NOT automatically retrain immediately when drift is detected.

First:

detect
↓
alert
↓
investigate
↓
trigger controlled retraining
↓
validate
↓
register
↓
promote only if quality gates pass

============================================================ 26. AUTOMATED RETRAINING
============================================================

Eventually support:

new data
↓
data validation
↓
feature engineering
↓
training
↓
evaluation
↓
quality gate
↓
MLflow registration
↓
promotion decision

A failed training run must not affect the existing production model.

Production alias must remain unchanged if the new model fails validation.

============================================================ 27. DEPLOYMENT VERIFICATION
============================================================

After deployment:

1. wait for service startup
2. check /health
3. check /readiness
4. verify expected model alias
5. verify expected model version
6. send deterministic prediction smoke test
7. verify response schema
8. verify finite prediction
9. verify HTTP status

Only consider deployment successful if all required checks pass.

============================================================ 28. ROLLBACK ON DEPLOYMENT FAILURE
============================================================

Deployment should follow:

deploy
↓
health check
↓
readiness check
↓
smoke prediction
↓
success?
|
+---- YES → deployment successful
|
+---- NO → rollback
↓
previous known-good version

Do not leave production knowingly pointing at a failed deployment.

============================================================ 29. CI/CD ARTIFACTS
============================================================

Preserve useful CI artifacts:

coverage
test reports
Docker build information where useful
security scan results
deployment logs where supported

Do not upload enormous or sensitive artifacts unnecessarily.

============================================================ 30. PIPELINE EFFICIENCY
============================================================

The project has relatively heavy ML dependencies.

Do not make every CI job unnecessarily rebuild everything.

Use:

dependency caching
pip caching
Docker layer caching where appropriate

Separate fast checks from slow checks when useful.

Example:

Fast CI:
lint/import/unit tests

Full CI:
integration/API tests

Deployment:
Docker build + deployment smoke tests

Do not sacrifice correctness for speed.

============================================================ 31. PRODUCTION QUALITY GATES
============================================================

The final pipeline should have gates at each layer.

Code gate:
tests pass

Dependency gate:
dependencies install and are compatible

Security gate:
no unacceptable vulnerabilities/secrets

Docker gate:
images build successfully

Integration gate:
services communicate successfully

Deployment gate:
health + readiness succeed

Inference gate:
prediction smoke test succeeds

ML gate:
validation quality threshold passes

Promotion gate:
only validated model can become production

============================================================ 32. FAILURE PHILOSOPHY
============================================================

The pipeline must fail safely.

Examples:

Test failure
→ do not merge

Docker build failure
→ do not deploy

Security failure
→ do not deploy

Model validation failure
→ do not promote model

Deployment health failure
→ rollback

MLflow unavailable during startup
→ fail readiness

Invalid model artifact
→ do not serve it

New model worse than quality threshold
→ retain previous production model

Never silently fall back to an unknown model.

Never silently bypass a failed quality gate.

============================================================ 33. DO NOT BREAK CURRENT ARCHITECTURE
============================================================

Before changing any existing component:

inspect it.

Do not rewrite:

ML pipeline
model registry
serving layer
API contracts
existing tests
Docker architecture

unless the change is justified by a concrete production requirement.

Preserve:

salary_prediction_model
production alias
V3 current production model
existing API contract
existing test structure

============================================================ 34. DEVELOPMENT METHOD
============================================================

Implement one phase at a time.

For every phase:

1. inspect existing files
2. identify required changes
3. explain the proposed change
4. modify only required files
5. run local tests
6. run relevant integration tests
7. verify GitHub Actions
8. verify Docker where applicable
9. document the change
10. create a focused commit

Never make a huge "productionize everything" commit.

Use focused commits such as:

feat(ci): add Docker build validation

feat(ci): add security scanning

feat(cd): add staging deployment

feat(cd): add deployment smoke tests

feat(cd): add rollback verification

feat(monitoring): add service health metrics

============================================================ 35. CURRENT IMMEDIATE TASK
============================================================

Do NOT jump directly to monitoring, drift detection, Kubernetes, or automated retraining.

First upgrade the existing GitHub Actions workflow.

Current CI:

checkout
↓
Python 3.12
↓
install requirements
↓
imports
↓
pytest + coverage
↓
coverage artifact

Next immediate target:

checkout
↓
Python setup
↓
dependency installation
↓
dependency/version verification
↓
project import verification
↓
pytest
↓
coverage
↓
Docker Compose configuration validation
↓
Docker image build verification
↓
security checks where practical
↓
artifacts
↓
PASS

After this is stable:

CI
↓
CD staging
↓
deployment verification
↓
production deployment
↓
rollback
↓
monitoring
↓
drift detection
↓
controlled retraining

============================================================ 36. EXPECTED FINAL ARCHITECTURE
============================================================

Final desired system:

Developer
|
v
Feature Branch
|
v
Pull Request
|
v
GitHub Actions
|
+--> dependency verification
|
+--> import checks
|
+--> unit tests
|
+--> integration tests
|
+--> API tests
|
+--> smoke tests
|
+--> coverage
|
+--> security scanning
|
+--> Docker build
|
+--> deployment validation
|
v
All required checks pass
|
v
GitHub Auto-Merge
|
v
main
|
v
CD
|
v
Staging
|
+--> health
+--> readiness
+--> smoke prediction
|
v
Production
|
+--> health monitoring
+--> readiness monitoring
+--> error monitoring
+--> model monitoring
|
+--> failure?
|
+--> application rollback
|
+--> model rollback
|
+--> alert

Model lifecycle:

Data
↓
Validation
↓
Training
↓
Evaluation
↓
Quality Gate
↓
MLflow
↓
Registered Model
↓
Production Alias
↓
Serving
↓
Monitoring
↓
Drift Detection
↓
Retraining
↓
Quality Gate
↓
Promotion

============================================================ 37. DEFINITION OF DONE
============================================================

Do not declare the CI/CD system production-ready merely because:

"GitHub Actions is green."

Production readiness requires evidence that:

- tests pass
- dependencies are reproducible
- Docker images build
- security checks run
- PR checks are required
- auto-merge works only after required checks
- deployment is reproducible
- health checks exist
- readiness checks exist
- model loading failures are detected
- deployment failures are detected
- rollback is possible
- model rollback is possible
- production model remains protected from failed training
- backups/persistence are addressed
- secrets are protected
- logs do not expose sensitive data
- production configuration is externalized
- monitoring exists
- failure scenarios have been tested
- recovery procedures are documented

Do not claim high availability or disaster recovery unless the infrastructure actually provides it.

============================================================ 38. MOST IMPORTANT RULE
============================================================

Do not over-engineer.

This is an MLOps portfolio project, but it should demonstrate real production engineering principles.

Prefer:

simple
reproducible
testable
observable
secure
rollbackable

over:

unnecessary Kubernetes
unnecessary microservices
unnecessary distributed infrastructure
unnecessary cloud complexity

Kubernetes is NOT required unless there is a concrete scaling requirement.

============================================================
FIRST ACTION
============================================================

Start by reviewing the existing GitHub Actions workflow and repository structure.

Do not modify anything yet.

Identify:

1. what the current CI already does
2. what is missing for production CI
3. which Docker checks can safely run in GitHub Actions
4. which checks require MLflow/model infrastructure
5. which checks belong to CI versus CD
6. what should be required for auto-merge
7. how the pipeline should handle failures

Then propose the exact Phase 1 implementation.

Only after approval should you modify the workflow.
