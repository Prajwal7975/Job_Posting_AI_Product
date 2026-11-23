# Security Exceptions Register

This document records individually reviewed security advisories that have been temporarily accepted under the current dependency constraints and deployment configuration. Each exception corresponds to exactly one `--ignore-vuln` flag in `.github/workflows/ci.yml`.

**Review Policy:**
- Exceptions are not permanent and must be re-evaluated whenever MLflow is upgraded.
- Do not broaden exceptions or add new IDs without individual review and a documented rationale.
- Any new vulnerability with a different ID will cause the CI scan to fail and block the merge.

---

## 1. PYSEC-2026-3687 / CVE-2026-64849

| Field | Value |
|-------|-------|
| **Advisory ID** | PYSEC-2026-3687 |
| **Alias** | CVE-2026-64849 |
| **Affected Package** | mlflow |
| **Affected Version** | 3.14.0 (pinned) |
| **Vulnerability** | Webhook SSRF via redirect-follow bypass |
| **Description** | The webhook API is unauthenticated in OSS MLflow. MLflow port 5000 is published on all host interfaces (0.0.0.0) via compose.yaml — other devices on the same network segment can reach it if host firewall rules permit. This is a residual network exposure. |
| **Exposure Assessment** | MLflow port 5000 is published on all host interfaces (0.0.0.0) and may be reachable by other devices depending on firewall rules. The webhook API is unauthenticated and this represents residual network exposure. |
| **Temporary Acceptance Reason** | Restricting the port binding to 127.0.0.1 is a pending security remediation tracked separately from this scan. |
| **Pending Remediation** | Fix: mlflow>=3.15.0. MLflow 3.14.0 is currently pinned in the production environment. A compatible upgrade path has not yet been validated. |
| **Conditions for Removal** | Upgrade to mlflow>=3.15.0 and restrict MLflow port binding to 127.0.0.1. |
| **Review Trigger** | MLflow version upgrade; change in network deployment topology. |

---

## 2. PYSEC-2026-3865 / CVE-2026-71211

| Field | Value |
|-------|-------|
| **Advisory ID** | PYSEC-2026-3865 |
| **Alias** | CVE-2026-71211 |
| **Affected Package** | mlflow |
| **Affected Version** | 3.14.0 (pinned) |
| **Vulnerability** | Gateway SSRF via unvalidated api_base in _create_gateway_secret |
| **Description** | Gateway SSRF via unvalidated api_base in _create_gateway_secret. |
| **Exposure Assessment** | No gateway secrets or endpoints are configured in our deployment. The vulnerable code path is not exercised. |
| **Temporary Acceptance Reason** | Gateway functionality is currently unconfigured in our deployment. No gateway secrets or endpoints are configured. The advisory remains accepted because the affected MLflow version (3.14.0) is still installed and the vulnerable code path is not exercised under current configuration. |
| **Pending Remediation** | No fix is published in any released mlflow version at the time of this writing. |
| **Conditions for Removal** | A fixed mlflow version is released and deployed; or gateway functionality is explicitly configured (requiring re-evaluation of this exception). |
| **Review Trigger** | MLflow version upgrade; introduction of gateway configuration. |

---

## 3. GHSA-gqvg-gmmx-x4hm

| Field | Value |
|-------|-------|
| **Advisory ID** | GHSA-gqvg-gmmx-x4hm |
| **Alias** | (none) |
| **Affected Package** | mlflow |
| **Affected Version** | 3.14.0 (pinned) |
| **Vulnerability** | Pickle-deserialization guard bypass in mlflow.statsmodels |
| **Description** | Pickle-deserialization guard bypass in mlflow.statsmodels. |
| **Exposure Assessment** | Our production model uses skops serialization (not pickle). statsmodels is not installed. |
| **Temporary Acceptance Reason** | statsmodels is not installed; skops serialization is used instead of pickle. |
| **Pending Remediation** | Fix: mlflow>=3.15.0 (pinned). |
| **Conditions for Removal** | Upgrade to mlflow>=3.15.0. |
| **Review Trigger** | MLflow version upgrade; addition of statsmodels dependency. |

---

## 4. CVE-2026-53995

| Field | Value |
|-------|-------|
| **Advisory ID** | CVE-2026-53995 |
| **Alias** | (none) |
| **Affected Package** | mlflow |
| **Affected Version** | 3.14.0 (pinned) |
| **Vulnerability** | FileStore path traversal via crafted trace ID |
| **Description** | FileStore path traversal via crafted trace ID. |
| **Exposure Assessment** | Our backend is sqlite:// (SqlAlchemyStore); FileStore is never instantiated. |
| **Temporary Acceptance Reason** | FileStore is not used; SqlAlchemyStore with sqlite backend is configured. |
| **Pending Remediation** | Fix: mlflow>=3.17.2 (pinned). |
| **Conditions for Removal** | Upgrade to mlflow>=3.17.2. |
| **Review Trigger** | MLflow version upgrade; switch to FileStore backend. |

---

## 5. CVE-2026-115886

| Field | Value |
|-------|-------|
| **Advisory ID** | CVE-2026-115886 |
| **Alias** | (none) |
| **Affected Package** | mlflow |
| **Affected Version** | 3.14.0 (pinned) |
| **Vulnerability** | Scorer SSRF via reflective class instantiation |
| **Description** | Scorer SSRF via reflective class instantiation. |
| **Exposure Assessment** | Requires trulens-providers-litellm, which is not installed. |
| **Temporary Acceptance Reason** | trulens-providers-litellm is not installed; the vulnerable code path is not exercised. |
| **Pending Remediation** | Fix: mlflow>=3.17.0 (pinned). |
| **Conditions for Removal** | Upgrade to mlflow>=3.17.0; or installation of trulens-providers-litellm (requiring re-evaluation). |
| **Review Trigger** | MLflow version upgrade; addition of trulens-providers-litellm dependency. |

---

## 6. CVE-2026-116691

| Field | Value |
|-------|-------|
| **Advisory ID** | CVE-2026-116691 |
| **Alias** | (none) |
| **Affected Package** | mlflow |
| **Affected Version** | 3.14.0 (pinned) |
| **Vulnerability** | CrewAI autolog leaks LLM API keys into trace telemetry |
| **Description** | CrewAI autolog leaks LLM API keys into trace telemetry. |
| **Exposure Assessment** | crewai is not installed; mlflow.crewai.autolog() is never called. |
| **Temporary Acceptance Reason** | crewai is not installed; the autolog integration is never invoked. |
| **Pending Remediation** | Fix: mlflow>=3.17.0 (pinned). |
| **Conditions for Removal** | Upgrade to mlflow>=3.17.0; or installation of crewai (requiring re-evaluation). |
| **Review Trigger** | MLflow version upgrade; addition of crewai dependency. |

---

## 7. PYSEC-2026-3554 / CVE-2026-69248

| Field | Value |
|-------|-------|
| **Advisory ID** | PYSEC-2026-3554 |
| **Alias** | CVE-2026-69248 |
| **Affected Package** | cryptography |
| **Affected Version** | <49.0.0 (mlflow==3.14.0 blocks upgrade, requires cryptography<49) |
| **Vulnerability** | cryptography: X.509 wildcard SAN escape past constrained CA |
| **Description** | X.509 wildcard SAN escape past constrained CA. |
| **Exposure Assessment** | We perform no custom X.509 chain verification. |
| **Temporary Acceptance Reason** | No custom certificate chain validation logic in our codebase. |
| **Pending Remediation** | Fix requires cryptography>=49.0.0, which mlflow==3.14.0 blocks (<49). |
| **Conditions for Removal** | Upgrade mlflow to a version that permits cryptography>=49.0.0, then upgrade cryptography. |
| **Review Trigger** | MLflow version upgrade; cryptography version upgrade. |

---

## 8. PYSEC-2026-3553 / CVE-2026-69249

| Field | Value |
|-------|-------|
| **Advisory ID** | PYSEC-2026-3553 |
| **Alias** | CVE-2026-69249 |
| **Affected Package** | cryptography |
| **Affected Version** | <49.0.0 (mlflow==3.14.0 blocks upgrade, requires cryptography<49) |
| **Vulnerability** | cryptography: cert-chain DoS via duplicate self-signed certs |
| **Description** | Certificate chain DoS via duplicate self-signed certificates. |
| **Exposure Assessment** | We parse no attacker-supplied certificate chains. |
| **Temporary Acceptance Reason** | No attacker-controlled certificate chain parsing in our codebase. |
| **Pending Remediation** | Fix requires cryptography>=49.0.0, which mlflow==3.14.0 blocks (<49). |
| **Conditions for Removal** | Upgrade mlflow to a version that permits cryptography>=49.0.0, then upgrade cryptography. |
| **Review Trigger** | MLflow version upgrade; cryptography version upgrade. |

---

## 9. PYSEC-2026-3552 / CVE-2026-69247

| Field | Value |
|-------|-------|
| **Advisory ID** | PYSEC-2026-3552 |
| **Alias** | CVE-2026-69247 |
| **Affected Package** | cryptography |
| **Affected Version** | <50.0.0 (mlflow==3.14.0 blocks upgrade, requires cryptography<49) |
| **Vulnerability** | cryptography: PKCS7 padding oracle via distinguishable errors |
| **Description** | PKCS7 padding oracle via distinguishable errors. |
| **Exposure Assessment** | We perform no PKCS7 decryption. |
| **Temporary Acceptance Reason** | No PKCS7 decryption logic in our codebase. |
| **Pending Remediation** | Fix requires cryptography>=50.0.0, which mlflow==3.14.0 blocks (<49). |
| **Conditions for Removal** | Upgrade mlflow to a version that permits cryptography>=50.0.0, then upgrade cryptography. |
| **Review Trigger** | MLflow version upgrade; cryptography version upgrade. |

---

## Summary Table

| # | Advisory ID | Alias | Package | Current Version | Blocking Upgrade |
|---|-------------|-------|---------|-----------------|------------------|
| 1 | PYSEC-2026-3687 | CVE-2026-64849 | mlflow | 3.14.0 | Upgrade compatibility not yet validated |
| 2 | PYSEC-2026-3865 | CVE-2026-71211 | mlflow | 3.14.0 | No fix released |
| 3 | GHSA-gqvg-gmmx-x4hm | — | mlflow | 3.14.0 | mlflow>=3.15.0 |
| 4 | CVE-2026-53995 | — | mlflow | 3.14.0 | mlflow>=3.17.2 |
| 5 | CVE-2026-115886 | — | mlflow | 3.14.0 | mlflow>=3.17.0 |
| 6 | CVE-2026-116691 | — | mlflow | 3.14.0 | mlflow>=3.17.0 |
| 7 | PYSEC-2026-3554 | CVE-2026-69248 | cryptography | <49.0.0 | mlflow==3.14.0 blocks |
| 8 | PYSEC-2026-3553 | CVE-2026-69249 | cryptography | <49.0.0 | mlflow==3.14.0 blocks |
| 9 | PYSEC-2026-3552 | CVE-2026-69247 | cryptography | <50.0.0 | mlflow==3.14.0 blocks |

---

*Generated as part of Phase 4 security scanning. This register must be kept in sync with `.github/workflows/ci.yml`.*