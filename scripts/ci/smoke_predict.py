"""
scripts/ci/smoke_predict.py

CI-ONLY HELPER -- sends one deterministic prediction request through
the running Docker stack using the existing POST /api/v1/predict
contract, and verifies the response has the expected shape and finite
numeric values.

Does NOT assert a specific salary figure: the CI smoke-test model
(see register_ci_smoke_model.py) is a tiny synthetic fixture, not the
production model, so only structural and numerical sanity are
meaningful here -- never a production-accuracy claim.

Payload contains no personal data (synthetic job attributes only).

Usage:
    python scripts/ci/smoke_predict.py http://localhost:8000/api/v1/predict
"""

from __future__ import annotations

import json
import math
import sys
import urllib.error
import urllib.request

PAYLOAD = {
    "title": "Data Scientist",
    "skill_list": "Python|SQL|Docker",
    "formatted_experience_level": "Mid-Senior level",
    "company_state": "CA",
    "company_country": "US",
    "top_industry": "Technology",
}

REQUIRED_FIELDS = {
    "predicted_annual_salary",
    "predicted_log_salary",
    "model_name",
    "model_alias",
    "registered_model_name",
}


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: smoke_predict.py <predict_url>", file=sys.stderr)
        sys.exit(2)

    url = sys.argv[1]

    request = urllib.request.Request(
        url,
        data=json.dumps(PAYLOAD).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            status = response.status
            body = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(
            f"Prediction request failed: HTTP {exc.code} -- {detail}",
            file=sys.stderr,
        )
        sys.exit(1)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        print(f"Prediction request failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if status != 200:
        print(f"Unexpected status code: {status}. Body: {body}", file=sys.stderr)
        sys.exit(1)

    missing = REQUIRED_FIELDS - set(body.keys())
    if missing:
        print(
            f"Response missing expected fields: {sorted(missing)}. " f"Body: {body}",
            file=sys.stderr,
        )
        sys.exit(1)

    for field in ("predicted_annual_salary", "predicted_log_salary"):
        value = body[field]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            print(
                f"Field {field!r} is not a finite number: {value!r}. " f"Body: {body}",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"Prediction smoke test passed: {body}")


if __name__ == "__main__":
    main()
