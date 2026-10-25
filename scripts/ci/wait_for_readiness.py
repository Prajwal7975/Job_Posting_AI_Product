"""
scripts/ci/wait_for_readiness.py

CI-ONLY HELPER -- polls FastAPI's GET /readiness until the response
body reports status == "ready" and model_loaded == true, or a timeout
elapses. Optionally verifies registered_model_name/model_alias match
expected values.

The response contract (status/model_loaded/registered_model_name/
model_alias/model_version keys) is read exactly as the real /readiness
endpoint defines it -- nothing here invents a new response format.

Usage:
    python scripts/ci/wait_for_readiness.py http://localhost:8000/readiness \\
        --timeout 60 --interval 2 \\
        --expected-model-name salary_prediction_model \\
        --expected-model-alias production
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--expected-model-name", default=None)
    parser.add_argument("--expected-model-alias", default=None)
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout
    last_body = None
    last_error = None
    attempts = 0

    while time.monotonic() < deadline:
        attempts += 1
        try:
            with urllib.request.urlopen(args.url, timeout=5) as response:
                last_body = json.loads(response.read())

                if (
                    response.status == 200
                    and last_body.get("status") == "ready"
                    and last_body.get("model_loaded") is True
                ):
                    print(f"Readiness OK after {attempts} attempt(s): {last_body}")

                    mismatches = []

                    if (
                        args.expected_model_name is not None
                        and last_body.get("registered_model_name")
                        != args.expected_model_name
                    ):
                        mismatches.append(
                            "registered_model_name: expected "
                            f"{args.expected_model_name!r}, got "
                            f"{last_body.get('registered_model_name')!r}"
                        )

                    if (
                        args.expected_model_alias is not None
                        and last_body.get("model_alias") != args.expected_model_alias
                    ):
                        mismatches.append(
                            "model_alias: expected "
                            f"{args.expected_model_alias!r}, got "
                            f"{last_body.get('model_alias')!r}"
                        )

                    if mismatches:
                        print(
                            "Readiness reported ready, but model identity "
                            "did not match expectations:\n  " + "\n  ".join(mismatches),
                            file=sys.stderr,
                        )
                        sys.exit(1)

                    return
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            last_error = str(exc)

        time.sleep(args.interval)

    print(
        f"Timed out after {args.timeout}s ({attempts} attempt(s)) waiting "
        f"for {args.url} to report ready.\n"
        f"Last response body: {last_body}\n"
        f"Last error: {last_error}",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
