"""
scripts/ci/wait_for_http.py

CI-ONLY HELPER -- generic HTTP readiness poller for the Phase 3 Docker
smoke test. Retries a GET request against a URL until it returns a 2xx
status or a timeout elapses, instead of relying on a single fixed
sleep. Used for plain liveness checks (MLflow /health, frontend /)
that don't need their response body inspected -- see
wait_for_readiness.py for the JSON-body-aware FastAPI /readiness check.

Usage:
    python scripts/ci/wait_for_http.py <url> [--timeout 60] [--interval 2]

Exits 0 as soon as a 2xx response is seen; exits 1 with a clear
diagnostic (last error, elapsed time) if the timeout is reached first.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout
    last_error: str | None = None
    attempts = 0

    while time.monotonic() < deadline:
        attempts += 1
        try:
            with urllib.request.urlopen(args.url, timeout=5) as response:
                if 200 <= response.status < 300:
                    print(
                        f"{args.url} is up (HTTP {response.status}) "
                        f"after {attempts} attempt(s)."
                    )
                    return
                last_error = f"HTTP {response.status}"
        except (urllib.error.URLError, OSError) as exc:
            last_error = str(exc)

        time.sleep(args.interval)

    print(
        f"Timed out after {args.timeout}s ({attempts} attempt(s)) waiting "
        f"for {args.url} to become available. Last error: {last_error}",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
