import argparse
import importlib
import json
import os
import asyncio
import sys
import time

import httpx
from src.shared.job_schemas import OptimizationResult

APP_MODE = os.getenv("APP_MODE", "prod")
if APP_MODE == "local":
    DEFAULT_API_URL = "http://127.0.0.1:8080"
    # Local mode binds to 127.0.0.1 and serves a single operator; the API
    # bypasses auth and ignores the Authorization header (see
    # src/api/auth/dependencies.py::get_current_user). No real key is needed,
    # so fall back to a placeholder when none is set.
    API_KEY = os.environ.get("LOCAL_API_KEY") or os.environ.get("TEST_API_KEY") or "local"
elif APP_MODE == "dev":
    DEFAULT_API_URL = "https://dev-api.codeevolver.ai"
    API_KEY = os.environ["TEST_API_KEY"]
else:
    DEFAULT_API_URL = "https://api.codeevolver.ai"
    API_KEY = os.environ["PROD_API_KEY"]

AUTH_HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Authorization-Type": "api_key",
}


async def connect_local_repo(
    client: httpx.AsyncClient, api_url: str, local_path: str
) -> str:
    """Register a local repo with the API; return the canonical repo_url."""
    response = await client.post(
        f"{api_url}/local/connect-repo",
        json={"local_path": local_path},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200, (
        f"POST /local/connect-repo failed: {response.status_code} {response.text}"
    )
    data = response.json()
    print(f"  Local repo registered: {data['repo_url']} -> {data['workspace_path']}")
    return data["repo_url"]


async def run_optimization(api_url: str, OPTIMIZE_CONFIG: dict) -> OptimizationResult:
    """Run the optimization and return results.

    This function is called once by the class-scoped fixture, and its results
    are shared across all tests in the class.
    """
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        # --- Health check ---
        try:
            health = await client.get(f"{api_url}/health", headers=AUTH_HEADERS)
            assert health.status_code == 200, (
                f"Health check failed: {health.status_code}."
            )
            print(f"Health check passed: {health.json()}")
        except httpx.ConnectError as exc:
            if APP_MODE == "local":
                error = (
                    f"Cannot connect to local API at {api_url}: {exc}. "
                    "Start the API with `SANDBOX_MODE=local APP_MODE=local "
                    "uvicorn src.api.main:app --host 127.0.0.1 --port 8080` "
                    "(see README.md)."
                )
            else:
                error = f"Cannot connect to API at {api_url}: {exc}"
            return OptimizationResult(
                final_status={"status": "failed", "error": error},
                score_history=[],
                job_id="",
                elapsed_seconds=0,
            )

        # --- Local repo registration (idempotent) ---
        if APP_MODE == "local":
            local_path = OPTIMIZE_CONFIG["repo_url"]
            canonical_url = await connect_local_repo(client, api_url, local_path)
            OPTIMIZE_CONFIG = {**OPTIMIZE_CONFIG, "repo_url": canonical_url}

        # --- Submit optimization job ---
        print(f"\nSubmitting optimization job...")
        print(f"  Repository: {OPTIMIZE_CONFIG['repo_url']}")
        print(f"  Program:    {OPTIMIZE_CONFIG['program']}")
        print(f"  Metric:     {OPTIMIZE_CONFIG['metric']}")
        print(f"  Trainset:   {OPTIMIZE_CONFIG['trainset_path']}")

        response = await client.post(
            f"{api_url}/optimize",
            json=OPTIMIZE_CONFIG,
            headers=AUTH_HEADERS,
        )
        assert response.status_code == 200, (
            f"POST /optimize failed: {response.status_code} {response.text}"
        )

        result = response.json()
        job_id = result["job_id"]
        assert result["status"] != "failed", (
            f"Job creation failed immediately: {result}"
        )
        print(f"  Job created: {job_id} (status: {result['status']})")
        return OptimizationResult(
            final_status={
                "status": result["status"],
                "job_id": job_id,
            },
            score_history=[],
            job_id=job_id,
            elapsed_seconds=0,
        )


TERMINAL_STATUSES = ("completed", "failed", "cancelled")


async def poll_until_terminal(
    api_url: str, job_id: str, interval: float = 30.0
) -> dict:
    """Block until `job_id` reaches a terminal status; return the final job doc.

    Transport errors are non-fatal — a long run must survive a brief API
    hiccup rather than reporting a false failure.
    """
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        while True:
            try:
                resp = await client.get(
                    f"{api_url}/job/{job_id}", headers=AUTH_HEADERS
                )
                if resp.status_code == 200:
                    job = resp.json()
                    if job.get("status") in TERMINAL_STATUSES:
                        return job
            except httpx.HTTPError as exc:
                print(f"  poll: transient error, retrying: {exc}", file=sys.stderr)
            await asyncio.sleep(interval)


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply `--set key=value` pairs over a config dict.

    Values are parsed as JSON when possible so `--set debug_max_iterations=2`
    yields an int and `--set repo_url=/workspace` stays a string.
    """
    out = dict(config)
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got: {item!r}")
        key, _, raw = item.partition("=")
        try:
            out[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            out[key.strip()] = raw
    return out


REQUIRED_CONFIG_KEYS = ("repo_url", "program", "metric", "trainset_path")


def resolve_config(
    module, program: str, requested: str | None,
    overrides: list[str] | None = None,
) -> tuple[str, dict]:
    """Pick which OPTIMIZE_CONFIG* dict to submit.

    `requested` (from --config) wins when supplied; otherwise default to
    OPTIMIZE_CONFIG_LOCAL in local mode and OPTIMIZE_CONFIG elsewhere.
    Raises SystemExit with the available names rather than an AttributeError /
    KeyError deep inside the request, since experiment inputs routinely stub
    out configs they don't use (e.g. `OPTIMIZE_CONFIG = {}`).
    """
    available = sorted(n for n in vars(module) if "OPTIMIZE_CONFIG" in n)

    def fail(reason: str) -> None:
        raise SystemExit(
            f"{reason}\nAvailable configs in experiments.{program}.input: "
            f"{', '.join(available) or '(none)'}"
        )

    if requested:
        config = getattr(module, requested, None)
        if config is None:
            fail(f"experiments.{program}.input has no config named '{requested}'.")
        name = requested
    elif APP_MODE == "local":
        name = "OPTIMIZE_CONFIG_LOCAL"
        config = getattr(module, name, None)
        if config is None:
            print(
                f"WARNING: experiments.{program}.input has no "
                "OPTIMIZE_CONFIG_LOCAL — falling back to OPTIMIZE_CONFIG."
            )
            name = "OPTIMIZE_CONFIG"
            config = getattr(module, name, None)
    else:
        name = "OPTIMIZE_CONFIG"
        config = getattr(module, name, None)

    if config is None:
        fail(f"experiments.{program}.input defines no '{name}'.")
    if not isinstance(config, dict):
        fail(f"experiments.{program}.input:{name} is not a dict.")
    # Overrides land BEFORE validation so `--set repo_url=/workspace` can
    # satisfy a required key the module leaves unset (the containerized
    # fleet runner relies on exactly this).
    config = apply_overrides(config, overrides or [])
    missing = [k for k in REQUIRED_CONFIG_KEYS if not config.get(k)]
    if missing:
        fail(
            f"experiments.{program}.input:{name} is missing required "
            f"key(s): {', '.join(missing)}. Pass --config to select another one."
        )
    return name, config


def main():
    parser = argparse.ArgumentParser(description="Run a CodeEvolver optimization experiment")
    parser.add_argument(
        "--program", required=True,
        help="Experiment name (e.g. hover). "
             "Loads the config dict from experiments.<program>.input",
    )
    parser.add_argument(
        "--url",
        default=os.getenv("API_URL", DEFAULT_API_URL),
        help=f"API URL (default: $API_URL or {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Name of the config dict to use from experiments.<program>.input "
             "(e.g. SMOKE_CONFIG_LOCAL). Defaults to "
             "OPTIMIZE_CONFIG_LOCAL in local mode, OPTIMIZE_CONFIG otherwise.",
    )
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="Override a config key (JSON-parsed when possible). Repeatable. "
             "E.g. --set repo_url=/workspace --set debug_max_iterations=2",
    )
    parser.add_argument(
        "--wait", action="store_true",
        help="Poll until the job reaches a terminal status; exit non-zero "
             "unless it completed.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit a single machine-readable JSON object on stdout and "
             "route human logging to stderr.",
    )
    args = parser.parse_args()

    if args.json:
        # Keep stdout a clean single JSON document for programmatic callers.
        sys.stdout = sys.stderr

    module = importlib.import_module(f"experiments.{args.program}.input")
    config_name, config = resolve_config(
        module, args.program, args.config, args.set
    )

    print(
        f"Running experiment: {args.program} "
        f"(APP_MODE={APP_MODE}, config={config_name})"
    )
    result = asyncio.run(run_optimization(args.url, config))
    print(f"\nResult: {result}")

    payload = {
        "job_id": result.job_id,
        "status": result.final_status.get("status"),
        "config": config_name,
        "program": args.program,
    }
    if not result.job_id:
        payload["error"] = result.final_status.get("error")
    elif args.wait:
        started = time.time()
        job = asyncio.run(poll_until_terminal(args.url, result.job_id))
        payload.update(
            status=job.get("status"),
            elapsed_seconds=round(time.time() - started, 1),
            final_test_score=job.get("final_test_score"),
            seed_test_score=job.get("seed_test_score"),
            final_test_scores=job.get("final_test_scores"),
            seed_test_scores=job.get("seed_test_scores"),
            error=job.get("error"),
        )

    if args.json:
        sys.stdout = sys.__stdout__
        print(json.dumps(payload))

    if payload["status"] not in ("pending", "running", "completed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()