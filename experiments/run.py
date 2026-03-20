import argparse
import importlib
import os
import asyncio

import httpx
from src.schemas.job_schemas import OptimizationResult

DEFAULT_MODAL_URL = "https://julianghadially--codeevolver-fastapi-app.modal.run"
#DEFAULT_MODAL_URL = "https://julianghadially--codeevolver-fastapi-app-dev.modal.run"


async def run_optimization(modal_url: str, OPTIMIZE_CONFIG: dict) -> OptimizationResult:
    """Run the optimization and return results.

    This function is called once by the class-scoped fixture, and its results
    are shared across all tests in the class.
    """
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        # --- Health check ---
        try:
            health = await client.get(f"{modal_url}/health")
            assert health.status_code == 200, (
                f"Health check failed: {health.status_code}. "
                "Is 'modal serve modal_app.py' running?"
            )
            print(f"Health check passed: {health.json()}")
        except httpx.ConnectError as exc:
            return OptimizationResult(
                final_status={
                    "status": "failed",
                    "error": f"Cannot connect to Modal app at {modal_url}: {exc}\n"
                    "Make sure 'modal serve modal_app.py' is running."
                },
                score_history=[],
                job_id="",
                elapsed_seconds=0,
            )

        # --- Submit optimization job ---
        print(f"\nSubmitting optimization job...")
        print(f"  Repository: {OPTIMIZE_CONFIG['repo_url']}")
        print(f"  Program:    {OPTIMIZE_CONFIG['program']}")
        print(f"  Metric:     {OPTIMIZE_CONFIG['metric']}")
        print(f"  Trainset:   {OPTIMIZE_CONFIG['trainset_path']}")

        response = await client.post(
            f"{modal_url}/optimize",
            json=OPTIMIZE_CONFIG,
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


def main():
    parser = argparse.ArgumentParser(description="Run a CodeEvolver optimization experiment")
    parser.add_argument(
        "--program", required=True,
        help="Experiment name (e.g. FactChecker, MultihopGEPA). "
             "Loads OPTIMIZE_CONFIG from experiments.<program>.input",
    )
    parser.add_argument(
        "--url",
        default=os.getenv("MODAL_APP_URL", DEFAULT_MODAL_URL),
        help=f"Modal app URL (default: $MODAL_APP_URL or {DEFAULT_MODAL_URL})",
    )
    args = parser.parse_args()

    module = importlib.import_module(f"experiments.{args.program}.input")
    config = module.OPTIMIZE_CONFIG

    print(f"Running experiment: {args.program}")
    result = asyncio.run(run_optimization(args.url, config))
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()