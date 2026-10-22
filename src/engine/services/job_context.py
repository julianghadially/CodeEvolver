"""Copyright © 2026 440 Labs LLC

Job/auth context for the optimizer.

A single carry-bundle for the six values that travel together through the
optimizer's git-push and API-callback plumbing: the GitHub repo coordinates
(`repo_url`, `github_token`, `initial_branch`) and the API callback
credentials (`callback_url`, `jwt_token`, `job_id`).

They flow through `run_optimization` / `run_seed_step` /
`_run_optimizer_loop` down to `git_push_authenticated`; `JobContext` bundles
them so each of those signatures takes one object instead of six kwargs.

Environment-agnostic — no DB or cloud SDK imports. The generic git
helpers in `subprocess_service` stay scalar; callers feed them via
`git_creds()`.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.engine.services.callbacks import JobCallbacks


@dataclass
class JobContext:
    """GitHub + callback credentials for a single optimization run."""

    repo_url: str | None = None
    github_token: str | None = None
    callback_url: str | None = None
    jwt_token: str | None = None
    job_id: str | None = None
    initial_branch: str = "main"

    @classmethod
    def from_config(cls, config: dict) -> "JobContext":
        """Build from the optimization config dict (the JSON contract).

        Empty `callback_url` / `jwt_token` strings normalize to None so the
        callbacks layer treats them as disabled.
        """
        return cls(
            repo_url=config.get("repo_url"),
            github_token=config.get("github_token"),
            callback_url=config.get("callback_url") or None,
            jwt_token=config.get("jwt_token") or None,
            job_id=config.get("job_id"),
            initial_branch=config.get("initial_branch", "main"),
        )

    def make_callbacks(self) -> JobCallbacks:
        """Construct the API callback client from this context."""
        return JobCallbacks(
            callback_url=self.callback_url,
            jwt_token=self.jwt_token,
            job_id=self.job_id,
        )

    def git_creds(self) -> dict:
        """Credential kwargs for `subprocess_service.git_push_authenticated`.

        Splat into the call: `git_push_authenticated(workspace=..., branch=...,
        **ctx.git_creds(), set_upstream=True)`. `initial_branch` is omitted —
        it is a checkout target, not a push credential.
        """
        return {
            "repo_url": self.repo_url,
            "github_token": self.github_token,
            "callback_url": self.callback_url,
            "job_id": self.job_id,
            "jwt_token": self.jwt_token,
        }


__all__ = ["JobContext"]
