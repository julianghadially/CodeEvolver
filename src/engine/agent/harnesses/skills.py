"""Shared CodeEvolver skill staging for provider harnesses.

Staging renders a small set of known ``{placeholder}`` tokens inside staged
``.md`` files (paths the skills reference: memory/history/traces dirs and the
workspace). Unknown braces are left untouched so JSON examples inside skills
survive verbatim.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Literal, Mapping

HarnessSkillTarget = Literal["claude", "opencode"]

CANONICAL_CODEEVOLVER_SKILLS_DIR = (
    Path(__file__).resolve().parents[2] / "optimizer" / "v3" / "skills"
)


def skill_staging_dir(workspace: str | Path, *, harness: HarnessSkillTarget) -> Path:
    """Return the provider's primary project-skill directory."""
    root = Path(workspace)
    if harness == "claude":
        return root / ".claude" / "skills"
    if harness == "opencode":
        return root / ".opencode" / "skills"
    raise ValueError(f"Unsupported skill staging harness: {harness!r}")


def _skill_template_values(workspace: str | Path | None) -> dict[str, str]:
    """Values substituted into staged skill .md files.

    Imported lazily so this module stays importable without the engine
    package side effects at import time.
    """
    from src.engine.paths import history_dir, memory_dir, traces_dir

    values = {
        "memory_dir": memory_dir(),
        "history_dir": history_dir(),
        "traces_dir": traces_dir(),
    }
    if workspace is not None:
        values["workspace_path"] = str(workspace)
    return values


def _render_skill_markdown(dest: Path, values: dict[str, str]) -> None:
    """Substitute known {placeholders} in every staged .md file, in place.

    Plain string replacement of exactly ``{key}`` for the known keys — NOT
    str.format — so JSON braces and unknown tokens in skill bodies survive.
    """
    for md in dest.rglob("*.md"):
        try:
            text = md.read_text()
        except OSError:
            continue
        rendered = text
        for key, value in values.items():
            rendered = rendered.replace("{" + key + "}", value)
        if rendered != text:
            md.write_text(rendered)


def stage_codeevolver_skills(
    workspace: str | Path,
    *,
    harness: HarnessSkillTarget,
    skills: Iterable[str] | None = None,
    overrides: Mapping[str, str | Path] | None = None,
    extra_values: Mapping[str, str] | None = None,
) -> Path:
    """Copy canonical optimizer skills into the selected harness location."""
    dest = skill_staging_dir(workspace, harness=harness)
    return stage_codeevolver_skills_to(
        dest,
        workspace=workspace,
        skills=skills,
        overrides=overrides,
        extra_values=extra_values,
    )


def stage_codeevolver_skills_to(
    dest: str | Path,
    *,
    workspace: str | Path | None = None,
    skills: Iterable[str] | None = None,
    overrides: Mapping[str, str | Path] | None = None,
    extra_values: Mapping[str, str] | None = None,
) -> Path:
    """Copy canonical optimizer skills into an explicit skill directory.

    ``skills`` optionally restricts staging to the named skill directories
    (the `OptimizerStrategy.skills_to_stage` seam — every arm passes an
    explicit tuple; `prompt_engineering` / `tracing` are never staged by an
    arm, per spec §29); None stages every canonical skill and is a
    test/dev convenience only.

    ``overrides`` maps a skill directory name to a file whose content
    REPLACES that skill's ``SKILL.md`` (the `OptimizerStrategy.skill_overrides`
    seam — pareto_evolution and freeform each swap `close_iteration`). The
    directory keeps its canonical name, so prompts and watchdogs that refer
    to a skill by name keep working; placeholders are rendered in the
    replacement exactly as in the canonical file.

    ``extra_values`` adds per-session ``{placeholder}`` values on top of the
    directory paths (the `OptimizerStrategy.skill_template_values` seam —
    freeform renders `{final_iteration_note}` into its close skill).
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    values = {**_skill_template_values(workspace), **dict(extra_values or {})}
    wanted = set(skills) if skills is not None else None
    override_map = {str(k): Path(v) for k, v in (overrides or {}).items()}
    canonical = {
        child.name for child in CANONICAL_CODEEVOLVER_SKILLS_DIR.iterdir()
        if child.is_dir()
    }
    unknown = sorted(set(override_map) - canonical)
    if unknown:
        raise ValueError(
            f"skill_overrides name unknown skill(s) {unknown}; canonical "
            f"skills are {sorted(canonical)}"
        )
    for child in CANONICAL_CODEEVOLVER_SKILLS_DIR.iterdir():
        if child.is_dir() and (wanted is None or child.name in wanted):
            target = dest / child.name
            shutil.copytree(child, target, dirs_exist_ok=True)
            override = override_map.get(child.name)
            if override is not None:
                (target / "SKILL.md").write_text(override.read_text())
            _render_skill_markdown(target, values)
    return dest


__all__ = [
    "CANONICAL_CODEEVOLVER_SKILLS_DIR",
    "HarnessSkillTarget",
    "skill_staging_dir",
    "stage_codeevolver_skills",
    "stage_codeevolver_skills_to",
]
