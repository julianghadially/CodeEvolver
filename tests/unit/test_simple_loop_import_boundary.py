"""Copyright © 2026 440 Labs LLC

The simple loop is Baseline 0 — the control the treatment (CodeEvolver) is
measured against. Its credibility rests on NOT being CodeEvolver, so what it
may import from the engine-3 optimizer is deliberately narrow and enforced
here rather than left to reviewer vigilance.

Allowed from `optimizer/v3/`: config + controls (`optimization_config`,
`constraint_gate`, `lm_config`, `coding_harness_config`), the shared eval
instrumentation (`run_eval_tool`), and the entry shim (`_entry`). Everything else — the
architect, the EE loop, closing logic, `IterationContext` /
`ArchitectConfig` and their behavior — is off-limits; the simple loop uses the
neutral `BaseIterationContext` / `BaseAgentConfig` schemas in
`src/engine/schemas/` instead.
"""

from __future__ import annotations

import ast
import pathlib

_PKG = pathlib.Path(__file__).resolve().parents[2] / "src/engine/optimizer/simple_loop"

_ALLOWED_V3 = {
    "src.engine.optimizer.v3._entry",
    "src.engine.optimizer.v3.constraint_gate",
    "src.engine.optimizer.v3.lm_config",
    "src.engine.optimizer.v3.coding_harness_config",
    "src.engine.optimizer.v3.optimization_config",
    "src.engine.optimizer.v3.run_eval_tool",
    # Trace rendering (digest.md / transcript.md), not EE logic — the capture
    # side already flows through v3.architect_trace inside the harness.
    "src.engine.optimizer.v3.architect_trace_markdown",
}
_V3_PREFIX = "src.engine.optimizer.v3"


def _imported_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module)
    return mods


def test_simple_loop_imports_no_v3_logic():
    offenders: dict[str, set[str]] = {}
    for py in sorted(_PKG.glob("*.py")):
        bad = {
            m for m in _imported_modules(py)
            if m.startswith(_V3_PREFIX) and m not in _ALLOWED_V3
        }
        if bad:
            offenders[py.name] = bad
    assert not offenders, (
        "simple_loop imported non-allowlisted v3 modules (add a control/schema, "
        f"do not import EE logic): {offenders}"
    )


def test_simple_loop_never_imports_the_logic_classes():
    """Belt-and-braces: the two behavior-bearing classes by name."""
    banned = (
        "iteration_context", "iteration_architect", "optimizer_loop", "closing",
        "close_effects", "analysis_agent", "run_reporting", "finalization",
    )
    for py in sorted(_PKG.glob("*.py")):
        for mod in _imported_modules(py):
            if mod.startswith(_V3_PREFIX):
                leaf = mod.rsplit(".", 1)[-1]
                assert leaf not in banned, f"{py.name} imports v3.{leaf}"
