"""Copyright © 2026 440 Labs LLC

Standalone best-of-population audit (`scripts/best_of_population/README.md`).

For each benchmark x arm (`freeform`, `pareto_evolution`) it audits ONE
completed research run — the one with the highest recorded final held-out
score — re-evaluates its closed-iteration programs on a frozen difficult row
set H, freezes a top-5 portfolio, selects a per-row winner from fresh draws,
confirms the winners with independent redraws and reports the verified
best-of-population score B, the final program's score F on the same rows and
the paired exploration gap G = B - F with standard errors.

Run via `python scripts/best_of_population/run.py <stage> --out <root> ...`.
A script with module structure: it imports the engine's evaluator but is not
part of the engine and never runs on the fleet.
"""
