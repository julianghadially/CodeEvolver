"""Copyright © 2026 440 Labs LLC

Pure arithmetic of the audit (`scripts/best_of_population/README.md`).
No engine imports, no I/O — every rule that decides a number lives here so the
synthetic-fixture tests can pin it without an evaluator.

Conventions: candidates are keyed by git tree (str); rows are int testset
indices; `scores[tree][row]` is one observation, `matrix[tree][row]` a list
of observations in draw order.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Iterable, Mapping, Sequence

# Two-sided 95 % t critical values by degrees of freedom (reps - 1).
T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
             8: 2.306, 9: 2.262, 10: 2.228, 14: 2.145, 19: 2.093, 29: 2.045}


def t_crit(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in T_CRIT_95:
        return T_CRIT_95[df]
    keys = sorted(T_CRIT_95)
    return T_CRIT_95[max(k for k in keys if k <= df)] if df > keys[0] else T_CRIT_95[keys[0]]


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def sample_sd(xs: Sequence[float]) -> float:
    """Sample standard deviation, denominator n-1 (NaN when n < 2)."""
    n = len(xs)
    if n < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def standard_error(xs: Sequence[float]) -> float:
    return sample_sd(xs) / math.sqrt(len(xs)) if len(xs) >= 2 else float("nan")


# ---------------------------------------------------------------------------
# §1 difficult set
# ---------------------------------------------------------------------------

def classify_score_type(values: Iterable[float]) -> tuple[str, float]:
    """`("binary", m)` when every observed value is 0 or one positive value m
    (hover scores 0 / 0.998), else `("fractional", max)`."""
    vals = {round(float(v), 6) for v in values}
    nonzero = sorted(v for v in vals if v != 0.0)
    if not nonzero:
        return "binary", 1.0
    if len(nonzero) == 1:
        return "binary", nonzero[0]
    return "fractional", max(nonzero)


def normalize_observation(score: float, score_type: str, max_score: float) -> float:
    """Binary: success iff score >= max/2 (hover's 0.998 -> 1). Fractional:
    the score as recorded (already on [0, 1])."""
    if score_type == "binary":
        return 1.0 if float(score) >= max_score / 2.0 else 0.0
    return float(score)


def difficult_set(obs_by_row: Mapping[int, Sequence[float]], threshold: float = 0.5) -> list[int]:
    """Rows whose mean observation is STRICTLY below `threshold`. Rows with no
    observations cannot be classified and are not returned."""
    return sorted(r for r, obs in obs_by_row.items() if obs and mean(obs) < threshold)


# ---------------------------------------------------------------------------
# §2 portfolio
# ---------------------------------------------------------------------------

def aggregate_means(scores: Mapping[str, Mapping[int, float]], rows: Sequence[int]) -> dict[str, float]:
    return {t: mean(s[r] for r in rows) for t, s in scores.items()}


def greedy_top_k(
    scores: Mapping[str, Mapping[int, float]], pool: Iterable[str], rows: Sequence[int], k: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Start with the top mean scorer; then repeatedly add the candidate with
    the largest additional row-wise performance beyond the current portfolio,
    sum_r max(0, s_c[r] - max_{p in portfolio} s_p[r]). Ties -> higher overall
    mean, then ascending tree (deterministic)."""
    pool = list(pool)
    if not pool:
        return [], []
    means = {t: mean(scores[t][r] for r in rows) for t in pool}
    order = lambda t: (-means[t], t)  # noqa: E731
    first = sorted(pool, key=order)[0]
    chosen = [first]
    steps = [{"tree": first, "gain": None, "mean": means[first], "rule": "top_mean"}]
    best_so_far = {r: scores[first][r] for r in rows}
    while len(chosen) < k and len(chosen) < len(pool):
        remaining = [t for t in pool if t not in chosen]
        gains = {t: sum(max(0.0, scores[t][r] - best_so_far[r]) for r in rows) for t in remaining}
        nxt = sorted(remaining, key=lambda t: (-gains[t], -means[t], t))[0]
        chosen.append(nxt)
        steps.append({"tree": nxt, "gain": gains[nxt], "mean": means[nxt], "rule": "max_additional_gain"})
        for r in rows:
            best_so_far[r] = max(best_so_far[r], scores[nxt][r])
    return chosen, steps


def force_final(portfolio: list[str], final_tree: str) -> tuple[list[str], str | None]:
    """The final program must be in the portfolio; replace the LAST added
    candidate when it is not. Returns (portfolio, replaced_tree)."""
    if final_tree in portfolio:
        return list(portfolio), None
    out = list(portfolio)
    replaced = out.pop() if out else None
    out.append(final_tree)
    return out, replaced


def backfill(portfolio: list[str], scores: Mapping[str, Mapping[int, float]],
             extra_pool: Iterable[str], rows: Sequence[int], k: int) -> tuple[list[str], list[str]]:
    """Fewer than k survivors: add candidates from `extra_pool` by aggregate
    score on the rows (ties ascending tree). Returns (portfolio, added)."""
    out = list(portfolio)
    means = {t: mean(scores[t][r] for r in rows) for t in extra_pool if t not in out}
    added = []
    for t in sorted(means, key=lambda t: (-means[t], t)):
        if len(out) >= k:
            break
        out.append(t)
        added.append(t)
    return out, added


# ---------------------------------------------------------------------------
# §3 selection
# ---------------------------------------------------------------------------

def row_winners(
    select: Mapping[str, Mapping[int, Sequence[float]]], rows: Sequence[int],
) -> tuple[dict[int, dict[str, Any]], dict[str, dict[int, float]], dict[str, float]]:
    """Per row, the candidate with the highest selection mean; ties -> the
    higher overall (all-rows) selection mean, then ascending tree.

    Returns (winners{row: {tree, mean, tied}}, select_means{tree: {row: mean}},
    overall_means{tree: mean})."""
    select_means = {t: {r: mean(select[t][r]) for r in rows} for t in select}
    overall = {t: mean(select_means[t][r] for r in rows) for t in select}
    winners: dict[int, dict[str, Any]] = {}
    for r in rows:
        ranked = sorted(select, key=lambda t: (-select_means[t][r], -overall[t], t))
        top = ranked[0]
        tied = [t for t in ranked if select_means[t][r] == select_means[top][r]]
        winners[r] = {"tree": top, "mean": select_means[top][r], "tied": tied if len(tied) > 1 else []}
    return winners, select_means, overall


# ---------------------------------------------------------------------------
# §4-§5 confirmation metrics
# ---------------------------------------------------------------------------

def confirmation_metrics(
    rows: Sequence[int],
    winners: Mapping[int, str],
    confirm: Mapping[int, Sequence[float]],
    final: Mapping[int, Sequence[float]],
    final_tree: str,
    select_means: Mapping[str, Mapping[int, float]] | None = None,
    reps: int | None = None,
) -> dict[str, Any]:
    """`confirm[row]` = the frozen winner's confirmation observations (one per
    round); `final[row]` = the final program's confirmation observations on
    rows where it is NOT the winner (rows where it is reuse `confirm[row]`,
    a paired contribution of exactly zero). Raises on any missing observation
    so a partial round can never be reported."""
    rows = list(rows)
    if not rows:
        raise ValueError("empty difficult set")
    R = reps or len(confirm[rows[0]])
    for r in rows:
        if len(confirm[r]) < R:
            raise ValueError(f"row {r}: {len(confirm[r])} confirmation observations, need {R}")
        if winners[r] != final_tree and len(final.get(r, ())) < R:
            raise ValueError(f"row {r}: {len(final.get(r, ()))} final-program observations, need {R}")
    B_rounds, F_rounds = [], []
    for k in range(R):
        b = [confirm[r][k] for r in rows]
        f = [confirm[r][k] if winners[r] == final_tree else final[r][k] for r in rows]
        B_rounds.append(mean(b))
        F_rounds.append(mean(f))
    D_rounds = [b - f for b, f in zip(B_rounds, F_rounds)]
    B, F, G = mean(B_rounds), mean(F_rounds), mean(D_rounds)
    se_B, se_F, se_G = standard_error(B_rounds), standard_error(F_rounds), standard_error(D_rounds)
    crit = t_crit(R - 1)
    n = len(rows)
    out: dict[str, Any] = {
        "n_H": n, "reps": R,
        "B_rounds": B_rounds, "F_rounds": F_rounds, "D_rounds": D_rounds,
        "B": B, "F": F, "G": G,
        "SE_B": se_B, "SE_F": se_F, "SE_G": se_G,
        "t_crit": crit,
        "CI95_B": [B - crit * se_B, B + crit * se_B],
        "CI95_G": [G - crit * se_G, G + crit * se_G],
        "sum_B": n * B, "sum_F": n * F, "sum_G": n * G,
        "degenerate_zero_variance_B": bool(se_B == 0.0),
        "degenerate_zero_variance_G": bool(se_G == 0.0),
        "n_rows_final_is_winner": sum(1 for r in rows if winners[r] == final_tree),
    }
    if select_means is not None:
        sel = mean(select_means[winners[r]][r] for r in rows)
        out["mean_selected_selection_score"] = sel
        out["selection_to_confirmation_drop"] = sel - B
    return out


def cross_arm(evo: Mapping[str, Any], freeform: Mapping[str, Any]) -> dict[str, Any]:
    """Evolution minus freeform in B; the arms' confirmation draws are
    independent, so SE = sqrt(SE_evo^2 + SE_ff^2). No round pairing."""
    diff = evo["B"] - freeform["B"]
    se = math.sqrt(evo["SE_B"] ** 2 + freeform["SE_B"] ** 2)
    crit = t_crit(min(evo["reps"], freeform["reps"]) - 1)
    return {"evo_minus_freeform_B": diff, "SE": se, "CI95": [diff - crit * se, diff + crit * se]}


# ---------------------------------------------------------------------------
# Synthetic audit (tests: no systematic selection optimism)
# ---------------------------------------------------------------------------

def simulate_audit(
    rng: random.Random, p_by_candidate: Sequence[float], n_rows: int, *,
    reps_select: int = 5, reps_confirm: int = 5, final_index: int = 0,
) -> dict[str, Any]:
    """One synthetic audit over Bernoulli candidates with row-independent
    success probabilities. Returns the confirmation metrics plus the true
    expected best-of-population score (mean over rows of the winner's p)."""
    trees = [f"c{i}" for i in range(len(p_by_candidate))]
    rows = list(range(n_rows))
    draw: Callable[[str], float] = lambda t: 1.0 if rng.random() < p_by_candidate[trees.index(t)] else 0.0  # noqa: E731
    select = {t: {r: [draw(t) for _ in range(reps_select)] for r in rows} for t in trees}
    winners, select_means, _ = row_winners(select, rows)
    win = {r: winners[r]["tree"] for r in rows}
    final_tree = trees[final_index]
    confirm = {r: [draw(win[r]) for _ in range(reps_confirm)] for r in rows}
    final = {r: [draw(final_tree) for _ in range(reps_confirm)] for r in rows if win[r] != final_tree}
    m = confirmation_metrics(rows, win, confirm, final, final_tree, select_means, reps_confirm)
    m["true_B"] = mean(p_by_candidate[trees.index(win[r])] for r in rows)
    m["true_F"] = p_by_candidate[final_index]
    return m


__all__ = [
    "T_CRIT_95", "aggregate_means", "backfill", "classify_score_type", "confirmation_metrics",
    "cross_arm", "difficult_set", "force_final", "greedy_top_k", "mean", "normalize_observation",
    "row_winners", "sample_sd", "simulate_audit", "standard_error", "t_crit",
]
