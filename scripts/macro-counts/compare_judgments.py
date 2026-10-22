#!/usr/bin/env python3
"""Compare two sets of judge/<pair>__<order>.json (e.g. a re-judge against the cached results).

    python compare_judgments.py ~/ce-analysis/macro-counts/judge path/to/other/judge [--top-k 7]

Prints per judgment the top-k count per arm on both sides, how many listed ids each
side left out of the combined ranking as shared, and whether the evidence hash
matched (same input), then the combined rankings side
by side wherever they differ. Counts are keyed by arm (from each wrapper's
``arm_of``), not by program letter.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import LIST_KEY, shared_ids, top_k_counts  # noqa: E402


def load(d: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))}


def by_arm(jd: dict, k: int) -> dict[str, int]:
    top = top_k_counts(jd["judgment"], k)
    return {jd.get("arm_of", {}).get(p, p): n for p, n in top.items()}


def ranking(jd: dict) -> list[str]:
    j = jd["judgment"]
    names = {a["id"]: a["name"] for key in LIST_KEY.values() for a in j[key]}
    arm_of = jd.get("arm_of", {})
    return [f"{rank}. {arm_of.get(c['program'], c['program'])}: {names.get(c['id'], '?')}"
            for rank, c in enumerate(j["combined_ranking"], 1)]


def n_shared(jd: dict) -> int:
    return sum(len(v) for v in shared_ids(jd["judgment"]).values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--top-k", type=int, default=7)
    args = ap.parse_args()
    if not (args.a.is_dir() and args.b.is_dir()):
        print(__doc__, file=sys.stderr)
        return 2
    A, B = load(args.a), load(args.b)
    common = sorted(set(A) & set(B))
    if not common:
        print("no common judgments")
        return 1
    k = args.top_k
    print(f"| judgment | A top{k} by arm | B top{k} by arm | left out as shared A/B | same evidence |")
    print("|---|---|---|---|---|")
    exact = within1 = 0
    for r in common:
        ca, cb = by_arm(A[r], k), by_arm(B[r], k)
        fmt = lambda c: ", ".join(f"{arm} {n}" for arm, n in sorted(c.items()))
        same_ev = (A[r].get("workspace_sha256") or A[r].get("evidence_sha256")) == \
                  (B[r].get("workspace_sha256") or B[r].get("evidence_sha256"))
        diffs = [abs(ca.get(arm, 0) - cb.get(arm, 0)) for arm in set(ca) | set(cb)]
        exact += max(diffs) == 0
        within1 += max(diffs) <= 1
        print(f"| {r} | {fmt(ca)} | {fmt(cb)} | {n_shared(A[r])}/{n_shared(B[r])} | "
              f"{'yes' if same_ev else 'NO'} |")
    print()
    print(f"{len(common)} judgments: top-{k} counts identical on {exact}, within +-1 per arm on {within1}")
    print()
    for r in common:
        ra, rb = ranking(A[r]), ranking(B[r])
        if ra == rb:
            continue
        print(f"## {r}")
        for i in range(max(len(ra), len(rb))):
            print(f"  A: {ra[i] if i < len(ra) else '-'}")
            print(f"  B: {rb[i] if i < len(rb) else '-'}")
            if i + 1 == k:
                print(f"  --- top {k} ---")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
