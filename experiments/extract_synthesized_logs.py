#!/usr/bin/env python3
"""Extract first 25 iterations from GEPA logs into synthesized format (4-5 lines per iteration)."""
import re
from pathlib import Path

MAX_ITER = 25
TRUNC = 200

def truncate(s, n=TRUNC):
    s = s.strip().replace("\n", " ")
    return (s[:n] + "...") if len(s) > n else s

def format_code_proposal(line):
    """Format _code proposed line as {"git_branch": "...", "change_request": "first 200 chars..."}."""
    rest = line.split("Proposed new text for _code:", 1)[-1].strip()
    branch_m = re.search(r'"git_branch"\s*:\s*"([^"]+)"', rest)
    # change_request: take content after "change_request": "
    req_m = re.search(r'"change_request"\s*:\s*"((?:[^"\\]|\\.)*)', rest)
    branch = branch_m.group(1) if branch_m else "unknown"
    req = ""
    if req_m:
        raw = req_m.group(1).replace('\\"', '"').replace("\\n", " ")
        req = (raw[:TRUNC] + "...") if len(raw) > TRUNC else raw
    else:
        req = truncate(rest, TRUNC)
    req = req.replace('"', "'")  # avoid breaking the line
    return f'{{"git_branch": "{branch}", "change_request": "{req}"}}'

def extract_log(logpath, program_name):
    with open(logpath) as f:
        lines = f.readlines()
    # Group by iteration: only keep key line types
    iters = {}
    for line in lines:
        m = re.match(r"^Iteration (\d+): (.+)$", line.rstrip())
        if not m:
            continue
        num = int(m.group(1))
        if num >= MAX_ITER:
            continue
        rest = m.group(2)
        if "Individual valset scores" in rest or "Updated valset pareto front programs" in rest or "New valset pareto front scores" in rest:
            continue
        if "Val aggregate for new program" in rest or "Valset score for new program" in rest and "coverage" in rest:
            continue
        if "Best program as per" in rest or "Best score on valset" in rest or "Linear pareto" in rest or "New program candidate index" in rest:
            continue
        if num not in iters:
            iters[num] = {}
        if "Base program full valset score" in rest:
            iters[num]["base"] = rest
        elif "Selected program" in rest:
            iters[num]["selected"] = rest
        elif "Proposed new text for _code:" in rest:
            if "proposed_code" not in iters[num]:
                iters[num]["proposed_code"] = line  # full line for format_code_proposal
        elif "Proposed new text for " in rest:
            if "proposed_text" not in iters[num]:
                # First program.X or similar proposal
                comp = rest.split("Proposed new text for ", 1)[1].split(":", 1)
                comp_name = comp[0].strip()
                text = comp[1].strip() if len(comp) > 1 else ""
                iters[num]["proposed_text"] = (comp_name, text)
        elif "New subsample score" in rest:
            iters[num]["subsample"] = rest
        elif "Found a better program" in rest:
            iters[num]["found_better"] = rest
        elif "Best valset aggregate score so far" in rest:
            iters[num]["best"] = rest
    # Build output
    out = [
        f"(env) (base) CodeEvolver ~ % python -m experiments.run --program {program_name}",
        "Starting optimization run",
    ]
    for num in range(MAX_ITER):
        if num not in iters:
            break
        d = iters[num]
        if "base" in d:
            out.append(f"Iteration {num}: {d['base']}")
        if "selected" in d:
            out.append(f"Iteration {num}: {d['selected']}")
        if "proposed_code" in d:
            out.append(f"Iteration {num}: Proposed new text for _code: {format_code_proposal(d['proposed_code'])}")
        elif "proposed_text" in d:
            comp_name, text = d["proposed_text"]
            out.append(f"Iteration {num}: Proposed new text for {comp_name}: {truncate(text)}")
        if "subsample" in d:
            out.append(f"Iteration {num}: {d['subsample']}")
        if "found_better" in d:
            out.append(f"Iteration {num}: {d['found_better']}")
        if "best" in d:
            out.append(f"Iteration {num}: {d['best']}")
    return "\n".join(out)

def main():
    base = Path(__file__).resolve().parent
    hover_log = base / "hover" / "logs_shortlivedhover.txt"
    hotpot_log = base / "hotpotGEPA" / "logs_final.txt"
    hover_out = base / "hover" / "logs_synthesized.txt"
    hotpot_out = base / "hotpotGEPA" / "logs_synthesized.txt"
    hover_out.write_text(extract_log(hover_log, "hover"), encoding="utf-8")
    hotpot_out.write_text(extract_log(hotpot_log, "hotpotGEPA"), encoding="utf-8")
    print(f"Wrote {hover_out} ({len((hover_out).read_text().splitlines())} lines)")
    print(f"Wrote {hotpot_out} ({len((hotpot_out).read_text().splitlines())} lines)")

if __name__ == "__main__":
    main()
