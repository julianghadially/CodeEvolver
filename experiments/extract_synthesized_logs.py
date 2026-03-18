#!/usr/bin/env python3
"""
Extract GEPA run logs into a synthesized format (4-5 lines per iteration).

Per iteration we output:
- Which program was selected (and its score)
- Full proposal text(s) — no truncation
- For _code proposals: parse as JSON, omit last_change_summary, print nicely
- New subsample score
- If full eval was run: current program score on val set (not best-so-far aggregate)
"""
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

# Line prefixes that end a multi-line proposal (do not treat as continuation)
PROPOSAL_END = re.compile(r"^(Iteration \d+:|\[|Adding new )")


def parse_code_proposal(raw: str) -> Optional[Dict[str, Any]]:
    """
    Parse _code proposal text as JSON. Strip last_change_summary (and fix truncated JSON).
    Returns dict with git_branch, parent_module_path, change_request; or None if parse fails.
    """
    text = raw.strip()
    if not text or not text.startswith("{"):
        return None
    # Try parse as-is
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Often truncated at last_change_summary; strip from that key to end and close
        idx = text.find('"last_change_summary"')
        if idx > 0:
            text = text[:idx].rstrip().rstrip(",").rstrip() + "}"
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                return None
        else:
            return None
    # Omit last_change_summary
    obj.pop("last_change_summary", None)
    # Unwrap nested change_request: sometimes it's '{"change_request": "..."}'
    cr = obj.get("change_request")
    if isinstance(cr, str) and cr.strip().startswith("{"):
        try:
            inner = json.loads(cr)
            if isinstance(inner, dict) and "change_request" in inner:
                obj["change_request"] = inner["change_request"]
        except json.JSONDecodeError:
            pass
    return obj


def format_code_proposal_for_output(parsed: Dict[str, Any]) -> str:
    """Format parsed _code dict as readable output (indented JSON)."""
    return json.dumps(parsed, indent=2, ensure_ascii=False)


def extract_log(logpath: Path, program_name: str, max_iter: Optional[int] = None) -> str:
    with open(logpath, encoding="utf-8") as f:
        lines = f.readlines()

    iters: Dict[int, dict] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^Iteration (\d+): (.+)$", line.rstrip())
        if not m:
            i += 1
            continue
        num = int(m.group(1))
        if max_iter is not None and num >= max_iter:
            i += 1
            continue
        rest = m.group(2)

        if num not in iters:
            iters[num] = {}

        if "Base program full valset score" in rest:
            iters[num]["base"] = rest
            i += 1
        elif "Selected program" in rest:
            iters[num]["selected"] = rest
            i += 1
        elif "Proposed new text for _code:" in rest:
            # Full line is the proposal (may be very long); no continuation for _code
            full = rest.split("Proposed new text for _code:", 1)[-1].strip()
            iters[num].setdefault("proposals", []).append(("_code", full))
            i += 1
        elif "Proposed new text for " in rest:
            # May be single-line or multi-line
            part = rest.split("Proposed new text for ", 1)[1]
            comp_name, _, first = part.partition(":")
            comp_name = comp_name.strip()
            text_parts = [first.strip()] if first.strip() else []
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                if PROPOSAL_END.match(next_line.strip()):
                    break
                text_parts.append(next_line.rstrip())
                j += 1
            full_text = "\n".join(text_parts).strip()
            iters[num].setdefault("proposals", []).append((comp_name, full_text))
            i = j
        elif "New subsample score" in rest:
            iters[num]["subsample"] = rest
            i += 1
        elif "Found a better program" in rest:
            iters[num]["found_better"] = rest
            i += 1
        elif "Val aggregate for new program" in rest:
            # Current program's score on val set (use this, not best-so-far)
            iters[num]["val_score"] = rest
            i += 1
        elif "Valset score for new program" in rest and "Val aggregate" not in rest:
            # Alternative line with same info
            if "val_score" not in iters[num]:
                iters[num]["val_score"] = rest
            i += 1
        else:
            # Skip other iteration lines (Individual valset scores, Best valset aggregate, etc.)
            i += 1

    # Build output
    out = [
        f"(env) (base) CodeEvolver ~ % python -m experiments.run --program {program_name}",
        "Starting optimization run",
    ]
    for num in sorted(iters.keys()):
        d = iters[num]
        if "base" in d:
            out.append(f"Iteration {num}: {d['base']}")
        if "selected" in d:
            out.append(f"Iteration {num}: {d['selected']}")
        for comp_name, text in d.get("proposals", []):
            if comp_name == "_code":
                parsed = parse_code_proposal(text)
                if parsed is not None:
                    out.append(f"Iteration {num}: Proposed new text for _code:")
                    out.append(format_code_proposal_for_output(parsed))
                else:
                    out.append(f"Iteration {num}: Proposed new text for _code: {text}")
            else:
                # Preserve newlines in proposal for readability
                out.append(f"Iteration {num}: Proposed new text for {comp_name}: {text}")
        if "subsample" in d:
            out.append(f"Iteration {num}: {d['subsample']}")
        if "found_better" in d:
            out.append(f"Iteration {num}: {d['found_better']}")
        if "val_score" in d:
            out.append(f"Iteration {num}: {d['val_score']}")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Extract synthesized log from GEPA run log")
    parser.add_argument("log", type=Path, help="Input log file (e.g. hover/logsfinal_partialconstraintissue.txt)")
    parser.add_argument("-o", "--out", type=Path, default=None, help="Output file (default: same dir as log, logs_synthesized.txt)")
    parser.add_argument("--program", type=str, default="hover", help="Program name for header")
    parser.add_argument("--max-iter", type=int, default=None, help="Max iteration to include (default: all)")
    args = parser.parse_args()

    out_path = args.out
    if out_path is None:
        out_path = args.log.parent / "logs_synthesized.txt"

    text = extract_log(args.log, args.program, max_iter=args.max_iter)
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote {out_path} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
