"""Copyright © 2026 440 Labs LLC

Helpers for the seed `codeevolver.md` architecture summary.

The seed step asks a reflection LM to write `codeevolver.md` — a short
architecture description the `IterationArchitect` reads at the start of
every iteration. This module owns three responsibilities:

1. `build_codeevolver_md_prompt()` — the prompt the reflection LM gets.
2. `build_codeevolver_md_fallback_body()` — a minimal stub used when the
   reflection LM call fails or the response is empty, so the loop can still
   proceed with *some* architecture file on disk.
3. `extract_architecture_title()` — parse the ARCHITECTURE TITLE line out of
   a codeevolver.md body (reflection LM output or fallback) so downstream
   consumers can use it for memory headers, UI display, etc.

The runtime template is framework-agnostic.
"""

from __future__ import annotations


def build_codeevolver_md_prompt(
    *,
    program_path: str,
    metric_path: str,
) -> str:
    """Build the reflection-LM prompt for seed `codeevolver.md` generation.

    Args:
        program_path: Dotted import path to the program entry point
            (e.g. `src.factchecker.pipeline:FactCheckerPipeline`).
        metric_path: Dotted import path to the metric function.
    """
    module_portion = program_path.split(":", 1)[0]
    file_hint = module_portion.replace(".", "/") + ".py"

    return f"""You are analyzing a codebase to generate a summary of how the program works including the architecture and key modules. Your goal is to provide any information that would be useful to an agent that needs to quickly gain context on this system before making improvements.

## Program Being Optimized
- **Entry Point**: `{program_path}`
- **Metric**: `{metric_path}`

## Your Task
1. Use the Read tool to examine the program entry point file: `{file_hint}`
2. Use Glob and Grep to find and search related source files in the same package
3. If there's a README.md at the repo root, read it for additional context

Then generate an architecture summary using this exact format:

```
PARENT_MODULE_PATH: {program_path}
METRIC_MODULE_PATH: {metric_path}

## ARCHITECTURE TITLE: <short title for the architecture, naming core modules and the approach>
## ARCHITECTURE SUMMARY:
<1-3 paragraphs naming the modules involved and describing the architecture at a high level. Succinct — save the details for the description below.>
## ARCHITECTURE DESCRIPTION:
<500-2500 characters covering what this program does, key modules and their responsibilities, data flow through the system, and the metric being optimized>
```

The PARENT_MODULE_PATH is the top-most module that the evaluator invokes per row. If the program has a pipeline wrapper, use that. Be specific about which files hold which components so a downstream architect can navigate the code quickly.

Respond with the summary only — no preamble, no trailing commentary.
"""


_FALLBACK_TITLE = "Seed architecture (reflection LM produced no summary)"


def build_codeevolver_md_fallback_body(
    *,
    program_path: str,
    metric_path: str,
    title: str | None = None,
) -> str:
    """Minimal stub body used when the reflection LM is unavailable.

    Keeps the same top-level shape as a real codeevolver.md so the
    `IterationArchitect` parser is never surprised by a missing section.
    """
    resolved_title = title or _FALLBACK_TITLE
    return (
        f"PARENT_MODULE_PATH: {program_path}\n"
        f"METRIC_MODULE_PATH: {metric_path}\n\n"
        f"## ARCHITECTURE TITLE: {resolved_title}\n"
        "## ARCHITECTURE SUMMARY:\n"
        "(seed architecture — reflection LM produced no summary)\n"
        "## ARCHITECTURE DESCRIPTION:\n"
        "(seed)\n"
    )


def extract_architecture_title(body: str) -> str | None:
    """Pull the `ARCHITECTURE TITLE:` line out of a codeevolver.md body.

    Returns the raw title text (without the header label or surrounding
    markdown punctuation) or None if the body has no title line.
    """
    for line in body.splitlines():
        if "ARCHITECTURE TITLE:" in line:
            raw = line.split("ARCHITECTURE TITLE:", 1)[1]
            # Strip leading ':'/'#'/space that the LM sometimes emits when
            # it copies the template header verbatim.
            return raw.strip().lstrip(":# ").strip() or None
    return None


def extract_architecture_summary(body: str) -> str | None:
    """Pull the `ARCHITECTURE SUMMARY:` block out of a codeevolver.md body.

    Returns the block between the `ARCHITECTURE SUMMARY:` marker and the
    next top-level `##` heading (or end of file). Strips surrounding
    whitespace. Lenient about the exact heading prefix (`##`, `###`) and
    about leading `:`/`#`/whitespace that the LM sometimes emits.
    Returns None if the section is missing or empty.
    """
    lines = body.splitlines()
    in_summary = False
    after_marker_on_first_line: str | None = None
    collected: list[str] = []

    for line in lines:
        if not in_summary:
            if "ARCHITECTURE SUMMARY:" in line:
                in_summary = True
                trailing = line.split("ARCHITECTURE SUMMARY:", 1)[1]
                trailing = trailing.strip().lstrip(":# ").strip()
                if trailing:
                    after_marker_on_first_line = trailing
            continue

        # In summary block: stop at the next ## heading (any other section).
        stripped = line.lstrip()
        if stripped.startswith("##"):
            break
        collected.append(line)

    if not in_summary:
        return None

    body_text = "\n".join(collected).strip()
    if after_marker_on_first_line and body_text:
        body_text = f"{after_marker_on_first_line}\n{body_text}"
    elif after_marker_on_first_line:
        body_text = after_marker_on_first_line

    return body_text or None


__all__ = [
    "build_codeevolver_md_prompt",
    "build_codeevolver_md_fallback_body",
    "extract_architecture_title",
    "extract_architecture_summary",
]
