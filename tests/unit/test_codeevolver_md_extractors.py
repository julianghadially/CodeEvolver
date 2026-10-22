"""Unit tests for the v3 codeevolver.md field extractors.

Covers `extract_architecture_title` and `extract_architecture_summary`
in `src.engine.optimizer.v3.codeevolver_md`.

Run:  pytest tests/test_codeevolver_md_extractors.py -v
"""

from src.engine.optimizer.v3.codeevolver_md import (
    extract_architecture_summary,
    extract_architecture_title,
)


FULL_BODY = """\
PARENT_MODULE_PATH: src.pipeline.Pipeline
METRIC_MODULE_PATH: eval.metric

## ARCHITECTURE TITLE: Multi-Hop QA Pipeline
## ARCHITECTURE SUMMARY:
A two-stage retrieval pipeline that first generates
search queries, then summarizes retrieved passages.
## ARCHITECTURE DESCRIPTION:
The pipeline uses ChainOfThought for query generation
and Predict for answer synthesis.
"""

INLINE_SUMMARY = """\
## ARCHITECTURE TITLE: Inline Title
## ARCHITECTURE SUMMARY: a one-liner summary on the same line
## ARCHITECTURE DESCRIPTION:
Long description here.
"""

SUMMARY_TO_EOF = """\
PARENT_MODULE_PATH: src.pipeline.Pipeline
## ARCHITECTURE TITLE: Trailing Summary
## ARCHITECTURE SUMMARY:
This summary runs to the end of the file with no
trailing description section.
"""

NO_SUMMARY_SECTION = """\
PARENT_MODULE_PATH: src.pipeline.Pipeline
## ARCHITECTURE TITLE: Title Only
## ARCHITECTURE DESCRIPTION:
Just a description, no summary block at all.
"""

LM_SLOP = """\
## ARCHITECTURE TITLE:: Slop Title
## ARCHITECTURE SUMMARY:: # extra colons and a hash
A summary that survives the slop.
## ARCHITECTURE DESCRIPTION:
desc
"""

EMPTY_SUMMARY = """\
## ARCHITECTURE TITLE: Empty Summary
## ARCHITECTURE SUMMARY:
## ARCHITECTURE DESCRIPTION:
desc
"""


class TestExtractArchitectureTitle:
    def test_full_body(self):
        assert extract_architecture_title(FULL_BODY) == "Multi-Hop QA Pipeline"

    def test_missing(self):
        assert extract_architecture_title("no headers here") is None

    def test_lm_slop_cleaned(self):
        assert extract_architecture_title(LM_SLOP) == "Slop Title"


class TestExtractArchitectureSummary:
    def test_full_body(self):
        out = extract_architecture_summary(FULL_BODY)
        assert out is not None
        assert "two-stage retrieval pipeline" in out
        # Stops at the next ## heading
        assert "ChainOfThought" not in out

    def test_inline_summary_on_marker_line(self):
        out = extract_architecture_summary(INLINE_SUMMARY)
        assert out == "a one-liner summary on the same line"

    def test_summary_runs_to_eof(self):
        out = extract_architecture_summary(SUMMARY_TO_EOF)
        assert out is not None
        assert "runs to the end of the file" in out
        # No description after, so we capture everything until EOF
        assert "trailing description section." in out

    def test_missing_section_returns_none(self):
        assert extract_architecture_summary(NO_SUMMARY_SECTION) is None

    def test_empty_section_returns_none(self):
        assert extract_architecture_summary(EMPTY_SUMMARY) is None

    def test_lm_slop_cleaned(self):
        out = extract_architecture_summary(LM_SLOP)
        assert out is not None
        assert "extra colons and a hash" in out
        assert "A summary that survives the slop." in out

    def test_empty_body(self):
        assert extract_architecture_summary("") is None
