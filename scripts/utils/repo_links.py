"""Convert local repo paths embedded in job records into GitHub repo links.

Local research runs (`APP_MODE=local`) reference the repo by its on-disk path
(e.g. `/Users/me/Documents/code/phantom-wiki`). When such a job is uploaded to
the cloud or transferred between accounts, those local paths are meaningless to
the developer portal — and the portal builds a program's GitHub branch link as
`{job.repo_url}/tree/{git_branch}` (see codeevolver-frontend
`src/app/portal/results/programs/page.tsx`). It reads the *job-level* `repo_url`,
not anything inside the iteration history.

So this module does two related things, both driven by one lookup table:

1. `find_repo_url_in_iteration_history` — discover which GitHub repo a job's
   iteration history belongs to, so a missing job-level `repo_url` can be
   backfilled (the field the portal actually needs for branch links).
2. `normalize_iteration_history` — rewrite any local absolute paths that appear
   *inside* the iteration-history text fields (change_request, analyses, error
   strings, …) into their GitHub equivalents, so the portal never displays a
   developer's local filesystem path.

The lookup table maps a substring that uniquely identifies the repo (normally
its directory / repo name) to the canonical GitHub URL. Add a row per repo.
"""

from __future__ import annotations

import re
from copy import deepcopy

# substring contained in a local repo path  ->  canonical GitHub repo URL.
# The key just has to be specific enough to identify the repo (its dir name).
REPO_PATH_TO_LINK: dict[str, str] = {
    "/LangProBe-CodeEvolver": "https://github.com/julianghadially/LangProBe-CodeEvolver",
    "/phantom-wiki": "https://github.com/julianghadially/phantom-wiki",
}


def convert_repo_path_to_link(repo_path: str | None) -> str | None:
    """Return the GitHub link if any lookup key is contained in `repo_path`.

    The whole `repo_path` string maps to the repo link — this is the identity
    check used to decide *which* repo a string belongs to. Returns None when no
    key matches (or the input is not a non-empty string).
    """
    if not isinstance(repo_path, str) or not repo_path:
        return None
    for key, link in REPO_PATH_TO_LINK.items():
        if key in repo_path:
            return link
    return None


def find_repo_url_in_iteration_history(iteration_history: dict | None) -> str | None:
    """Scan every string value in the iteration history for a known repo.

    Returns the first matching GitHub link, or None if no record string
    contains a known repo path. Used to backfill a job's missing `repo_url`.
    """
    for record in (iteration_history or {}).values():
        if not isinstance(record, dict):
            continue
        for value in record.values():
            link = convert_repo_path_to_link(value) if isinstance(value, str) else None
            if link:
                return link
    return None


def _normalize_string(text: str) -> str:
    """Replace local absolute repo paths in `text` with their GitHub links.

    Matches an absolute path whose final segment is a lookup key
    (`/Users/me/code/phantom-wiki`) and rewrites the *repo-root prefix* to the
    GitHub URL, preserving any trailing in-repo subpath:
        /Users/me/code/phantom-wiki/src/x.py
        -> https://github.com/julianghadially/phantom-wiki/src/x.py
    A bare repo name with no leading path (mentioned in prose) is left alone.
    """
    for key, link in REPO_PATH_TO_LINK.items():
        # Leading `/segment` parts, then the repo segment == key, with a
        # boundary lookahead so we don't match `phantom-wiki-2`.
        pattern = re.compile(r"(?:/[^\s'\"/]+)*/" + re.escape(key) + r"(?![^\s'\"/])")
        text = pattern.sub(link, text)
    return text


def normalize_iteration_history(
    iteration_history: dict | None,
) -> tuple[dict, str | None]:
    """Return a copy of `iteration_history` with local paths rewritten to links.

    Also returns the repo URL detected while scanning (None if none found), so
    callers can backfill a missing job-level `repo_url` in the same pass.
    """
    if not iteration_history:
        return {}, None

    normalized = deepcopy(iteration_history)
    detected = find_repo_url_in_iteration_history(iteration_history)

    for record in normalized.values():
        if not isinstance(record, dict):
            continue
        for field, value in record.items():
            if isinstance(value, str):
                record[field] = _normalize_string(value)

    return normalized, detected
