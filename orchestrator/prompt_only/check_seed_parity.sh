#!/usr/bin/env bash
# Copyright © 2026 440 Labs LLC
# Prompt-only ablation: prove the `prompt-only` seed renders byte-identical LM
# prompts to `main` (every predictor formatted with dspy.ChatAdapter on
# placeholder inputs, incl. phantomwiki's ReAct tool listing). Uses temporary
# worktrees of both branches; never touches the client checkouts.
#
#   ./orchestrator/prompt_only/check_seed_parity.sh     # exit 0 = identical
set -euo pipefail
ROOT=${CE_REPO_ROOT:-~/Documents/code}
HERE=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d); trap 'git -C "$ROOT/LangProBe-CodeEvolver" worktree prune; git -C "$ROOT/phantom-wiki" worktree prune; rm -rf "$TMP"' EXIT
rc=0
check() {  # repo module class
  local repo=$ROOT/$1
  for br in main prompt-only; do
    git -C "$repo" worktree add -q --detach "$TMP/$1-$br" "$br"
    "$repo/env/bin/python" "$HERE/render_prompts.py" "$TMP/$1-$br" "$2" "$3" > "$TMP/$1-$br.json"
  done
  if cmp -s "$TMP/$1-main.json" "$TMP/$1-prompt-only.json"; then echo "IDENTICAL  $1 ($3)"
  else echo "DIFFERENT  $1 ($3)"; diff "$TMP/$1-main.json" "$TMP/$1-prompt-only.json" | head -20; rc=1; fi
}
check LangProBe-CodeEvolver langProBe.hover.hover_program HoverMultiHop
check phantom-wiki src.program.phantomwiki_module PhantomWikiReAct
exit $rc
