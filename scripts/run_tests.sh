#!/usr/bin/env bash
# Copyright © 2026 440 Labs LLC
#
# Unified CodeEvolver test runner. One command runs every test except the
# long-running end-to-end arms, which are opt-in per arm.
#
# Sections:
#   Phase 1 — deterministic suites (unit / controls / coding_harness / api).
#             Fast, offline, no real SDK.
#   Phase 2 — functional agent tests, swept over the coding harness
#             (claude, then opencode), via local_test_app.py. Needs API keys.
#   Phase 3 — end-to-end optimizer runs, one per arm. Long-running and
#             billable; each arm opts in on its own.
#
# Usage:
#   scripts/run_tests.sh [--harness claude|opencode|both]
#                        [--phase all|deterministic|functional]
#                        [--include-e2e | --include-ee-e2e | --include-pareto-e2e
#                         | --include-freeform-e2e | --include-simple-e2e
#                         | --include-asa-e2e]
#
#   --phase all           (default) Phase 1 + Phase 2 (+ Phase 3 per e2e flags)
#   --phase deterministic only Phase 1 (offline pytest suites)
#   --phase functional    only Phase 2 (functional agent sweep); skips Phase 1
#
#   --include-ee-e2e      run the EE arm only
#   --include-pareto-e2e  run the fixed-evolution arm only
#   --include-freeform-e2e run the freeform (Baseline 1) arm only
#   --include-simple-e2e  run the simple-loop arm only
#   --include-asa-e2e     run the ASA arm only
#   --include-e2e         run all five; equivalent to passing all of the above
#
#   The per-arm flags combine, so e.g.
#     scripts/run_tests.sh --phase functional --include-simple-e2e
#   runs the functional sweep plus the simple-loop arm and nothing else. Each
#   arm is a real optimizer run against a real agent — pick deliberately.
#
# Env:
#   ANTHROPIC_API_KEY            required for the claude functional pass
#                                (+ the Sonnet 5 constraint-checker eval)
#   OPENAI_API_KEY               required for the opencode pass (analysis-agent
#                                change-summary judge + any openai/* model)
#   DEEPINFRA_API_KEY            required when CODING_HARNESS_MODEL is a
#                                deepinfra/* model (the default GLM 5.2)
#   CODING_HARNESS_MODEL         opencode provider/model
#                                (default: deepinfra/zai-org/GLM-5.2)
set -uo pipefail

# --- args -------------------------------------------------------------------
HARNESS="both"
PHASE="all"
# Phase 3 arms, each independently selectable. --include-e2e turns on all five.
E2E_EE=0
E2E_PARETO=0
E2E_FREEFORM=0
E2E_SIMPLE=0
E2E_ASA=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --harness)     HARNESS="$2"; shift 2 ;;
    --phase)       PHASE="$2"; shift 2 ;;
    --include-e2e) E2E_EE=1; E2E_PARETO=1; E2E_FREEFORM=1; E2E_SIMPLE=1; E2E_ASA=1; shift ;;
    --include-ee-e2e)     E2E_EE=1; shift ;;
    --include-pareto-e2e) E2E_PARETO=1; shift ;;
    --include-freeform-e2e) E2E_FREEFORM=1; shift ;;
    --include-simple-e2e) E2E_SIMPLE=1; shift ;;
    --include-asa-e2e)    E2E_ASA=1; shift ;;
    -h|--help)
      # Print the leading comment block, so the usage text cannot drift out of
      # sync with a hard-coded line range as this file grows.
      awk 'NR>1 { if (/^#/) print; else exit }' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

case "$HARNESS" in claude|opencode|both) ;; *) echo "bad --harness: $HARNESS" >&2; exit 2 ;; esac
case "$PHASE" in all|deterministic|functional) ;; *) echo "bad --phase: $PHASE" >&2; exit 2 ;; esac

# Which phases run for the selected --phase.
RUN_DETERMINISTIC=0; RUN_FUNCTIONAL=0
case "$PHASE" in
  all)           RUN_DETERMINISTIC=1; RUN_FUNCTIONAL=1 ;;
  deterministic) RUN_DETERMINISTIC=1 ;;
  functional)    RUN_FUNCTIONAL=1 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# Export so every subprocess (pytest, local_test_app.py) inherits it.
# Without the export, the deterministic pytest phase runs with APP_MODE unset,
# which src/config.py defaults to `prod` 
export APP_MODE="${APP_MODE:-local}"
# GLM 5.2 (via DeepInfra) is the vetted OpenCode harness model; override for
# one-off cross-model sweeps (e.g. CODING_HARNESS_MODEL=openai/gpt-5.4-mini).
CODING_HARNESS_MODEL="${CODING_HARNESS_MODEL:-deepinfra/zai-org/GLM-5.2}"

# Harnesses to sweep for the functional section.
if [[ "$HARNESS" == "both" ]]; then HARNESSES=(claude opencode); else HARNESSES=("$HARNESS"); fi

# --- result tracking --------------------------------------------------------
declare -a RESULTS
FAILED=0

run_step() {  # run_step "<label>" <cmd...>
  local label="$1"; shift
  echo ""
  echo "────────────────────────────────────────────────────────────"
  echo "▶ $label"
  echo "  \$ $*"
  echo "────────────────────────────────────────────────────────────"
  if "$@"; then
    RESULTS+=("PASS  $label")
  else
    RESULTS+=("FAIL  $label")
    FAILED=1
  fi
}

skip_step() {  # skip_step "<label>" "<reason>"
  echo ""
  echo "⏭  SKIP  $1  ($2)"
  RESULTS+=("SKIP  $1  ($2)")
}

# Why a harness can't run here (empty => good to go).
harness_skip_reason() {
  case "$1" in
    claude)
      [[ -z "${ANTHROPIC_API_KEY:-}" ]] && { echo "missing ANTHROPIC_API_KEY"; return; }
      ;;
    opencode)
      # OPENAI_API_KEY is needed even for non-openai harness models: the
      # analysis_agent change-summary judge is a litellm openai/gpt-5.4-mini call.
      [[ -z "${OPENAI_API_KEY:-}" ]] && { echo "missing OPENAI_API_KEY"; return; }
      if [[ "$CODING_HARNESS_MODEL" == deepinfra/* && -z "${DEEPINFRA_API_KEY:-}" ]]; then
        echo "missing DEEPINFRA_API_KEY for ${CODING_HARNESS_MODEL}"; return
      fi
      if ! command -v opencode >/dev/null 2>&1; then
        echo "opencode binary not on PATH"; return
      fi
      # OpenCode's catalog (`opencode models openai`) lists models the OpenAI
      # account may not be entitled to call — gated previews resolve in the
      # catalog but 404 on the live API (e.g. gpt-5.3-codex-spark). Pre-flight
      # the configured model so an unavailable one SKIPs with a clear reason
      # instead of failing every functional test with a cryptic provider error.
      if [[ "$CODING_HARNESS_MODEL" == openai/* ]]; then
        local code
        code="$(curl -s -o /dev/null -w '%{http_code}' \
          "https://api.openai.com/v1/models/${CODING_HARNESS_MODEL#openai/}" \
          -H "Authorization: Bearer ${OPENAI_API_KEY}" 2>/dev/null || echo 000)"
        if [[ "$code" == "404" || "$code" == "403" ]]; then
          echo "model ${CODING_HARNESS_MODEL} not available on this OpenAI account (HTTP ${code})"
          return
        fi
      fi
      ;;
  esac
}

# Run one functional command for a given harness.
functional_cmd() {  # functional_cmd <harness> <command> [extra args...]
  local h="$1" cmd="$2"; shift 2
  local model_args=()
  [[ "$h" == "opencode" ]] && model_args=(--coding-harness-model "$CODING_HARNESS_MODEL")
  python local_test_app.py "$cmd" --coding-harness "$h" ${model_args[@]+"${model_args[@]}"} "$@"
}

# ===========================================================================
# Phase 1 — deterministic suites (offline)
# ===========================================================================
if [[ "$RUN_DETERMINISTIC" == "1" ]]; then
  # tests/api exercises the hosted service and is left out of the public release.
  deterministic_suites=(tests/unit tests/controls tests/coding_harness)
  [[ -d tests/api ]] && deterministic_suites+=(tests/api)
  echo "=========================================================="
  echo " Phase 1 — deterministic suites (${deterministic_suites[*]#tests/})"
  echo "=========================================================="
  run_step "pytest: ${deterministic_suites[*]#tests/}" \
    python -m pytest "${deterministic_suites[@]}" \
    -m "not live and not integration" -q
fi

# ===========================================================================
# Phase 2 — functional agent tests (claude → opencode)
# ===========================================================================
if [[ "$RUN_FUNCTIONAL" == "1" ]]; then
  echo ""
  echo "=========================================================="
  echo " Phase 2 — functional agent tests (harness=$HARNESS)"
  echo "=========================================================="

  # Constraint checker is harness-agnostic. Unit cases always run offline; the
  # real-LM diff eval cases gate on the provider key for CONSTRAINT_CHECK_MODEL.
  # The eval runs against the production judge (Claude Sonnet 5, hardcoded as
  # lm_config.CONSTRAINT_CHECK_LM) and self-skips if ANTHROPIC_API_KEY is absent.
  run_step "pytest: constraint_checker [claude-sonnet-5]" \
    env CONSTRAINT_CHECK_MODEL=anthropic/claude-sonnet-5 \
    python -m pytest tests/functional/test_constraint_checker.py -q

  # Finalization test-set sampling is harness-independent (no agent session;
  # four single-row evals of the FactChecker fixture). Runs once, gated on the
  # fixture program's provider key.
  if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    run_step "functional: test_sampling" python local_test_app.py test_sampling
  else
    skip_step "functional: test_sampling" "OPENAI_API_KEY unset"
  fi

  # architect_trace (capture) before meta_reflection (consumption).
  FUNCTIONAL_CMDS=(close_iteration session_resume comparison_credits analysis_agent inspect_traces architect_trace meta_reflection)

  for h in "${HARNESSES[@]}"; do
    reason="$(harness_skip_reason "$h")"
    if [[ -n "$reason" ]]; then
      skip_step "functional[$h]" "$reason"
      continue
    fi
    for cmd in "${FUNCTIONAL_CMDS[@]}"; do
      run_step "functional[$h]: $cmd" functional_cmd "$h" "$cmd"
    done
    # Behavioral holdout guard (held-out test set): probe both armed agents.
    run_step "functional[$h]: holdout_guard (inspection)" \
      functional_cmd "$h" holdout_guard --agent inspection
    run_step "functional[$h]: holdout_guard (architect)" \
      functional_cmd "$h" holdout_guard --agent architect
    # run_variance_estimate: a real session calls the zero-argument tool
    # against the FactChecker fixture (4 rows of its LM), so it also needs
    # the fixture program's key on the claude pass.
    if [[ -n "${OPENAI_API_KEY:-}" ]]; then
      run_step "functional[$h]: variance_tool" functional_cmd "$h" variance_tool
    else
      skip_step "functional[$h]: variance_tool" "OPENAI_API_KEY unset"
    fi
  done
fi

# ===========================================================================
# Phase 3 — EE-E2E (long-running; opt-in, functional phases only)
# ===========================================================================
if [[ "$RUN_FUNCTIONAL" == "1" ]]; then
  # One row per arm: <selected>|<label / subcommand>|<opt-in flag>.
  #
  # All five share one driver (tests/functional/test_e2e.py::run_e2e) and one
  # argument set, differing only in which optimization_entry.py runs, so the
  # bodies below are identical and the differences live in this table.
  #
  # Meta-reflection and run-integrity are ON by default in local_test_app.py,
  # matching runner/entrypoint.py::RESEARCH_DEFAULTS — no flag needed here.
  # Meta-reflection in particular is what gives the `opencode export`
  # architect-trace path end-to-end coverage, where a regression is otherwise
  # silent (a `captured:false` stub trace). Pass --no-<name> to
  # local_test_app.py directly for a cheaper one-off.
  #
  # The third inspector, clean-slate, is OFF here: the host reuses one cached
  # clone and keeps every past e2e run's artifacts under ~/.codeevolver/e2e/,
  # so the inspector would report on other runs and a `contaminated` verdict
  # would abort a healthy one. The Docker runner arms it (fresh container per
  # run); `--clean-slate` arms it locally for a deliberate one-off.
  #
  # pareto_evolution exercises loop-owned parent selection, the counted seed
  # comparison eval and the population admission path; simple_loop is research
  # Baseline 0 and has no clean-slate inspector of its own (that check reports
  # n/a).
  E2E_ARMS=(
    "$E2E_EE|ee_e2e|--include-ee-e2e"
    "$E2E_PARETO|pareto_e2e|--include-pareto-e2e"
    "$E2E_FREEFORM|freeform_e2e|--include-freeform-e2e"
    "$E2E_SIMPLE|simple_loop_e2e|--include-simple-e2e"
    "$E2E_ASA|asa_e2e|--include-asa-e2e"
  )

  if (( E2E_EE || E2E_PARETO || E2E_FREEFORM || E2E_SIMPLE || E2E_ASA )); then
    echo ""
    echo "=========================================================="
    echo " Phase 3 — end-to-end optimizer runs (long-running)"
    echo "=========================================================="
    for h in "${HARNESSES[@]}"; do
      reason="$(harness_skip_reason "$h")"
      for arm in "${E2E_ARMS[@]}"; do
        IFS='|' read -r selected cmd _flag <<< "$arm"
        [[ "$selected" == "1" ]] || continue
        if [[ -n "$reason" ]]; then
          skip_step "$cmd[$h]" "$reason"
        else
          run_step "$cmd[$h]" functional_cmd "$h" "$cmd"
        fi
      done
    done
  fi

  # Announce every arm that was NOT selected, once, so the summary shows the
  # full e2e inventory rather than silently omitting what did not run.
  for arm in "${E2E_ARMS[@]}"; do
    IFS='|' read -r selected cmd flag <<< "$arm"
    [[ "$selected" == "1" ]] && continue
    skip_step "$cmd" "long-running; pass $flag (or --include-e2e) to run"
  done
fi

# ===========================================================================
# Summary
# ===========================================================================
echo ""
echo "=========================================================="
echo " Summary"
echo "=========================================================="
for r in "${RESULTS[@]}"; do echo "  $r"; done
echo "=========================================================="
if [[ "$FAILED" == "1" ]]; then
  echo "RESULT: FAIL (one or more steps failed)"
  exit 1
fi
echo "RESULT: PASS"
exit 0
