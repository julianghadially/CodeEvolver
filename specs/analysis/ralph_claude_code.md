# Ralph Claude Code Analysis

Analysis of [frankbria/ralph-claude-code](https://github.com/frankbria/ralph-claude-code) for autonomous agent patterns applicable to CodeEvolver.

## Executive Summary

Ralph is a bash-based autonomous development loop that invokes the **Claude Code CLI directly** (not the Python SDK). It does NOT:
- Block plan mode
- Implement a user proxy
- Use `--dangerously-skip-permissions`

Instead, Ralph uses a fundamentally different architecture:
1. **CLI invocation** with `--allowedTools` restrictions
2. **JSON output parsing** for structured status detection
3. **External loop control** via bash script
4. **Circuit breaker** for stuck loop detection

This is architecturally different from CodeEvolver's Python SDK approach, making direct adoption impractical. However, several patterns are valuable.

## Ralph Architecture

### Core Loop Mechanism

```
┌─────────────────────────────────────────────────────────────────┐
│  ralph_loop.sh (bash)                                            │
│                                                                  │
│  while true:                                                     │
│    1. Read .ralph/PROMPT.md                                      │
│    2. Build CLI command with --allowedTools, --output-format json│
│    3. Execute: claude -p "$prompt" [flags]                       │
│    4. Parse JSON response for EXIT_SIGNAL                        │
│    5. Check circuit breaker (stuck detection)                    │
│    6. If EXIT_SIGNAL=true AND completion_indicators≥2: break     │
│    7. Sleep, repeat                                              │
└─────────────────────────────────────────────────────────────────┘
```

### Claude CLI Invocation

Ralph builds a command like:
```bash
claude \
  --output-format json \
  --allowedTools Write Read Edit "Bash(git *)" "Bash(npm *)" \
  --resume $SESSION_ID \
  --append-system-prompt "$loop_context" \
  -p "$prompt_content"
```

Key flags:
- `--output-format json` - Structured responses for parsing
- `--allowedTools` - Restricts Claude's tool access
- `--resume $SESSION_ID` - Session continuity between loops
- `-p` - Prompt content from file

**Notably absent**: `--dangerously-skip-permissions` is explicitly NOT used (code comments state this is intentional for preserving permission denial detection).

### Why Ralph Doesn't Need Plan Mode Blocking

1. **External Task Queue**: Ralph uses `.ralph/fix_plan.md` as the task list, not Claude's internal planning
2. **One Task Per Loop**: Prompt instructs "focus on ONE task per loop"
3. **Structured Output**: Claude must output a `---RALPH_STATUS---` block with EXIT_SIGNAL
4. **Loop Detection**: If Claude gets stuck, circuit breaker halts after 3 no-progress loops

The bash loop IS the plan execution - Claude just executes one step at a time.

### Dual-Condition Exit Gate

Ralph requires BOTH conditions to exit:
```python
if recent_completion_indicators >= 2 AND EXIT_SIGNAL == true:
    exit_loop()
```

This prevents premature exits from natural language patterns like "done" or "complete".

### Circuit Breaker Thresholds

| Condition | Threshold | Action |
|-----------|-----------|--------|
| No progress | 3 loops | HALF_OPEN state |
| Same error | 5 occurrences | OPEN state |
| Permission denied | 2 loops | OPEN state (critical) |

## Comparison: Ralph vs CodeEvolver

| Aspect | Ralph | CodeEvolver |
|--------|-------|-------------|
| **Interface** | Claude CLI (bash) | Claude Agent SDK (Python) |
| **Loop Control** | External bash script | SDK's internal execution |
| **Plan Mode** | N/A (external task queue) | Must be blocked via hooks |
| **Permissions** | `--allowedTools` restriction | `bypassPermissions` + hooks |
| **Output Format** | JSON with status block | SDK message stream |
| **Session** | `--resume` with session ID | SDK session management |

### Why Direct Adoption Won't Work

1. **SDK vs CLI**: CodeEvolver uses the Python SDK which spawns the CLI internally - we can't inject CLI flags
2. **Single-shot Execution**: CodeEvolver runs one agent invocation per mutation, not a loop
3. **Permission Model**: SDK's `bypassPermissions` doesn't prevent `EnterPlanMode`/`ExitPlanMode`

## Applicable Patterns

### 1. Structured Status Output (Adopt)

Ralph requires Claude to output a structured status block:
```
---RALPH_STATUS---
STATUS: IN_PROGRESS
EXIT_SIGNAL: false
WORK_TYPE: IMPLEMENTATION
FILES_MODIFIED: 3
---END_RALPH_STATUS---
```

**CodeEvolver could adopt**: Require structured output at end of each mutation, parsed to detect completion/failure.

### 2. Circuit Breaker (Adopt)

Track consecutive failures and halt before wasting resources:
- No file changes for N iterations
- Same error repeated M times
- Agent stuck in a loop (e.g., repeated ExitPlanMode calls)

**Implementation for CodeEvolver**:
```python
class AgentCircuitBreaker:
    no_progress_count = 0
    same_error_count = 0

    def check(self, result: AgentResult) -> bool:
        if not result.has_changes:
            self.no_progress_count += 1
        else:
            self.no_progress_count = 0

        if self.no_progress_count >= 3:
            raise CircuitBreakerOpen("No progress in 3 iterations")
```

### 3. Task Queue File (Consider)

Ralph uses `.ralph/fix_plan.md` as an external task queue that Claude reads and updates.

**Could apply to CodeEvolver**: Write change requests to `codeevolver_tasks.md`, have agent mark complete.

### 4. Explicit Completion Signal (Adopt)

Don't rely on heuristics - require explicit completion declaration:
```python
# In system prompt:
"When your changes are complete, output: MUTATION_COMPLETE: true"

# In parser:
if "MUTATION_COMPLETE: true" in output:
    return MutationResult(success=True)
```

## Recommendation for CodeEvolver

The hooks-based approach (blocking `EnterPlanMode`/`ExitPlanMode`/`AskUserQuestion`) remains the correct solution for the SDK. Ralph's CLI-based approach is architecturally incompatible.

However, adopt these Ralph patterns:

### Immediate (v1)

1. **Keep hooks approach** - Block plan mode tools at SDK level (already implemented)
2. **Add circuit breaker** - Detect stuck agents (repeated no-progress, same errors)
3. **Require completion signal** - Add to system prompt, parse from output

### Future (v2)

4. **Structured status output** - JSON block at end of agent output for better parsing
5. **External task queue** - Consider `codeevolver_tasks.md` pattern for complex mutations

## Key Takeaways

1. **Ralph doesn't solve the plan mode problem** - It avoids it through CLI architecture
2. **No user proxy exists** - Ralph is fully autonomous with no human-in-the-loop
3. **Hooks are correct for SDK** - Plan mode blocking via PreToolUse hooks is the right approach
4. **Circuit breaker is valuable** - Adopt for detecting stuck agents
5. **Explicit signals > heuristics** - Require completion declarations, don't guess

## References

- [Ralph Claude Code Repository](https://github.com/frankbria/ralph-claude-code)
- [Claude Agent SDK Hooks](https://platform.claude.com/docs/en/agent-sdk/hooks)
- [Claude Agent SDK Permissions](https://platform.claude.com/docs/en/agent-sdk/permissions)
- [Original Ralph Technique by Geoffrey Huntley](https://ghuntley.com/ralph/)
