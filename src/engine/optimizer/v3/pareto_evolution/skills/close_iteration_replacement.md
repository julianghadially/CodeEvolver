---
name: close_iteration
description: >
  How to finalize an iteration, including running the comparison set and saving reflections to memory. Invoke this skill before exiting the iteration. 
---

# Close Iteration Skill
Once your planned improvement(s) are complete along with any follow-on changes you want to make, it is time to finalize the iteration. 

1. Make sure the program is **committed** at the point you want to finalize. If you did not like your changes, `git reset --hard` to where you want to leave the branch. The loop continues improving from wherever you leave HEAD.
2. Once committed, decide whether this branch should join the candidate population. If yes, run the comparison set eval (`run_eval(eval_sample="comparison")`) on that committed state. The comparison-set eval is the sole admission criterion: it is what admits your branch to the population and decides which branch the next agent works on, and it is how future agents decide whether to merge ideas from your branch. A training result that did not improve within the noise floor is still a legitimate reason to run it — the population ranks by comparison score, not by your training delta. If you skip it, the branch is not a candidate (unless your best training eval strictly beat the baseline, in which case the optimizer runs the comparison for you at close). If the change is clearly worse, `git reset --hard` and skip the comparison set.
    - **The comparison run is the close step: only one is permitted per iteration, and after it runs no further evals are accepted.** Do NOT modify program files after it — any program changes made after the comparison run are reverted back to the auto-committed comparison-run point. Editing `{memory_dir}/memory.md` is fine — it is outside the repository.
3. Before exiting, **update memory** at the {memory_dir}/memory.md file with key metrics and learnings (See "Saving to Memory" below), including your branch name and its comparison-set score.
4. Finally, call the close_iteration tool to finalize the iteration. 

Do not rewrite `{workspace_path}/codeevolver.md` yourself — a post-iteration agent updates it to describe the program on your branch after you close.

## Close Iteration Tool
Call the close_iteration tool to finalize the iteration, and one field:
- `memory_updated`: true / false

You do not register a branch: the optimizer already knows which branch this iteration ran on and records its scores itself. Stay on the branch you were given — the tool will refuse to close if you are on a different branch or have uncommitted changes.

## The Population
The candidate population is managed by the optimizer from comparison-set scores. Per-branch change descriptions live in {memory_dir}/iteration_change_summaries.md and comparison-set results in {history_dir}/comparison_set_history.jsonl; use these together with memory.md when deciding what to try next or which branch's ideas to port over.

Run the comparison set once per iteration, at the end, since it consumes budget!

## Saving to Memory

Reflect on the iteration and record in `/memory/memory.md` for future agents to benefit.

Document the memories you find the most important. Generally speaking it is useful to share the approach you took, the overall progression on evaluations, observations, and possibly ideas. If there are concerns, note them down and how to avoid.

Keep memories concise and to the point. A future agent can inspect the branch for details. Leave room for other iterations.

Note: **Delete or tidy-up stale content** - Move unhelpful old memories to old_memory.md. The file gets really big after ~15+ iterations! It should never exceed 150k characters

## Notes

- Memory location: add your reflection to `{memory_dir}/memory.md`. memory.md functions as cumulative memory across iterations.
- Git branches: your branch already contains the codeevolver- prefix. Keep it that way. DO NOT MODIFY BRANCHES THAT DON'T HAVE THE PREFIX `codeevolver-`. 
- Comparison set runs are called using run_eval(eval_sample="comparison"). You'll be prompted with an acknowledgement note with a code. Then you recall run_eval with the code in the `acknowledge_message` argument.