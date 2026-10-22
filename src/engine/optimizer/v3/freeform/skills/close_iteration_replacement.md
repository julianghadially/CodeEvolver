---
name: close_iteration
description: >
  How to finalize an iteration, including saving reflections to memory. Invoke this skill before exiting the iteration. 
---

# Close Iteration Skill
Once your planned improvement(s) are complete along with any follow-on changes you want to make, it is time to finalize the iteration. 

{final_iteration_note}

1. Make sure the program is committed at the point you want to finalize. If you did not like your changes, `git reset --hard` to where you want to leave the branch. The loop continues improving from wherever you leave HEAD, and auto-commits any dirty branch states.
2. Before exiting, **update memory** at the {memory_dir}/memory.md file with key metrics and learnings (See "Saving to Memory" below).
3. Finally, call the close_iteration tool to finalize the iteration. 

Do not rewrite `{workspace_path}/codeevolver.md` yourself — a post-iteration agent updates it to describe the program on your branch after you close.

## Close Iteration Tool
Call the close_iteration tool to finalize the iteration, and one field:
- `memory_updated`: true / false

## Saving to Memory

Reflect on the iteration and record in `/memory/memory.md` for future agents to benefit.

Document the memories you find the most important. Generally speaking it is useful to share the approach you took, the overall progression on evaluations, observations, and possibly ideas. If there are concerns, note them down and how to avoid.

Keep memories concise and to the point. A future agent can inspect the branch for details. Leave room for other iterations.

Note: **Delete or tidy-up stale content** - Move unhelpful old memories to old_memory.md. The file gets really big after ~15+ iterations! It should never exceed 150k characters

## Notes

- Memory location: add your reflection to `{memory_dir}/memory.md`. memory.md functions as cumulative memory across iterations.
- Git branches: your branch must contain the codeevolver- prefix. DO NOT MODIFY BRANCHES THAT DON'T HAVE THE PREFIX `codeevolver-`. 
