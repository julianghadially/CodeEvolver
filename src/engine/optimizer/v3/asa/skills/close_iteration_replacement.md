---
name: close_iteration
description: >
  How to finalize an iteration, including saving reflections to memory. Invoke this skill before exiting the iteration. 
---

# Close Iteration Skill
Once your planned improvement(s) are complete along with any follow-on changes you want to make, it is time to finalize the iteration. 

1. Make sure the program is **committed** at the point you want to finalize. If you did not like your changes, `git reset --hard` to where you want to leave the branch. The loop continues improving from wherever you leave HEAD.
2. If your changes made meaningful progress and this branch is likely to stay or get added to the candidate pool, you should consider running the comparison set eval. See "Comparison Set Usage" below.
3. Before exiting, **update memory** at the {memory_dir}/memory.md file with key metrics and learnings (See "Saving to Memory" below) including **adding your branch to the candidate pool** if you want future agents to have access to it.
4. Finally, call the close_iteration tool to finalize the iteration. 

Do not rewrite `{workspace_path}/codeevolver.md` yourself — a post-iteration agent updates it to describe the program on your branch after you close.

## Close Iteration Tool
Call the close_iteration tool to finalize the iteration, and pass three fields:
- `register_branch`: Record the branch that you are submitting your improvements on. Record it using the json below. 
    - `was_primary`: What if the branch was the primary branch at the start of the run 
    - `is_primary`: Whether the branch is the primary branch after your changes have hit. If it is newly promoted to be the primary branch, you need to provide evidence for promotion. (see "Promoting a branch to the primary branch" below) 
    - `promotion_evidence`:  Evidence for promotion. You can reference comparison set scores here.   
- `improvement_mode` (exploration / exploitation)
- `memory_updated`: true / false

`register_branch` is a JSON object with the following format:
```json
{  
   "branch": "insert branch name",
   "was_primary": "True / False",
   "is_primary": "True / False",
   "promotion_evidence": "insert rationale / n/a"
}
```


## The Candidate Pool
If your new macro approach is sufficiently different from the existing methods, you should build it on its own branch and add it to your candidate pool, which you can track in the memory.md file. Use a new git branch when an approach is sufficiently new, and name it based on the macro approach, with the prefix `codeevolver-`. 

At any given time, there is one branch that is the "primary" / best branch. Before your run, you were provided a primary branch. If a new branch should be promoted to primary, you may update it at the end of your run by passing the appropriate branch as primary when you call the close_iteration tool. The primary branch is important because future exploit-focused agents will focus on improving the primary branch whenever they decide to exploit the best program. It's also important because the final program is the latest primary. Do NOT set a non-best program as primary.

Branches you do not plan to continue working on can be removed from the candidate pool that you're tracking. (This doesn't mean deleting the branch just removing them from the candidate pool that you're tracking in the memory.md file). If a branch is useful for future experiments, keep it in the pool! Note that the loop keeps recognizing any registered branch that still has positive behavioral contribution as an eligible parent, whether or not it is in your memory.md pool — dropping a branch from the pool does not retire it.

Source of truth: the memory.md file is the source of truth for organizing which branches are still in the candidate pool. The close_iteration tool is the sole place to set the primary branch for a future agent.

### Non-primary branches
A non-primary branch is only valuable if it offers a positive behavioral advantage that the primary branch does not offer. You can measure this behavioral advantage using the `branch_comparison` tool (see below).

Once there is a non-primary branch with sufficient positive behavioral advantage, it becomes worthwhile to explore improvements to it, especially when that branch takes a meaningfully different approach.

For the period of time where non-primary branches have meaningfully positive behavioral advantage, you should consider dedicating iteration(s) to them some fraction of the time, even if their overall score is lower than the primary branch. The seed program is kept as its own lineage (`codeevolver-<run>-seed`) for exactly this reason.

### Branch comparison tool
The `branch_comparison` tool computes positive behavioral contribution between branches and emits Pareto scores per branch. It is read-only and costs no eval rows.
- Input arguments: pass a reference branch (usually the primary branch). The tool will compute behavioral contributions against this branch.
- Positive behavioral contribution is the summation of positive row-wise score differences on the comparison set between two branches. It does not include rows where the branch is inferior - use the overall comparison set scores for that.
- Candidates are entire branch lineages and utilize the most recent comparison set taken for that branch.
- The tool will note whenever the comparison set is out of date for a branch. That is okay.
- A branch with no comparison-set record cannot be compared (or become an eligible parent) until you run `run_eval(eval_sample="comparison")` on it.

### Promoting a branch to the primary branch / Comparison Set Usage
If you believe your new candidate might outperform the primary candidate, we need to run a larger, fixed comparison set where we can evaluate branches apples-to-apples and see if a branch should be promoted as the primary candidate. This only applies when your baseline was not taken from the primary candidate. **IMPORTANT: when comparing two branches, given the high variance between different training sub-samples, we cannot compare scores from different training samples; we should use a fixed comparison set instead.** Before promoting a branch to be the new primary candidate you need an apples-to-apples comparison. The comparison set exists as a fixed eval to compare (use `run_eval` on the new branch with eval_sample = "comparison"). 

To compare this branch's comparison set to the 'incumbent' primary branch, check if the latest version of the primary candidate branch has already run a comparison set score - see scores in {history_dir}/comparison_set_history.jsonl file. Otherwise, you may need to check out the incumbent primary branch and re-run the comparison set (eval_sample = "comparison") as well before declaring a new primary candidate.

Note: Comparison runs are limited by credits that accumulate over your iterations. Check replenishment rate with the run_eval tool. Even at one comparison credits per iteration, you have enough budget if you save. There are two strategies:
1. Save credits for running both the new and incumbent branch at promotion time
2. Save the comparison set run for the end of every primary iteration so it's always up to date. And run on challenger branches once when you need to promote. 

You can view your credits by calling run_eval(eval_sample="comparison") (You'll be prompted with an acknowledgement note with a code. Then you recall run_eval with the code in the `acknowledge_message` argument).

Note: While the score in the comparison set provides guidance for promotion, do also use logic as to which one to keep as primary (e.g., ask which will generalize better). And note that our comparison set is drawn from the same set as the training set, which means one branch could have a slight advantage if it trained on more of the comparison set rows than the other. 

IMPORTANT: it is best practice to wait to run the comparison set until the end of your improvement run. This is because there are limited credits and future agents will select which branch to work on based on the results of the comparison set, including whether it demonstrates behavioral advantages over other branches (via the branching tool). 

Always commit before running the comparison set and run it using `run_eval(eval_sample="comparison")`. (If the tree is dirty, run_eval commits it for you before a comparison eval so the recorded sha is the evaluated content.) The first comparison call of an iteration returns an acknowledgement note with a verification code — re-call run_eval with `acknowledge_message` set to that code to proceed. Comparison runs are refused when no credits remain; credits replenish at the start of each iteration.

## Saving to Memory

Reflect on the iteration and record in `/memory/memory.md` for future agents to benefit.

Document the memories you find the most important. Generally speaking it is useful to share the approach you took, the overall progression on evaluations, observations, and possibly ideas. If there are concerns, note them down and how to avoid.

Track the top branches that are worth maintaining in the candidate pool section of the memory.md file.

Keep memories concise and to the point. A future agent can inspect the branch for details. Leave room for other iterations.

Note: **Delete or tidy-up stale content** - Move unhelpful old memories to old_memory.md. The file gets really big after ~15+ iterations! It should never exceed 150k characters

## Notes

- Memory location: add your reflection to `{memory_dir}/memory.md`. memory.md functions as cumulative memory across iterations.
- Git branches: your branch must contain the codeevolver- prefix. DO NOT MODIFY BRANCHES THAT DON'T HAVE THE PREFIX `codeevolver-`. 