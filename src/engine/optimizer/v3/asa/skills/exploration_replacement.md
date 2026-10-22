---
name: exploration
description: >
  Best practices for program exploration, 
  macro approaches, managing the candidate pool of lineage branches,
  and comparing branches on the fixed comparison set. Invoke when this
  iteration explores rather than exploits.
---

# Exploration Skill
During exploration, you are an AI architect evaluating new macro approaches that you think could be promising. Macro approaches are non-incremental changes which provide a fundamentally new solution, often distinct from or divergent from existing approaches. This often includes a different approach to the program's architecture. 

If the new program performs well, you should add it to your memory.md file as one of the macro lineages that you want to maintain for future iterations and future improvements.

### Step 1: Ideate and Plan
1. Study {memory_dir}/memory.md to learn about past architectures attempted and {workspace_path}/codeevolver.md to understand the system architecture for a given branch.
2. Ideate opportunities for improvement (exploitation if incremental) or new approaches (exploration).
3. Determine your plan for improvement.
4. Select which existing branch to start from and/or create a new git branch from the candidate pool. 
5. Before making edits, call the run_eval tool to baseline the branch on a training subsample, which will be set for the full iteration. The subsample size is set by the run config — you do not choose it. If your approach targets a specific slice of the data you may pass a filter to scope the subsample — e.g., filter={{"category": "metals"}} or filter={{"description": {{"contains": "steel"}}}}. A filter may yield fewer rows than the configured size when the matching pool is smaller. The sample and filter used for the baseline will be locked for the rest of this iteration. If the response comes back with baseline_accepted: false, reissue run_eval(eval_sample="train", ...) with a different filter (or drop it); the new call becomes the baseline.

### Phase 2: Make changes
- Make changes! either enact it yourself or with the `coding_agent`. You have two paths:
   - Prompt edits (rewriting an existing prompt instruction string in the codebase, no structural change)
   - Code / structural changes: call coding_agent(mode="execute", change_request=...) 
   - Note: If you edit directly via Edit / Write, then commit via Bash (`git add -A && git commit -m "iter N: <summary>"`) before calling run_eval. You MUST commit before re-evaluating. If you use the `coding_agent`, it handles the edit + commit cycle for you. Follow any constraints from the additional instructions below.
- You may find the trace meta useful for identifying failing or low-scoring samples, which you can inspect using the inspect_traces tool (`trace="parent"` for the baseline eval, `trace="child"` for your latest post-change eval). Traces often contain valuable examples to learn from how a program executes. Pass a specific, substantive query — the inspector is your eyes into the trace file.

### Phase 3: Measure and iterate
- Call run_eval on the new modified program and determine whether there is a positive progression.
- Iterate if needed, based on the progression. Keep iterations below 4 change rounds per iteration.
- If progression is negative and you feel the current eval did not do the program justice, you might try additional rounds of changes or you might also git reset --hard to discard the change and hand off to the next iteration.

Note: Measurements include variance, and your decisions should factor in that variance. 

### Phase 4: Closing
Refer to the main prompt for how to close - including calling the close_iteration_tool.

## General Guidelines
- Spend about ~30-50 percent of the iterations improving the best performing program. (Exploitation)
- Spend another ~30-50 percent of the iterations experimenting with a variety of macro approaches. (Exploration)
- When it makes sense to do so, consider merging two compatible macro approaches.
- For AI applications, make sure to balance changes made between code changes versus prompt changes (and sometimes both in tandem). If the memory.md file shows that you're relying too heavily on one or the other, make sure to attempt changes to the other as well.
- Avoid rerunning experiments that have been ruled out as not helpful for good reason (other than execution issues).
- Evaluations must be run only via the run_eval tool. NEVER run the metric, the program, or any eval script from Bash, Python, or any other shell. The run_eval tool is the sole source of valid scores, trace files, and feedback for this iteration. Attempts to invoke evaluation through other means will disqualify your run and will produce scores you cannot trust.
- You MAY inspect the training dataset (read rows, scan for patterns, study examples) to guide your hypothesis. You MAY NOT inspect the holdout / test set by any means. Treat the test set as a complete holdout set you never touch (even at the end) - this helps us prevent overfitting.
- Always respect the constraints in the additional section if available because that comes directly from the client.


### Explore Multiple Macro Approaches during Exploration
Macro approaches are non-incremental changes which provide a fundamentally new solution, often distinct from or divergent from existing approaches. This often includes a different approach to the program's architecture. 

Note that a new macro approach often compares a new, unoptimized approach to a past approach which may have already undergone many rounds of improvements and thus is already quite optimized. If the new macro approach is promising it may need a little more work before it reaches its full potential. 


## The Candidate Pool
If your new macro approach is sufficiently different from the existing methods, you should build it on its own branch and add it to your candidate pool, which you can track in the memory.md file. Use a new git branch when an approach is sufficiently new, and name it based on the macro approach, with the prefix `codeevolver-`. DO NOT TOUCH BRANCHES THAT DON'T HAVE THE PREFIX `codeevolver-`. 

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
If you believe your new candidate might outperform the primary candidate, we need to run a larger, fixed comparison set where we can evaluate branches apples-to-apples and see if a branch should be promoted as the primary candidate. This only applies when your baseline was not taken from the primary candidate. **IMPORTANT: when comparing two branches, given the high variance between different training sub-samples, we cannot compare scores from different training samples; we should use a fixed comparison set instead.** Before promoting a branch to be the new primary candidate you need an apples-to-apples comparison. The comparison set exists as a fixed eval to compare (use `run_eval` on the new branch with eval_sample = "comparison"). **Save the comparison run for the end!**

To compare this branch's comparison set to the 'incumbent' primary branch, check if the latest version of the primary candidate branch has already run a comparison set score - see scores in {history_dir}/comparison_set_history.jsonl file. Otherwise, you may need to check out the incumbent primary branch and re-run the comparison set (eval_sample = "comparison") as well before declaring a new primary candidate.

Note: Comparison runs are limited by credits that accumulate over your iterations. Check replenishment rate with the run_eval tool. Even at one comparison credits per iteration, you have enough budget if you save. There are two strategies:
1. Save credits for running both the new and incumbent branch at promotion time
2. Save the comparison set run for the end of every primary iteration so it's always up to date. And run on challenger branches once when you need to promote. 

You can view your credits by calling run_eval(eval_sample="comparison") (You'll be prompted with an acknowledgement note with a code. Then you recall run_eval with the code in the `acknowledge_message` argument).

Note: While the score in the comparison set provides guidance for promotion, do also use logic as to which one to keep as primary (e.g., ask which will generalize better). And note that our comparison set is drawn from the same set as the training set, which means one branch could have a slight advantage if it trained on more of the comparison set rows than the other. 

IMPORTANT: it is best practice to wait to run the comparison set until the end of your improvement run. This is because there are limited credits and future agents will select which branch to work on based on the results of the comparison set, including whether it demonstrates behavioral advantages over other branches (via the branching tool). 

Always commit before running the comparison set and run it using `run_eval(eval_sample="comparison")`. (If the tree is dirty, run_eval commits it for you before a comparison eval so the recorded sha is the evaluated content.) The first comparison call of an iteration returns an acknowledgement note with a verification code — re-call run_eval with `acknowledge_message` set to that code to proceed. Comparison runs are refused when no credits remain; credits replenish at the start of each iteration.