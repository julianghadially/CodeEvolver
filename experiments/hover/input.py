'''
Controls
- search count
- final document count
- reasoning variable controlled by allowing chain of thought reasoning
- The program is limited to Wikipedia documents.
Not controlled
- module creation, removal, and modification
- # of database documents retrieved per search (same cost)

Future:
- Consider different search architectures (iterative search, Evidence workspace, Searching specific websites (e.g., Wikipedia))
'''

from experiments.repo_paths import repo_path

additional_instructions = """
## Hover
Hover is designed to retrieve information from 2017 Wikipedia Abstracts (5.9M) using multi-hop retrieval.

#### Task Type: Document Retrieval
This is a document retrieval task. The metric measures whether the correct supporting documents are retrieved — it does NOT measure claim verification or fact-checking accuracy. Do not add claim verification, fact-checking, or classification modules. Focus on improving the primary metric - i.e., retrieval recall and precision.

#### What's Allowed
- The program is allowed to create or remove modules
- The program is allowed to create or remove dynamic prompts
- The program is allowed to add rerankers, provided the final document count remains the same - 21 final outputted documents
- There is no limit on the number of search results to retrieve per query (same cost)
- Available services: wikipedia colbert-server (Via dspy.Retrieve)

#### Constraints:
- The final output of the system CANNOT return more than 21 documents. This is a hard requirement.
- Do NOT use any external websearch services.
- Do not change the language model used for the task. It must remain DeepSeek v4 Flash
- `langProBe/lm_provider.py`: Do not edit, delete, replace or bypass it: it owns the pinned model, the provider routing table, `reasoning_effort`, and the cross-provider fallback. Build task LMs with `build_task_lm()` and run your modules under `dspy.context(lm=self.lm, rm=self.rm)`


### Available Services

The following external services are available with API keys already configured in the environment:

### Wikipedia colbert-server (Via dspy.Retrieve)
- **Documentation**: https://github.com/julianghadially/colbert-server
- **Use cases**:
  - Retrieve information from Wikipedia abstracts
  - Useful for fact-checking and information retrieval
- **Python usage**: `from dspy import Retrieve`

## Ideas for Optimization
- Consider increasing the k retrieved per query, and then reranking the final results to 21 (final output limited to 21 documents)
- Consider different context retrieval pipelines, including different kinds of rerankers (list rerankers, score-based reranking, pairwise rerankers, etc.)
- Consider a gap analysis before generating queries
- Consider modifying the architecture and retrieval pipelines (e.g., using more hops, more queries, react agent for adaptive retrieval).

Additional notes:
- If you increase k retrieved per query, do not increase it greater than 25 per query as it may overload the Colbert server.
"""

OPTIMIZE_CONFIG = {}

# Engine 3 optimization config. Schema of record:
# src/engine/optimizer/v3/optimization_config.py (wire schema:
# src/api/schemas/requests.py::OptimizeRequest). Engine 3 removed the
# validation set: valset_path / valset_json / valset_bar / max_valset_size are
# rejected outright, and the architect compares branches on a fixed
# comparison set drawn from the trainset instead.
OPTIMIZE_CONFIG_LOCAL = {
    "repo_url": repo_path("LangProBe-CodeEvolver"),
    "program": "langProBe.hover.hover_pipeline.HoverMultiHopPipeline",
    "metric": "langProBe.hover.hover_utils.discrete_retrieval_eval_with_resource_penalty_and_feedback",
    "trainset_path": "data/hoverBench_trainval.json",  # engine 3: combined train+val (450 rows)
    "testset_path": "data/hoverBench_test.json",
    "input_keys": ["claim"],
    "reflection_lm": "deepinfra/zai-org/GLM-5.2", #"anthropic/claude-sonnet-4-6",
    "coding_harness": "opencode", #"claude",
    "coding_harness_model": "deepinfra/zai-org/GLM-5.2", #"anthropic/claude-sonnet-4-6",
    # Engine 3 budget accounting: every architect eval row counts here —
    # subsamples AND comparison-set runs. Only the seed step's sanity rows and
    # the finalization test-set eval are free. ~50 rows/subsample x 2 per
    # iteration + a 100-row comparison run when the architect compares
    # branches.
    "max_metric_calls": 9000,
    "num_threads": 25,  # parallel rows per eval (clamped to 128 and to the row count)
    # Fixed comparison set: 100 of the 450 trainset rows, drawn with seed 42.
    # Overlaps the training pool by design (engine 3 has no valset). Omitting
    # this falls back to the server default of 50 (src/config.py).
    "comparison_set_size": 200,
    "seed": 42,
    "additional_instructions": additional_instructions,
    "initial_branch": "main",
    "subsample_size": 50,
    # Hard backstop (seconds) on ONE subsample eval subprocess — engine 3
    # redefines this as a hung-program killer, not a performance bound. It is
    # deliberately generous: a subprocess timeout raises before any score comes
    # back, so a too-tight cap throws away the whole subsample's provider spend
    # and hands the architect a tool error instead of a number. Evolved
    # pipelines fan out to several retrieval streams per claim.
    "subsample_eval_timeout": 7200,  # 2 hours - matches the engine 3 default
    "comparison_eval_timeout": 14400,  # 4 hours - comparison-set + finalization test evals
    # Post-run meta-reflection (advisory, non-fatal). Explicit because the
    # default differs by entry point: the API defaults it True, while
    # OptimizationConfig / local_launcher default it False.
    "enable_meta_reflection": True,
    # No additional_deny_paths: testset_path is fenced automatically (full
    # path + basename + the `_test.jsonl` trace suffix) by
    # _build_holdout_deny_paths, and no other file in the client repo carries
    # the holdout claims (verified: zero claim overlap between
    # hoverBench_test.json and dev/train/val/trainval). trainval IS the
    # trainset, so fencing it would fence the agent's own training data.
}
