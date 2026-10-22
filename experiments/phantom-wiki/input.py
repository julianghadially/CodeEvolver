'''
Controls
see additional instructions

Future:
- RLMs are great at long-context reasoning and needle-in-the-haystack problems especially when the context is very large, and a normal model would suffer from context rot. Try dspy.RLM that has access to search the wiki. wiki search can be passed as a tool, but make sure RLM has enough iterations to explore search results thoroughly.
'''

from experiments.repo_paths import repo_path

additional_instructions = """
#### Task Type: Document Retrieval
This is a document retrieval + Question Answering task. The metric measures F1 score - which includes the precision of predicted answers as well as the recall of all available answers. Some questions have multiple correct answers.

To score high, the system must be able to reason over multiple retrieval steps. Additionally, many questions have multiple correct answers, so the system must also be able to reason over multiple chains of logic simultaneously. 

The core difficulty of Phantom Wiki is the large amount of information that must be processed across multiple chains of logic. Additionally, if context accumulates across multiple chains of logic and the context is too long, the model can suffer from context rot.

#### What's Allowed
- The program is allowed to change the code, including, but not limited to, changing modules, dynamic prompts, tool calls, reasoning steps, retrieval techniques, etc. Retrieval methods are allowed unless they hardcode the answer.
- There is no limit on the number of search results to display per query or the number of searches to make

#### Constraints:
- Do not change the language model used. It must remain DeepSeek-V4-Flash.
- Every evaluated row must be answered by the LLM. At least one LM call must contribute to every prediction. A code path that returns an answer for any row without calling the LLM is a violation, even if an LLM path exists elsewhere as a fallback. A code path that reduces the LLM to formatting, verifying, or selecting a result that code computed is a violation.
- You may not implement a solver, resolver, or rules engine that derives the answer.
- `langProBe/lm_provider.py`: Do not edit, delete, replace or bypass it: it owns the pinned model, the provider routing table, `reasoning_effort`, and the cross-provider fallback. Build task LMs with `build_task_lm()` and run your modules under `dspy.context(lm=self.lm, rm=self.rm)`

#### Ideas
- Try to address the multiple chains of logic problem by breaking up investigation paths into multiple sub-agents.
- Try adding reasoning steps and/or structured thinking and/or logic guidance via new modules in the workflow.
- Try creating a secondary workspace to jot down persistent reasoning logic that multiple agents can add to or remove as they interact with more documents - but make sure they actually use it, using the prompts!
- Try explicitly managing multiple chains of logic in the AI workflow architecture (e.g., modules or workspace file for work-in-progress reasoning logic).
- Try iterative search methods.
- Try increasing the maximum number of retrieval steps
- Try modifying the number of documents returned per query.

### Available Services
The following external services are available with API keys already configured in the environment:
- PhantomWiki colbert-server (Via dspy.Retrieve)

### Additional notes
- Many answers are surfaced over multiple steps of reasoning and retrieving (i.e., Pure parallel querying is unlikely to work without multiple retrieval steps).
- Different questions require different numbers of retrieval steps.
- Do not attempt to use RLMs for this task.


"""

OPTIMIZE_CONFIG = {}
OPTIMIZE_CONFIG_LOCAL = {
    "repo_url": repo_path("phantom-wiki"),
    "program": "src.program.phantomwiki_pipeline.PhantomWikiReActPipeline",
    "metric": "src.metric.metric.phantomwiki_f1_feedback",
    "trainset_path": "data/phantomwiki_trainval_omitsuperlong.json",  # engine 3: combined train+val (228 rows)
    "testset_path": "data/phantomwiki_test_omitsuperlong.json",
    "input_keys": ["question"],
    "reflection_lm": "deepinfra/zai-org/GLM-5.2", #"anthropic/claude-sonnet-4-6",
    "coding_harness": "opencode", #"claude",
    "coding_harness_model": "deepinfra/zai-org/GLM-5.2", #"anthropic/claude-sonnet-4-6",
    "max_metric_calls": 4500, #Provider reliability issues require shifting timeout from 2400 -> 3000, and we're capped at a 72 hr limit per bm. Decreasing total calls by 10% to make up the difference.
    "num_threads": 25,
    "comparison_set_size": 100,
    "seed": 42,
    "additional_instructions": additional_instructions,
    "initial_branch": "main",
    "subsample_size": 50,
    "subsample_eval_timeout": 3000,
    "comparison_eval_timeout": 14400,
}

