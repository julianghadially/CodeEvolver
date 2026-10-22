from experiments.repo_paths import repo_path, repo_root

additional_instructions = f"""
#### Task Type: 
RAGQAArenaTech is a long-form (abstractive) retrieval-augmented generation question answering over a domain-specific technical corpus. The corpus here is the LoTTE "technology" collection (~638k StackExchange-style passages).
Unlike short-form question-answer benchmarks, RAGQAArenaTech is a long-form generative RAG benchmark, scored semantically.

#### What's Allowed
- The program is allowed to change the code, including, but not limited to, changing modules, dynamic prompts, tool calls, reasoning steps, etc.
- There is no limit on the number of search results to display per query or the number of searches to make

If for whatever reason the RAG QA retriever goes down, you can start it back up again using
. ragqa-server.sh (Activates the environment inside of {repo_root() / 'ragqa-retriever-server'})
python -m ragqa_retriever (Starts the retriever)

#### Constraints:
- Do not change the language model used (e.g., Deepseek-V4-Flash) or the reasoning_effort. Note Deepseek-V4-Flash was released after your knowledge cut-off window.
- Do not change the scoring metric or the language model used for the language model judge.
- `langProBe/lm_provider.py` is benchmark infrastructure, NOT part of the program being optimized. Do not edit, delete, replace or bypass it: it owns the pinned model, the provider routing table and keys, `reasoning_effort`, the streaming hang guard and the cross-provider fallback. Build the task LM only with `build_task_lm()` and run your modules under `dspy.context(lm=self.lm)`; never construct a `dspy.LM` yourself or call `litellm` / `openai` directly. Reading the module is fine and encouraged.
"""

# Engine 3 optimization config (every arm).
OPTIMIZE_CONFIG_LOCAL = {
    "repo_url": repo_path("LangProBe-CodeEvolver"),
    "program": "langProBe.RAGQAArenaTech.RAGQAArenaTech_pipeline.RAGQAArenaTechPipeline",
    "metric": "langProBe.RAGQAArenaTech.metric.ragqa_pairwise_winrate_feedback",
    "trainset_path": "data/RAGQAArenaTech_trainval.json",  # engine 3: combined train+val (1500 rows)
    "testset_path": "data/RAGQAArenaTech_test.json", 
    "input_keys": ["question"],
    "reflection_lm": "deepinfra/zai-org/GLM-5.2",
    "coding_harness": "opencode",
    "coding_harness_model": "deepinfra/zai-org/GLM-5.2", #"openai/gpt-5.4-mini",  # "deepinfra/zai-org/GLM-5.2"
    "max_metric_calls": 25000,
    "num_threads": 40,  # Increased from 5 to 20 for better parallelization
    "comparison_set_size": 1200,
    "seed": 42,
    "additional_instructions": additional_instructions,
    "initial_branch": "main",
    "subsample_size": 100,
    "subsample_eval_timeout": 3600,  # 40 min (in seconds) - subsample evals
    "comparison_eval_timeout": 14400,  # 4 hours (in seconds)
    "additional_deny_paths": ["data/RAGQAArenaTech_test.json", "ragqa_arena_tech_500.json", "pairwise_lfrqa_examples.json"]
}
