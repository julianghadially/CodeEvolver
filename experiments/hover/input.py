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

additional_instructions = """
## Hover
Hover is designed to retrieve information from 2017 Wikipedia Abstracts (5.9M) using multi-hop retrieval.

#### Task Type: Document Retrieval
This is a document retrieval task. The metric measures whether the correct supporting documents are retrieved — it does NOT measure claim verification or fact-checking accuracy. Do not add claim verification, fact-checking, or classification modules. Focus on improving retrieval recall and precision through better queries, re-ranking, or retrieval strategies.

#### What's Allowed
- The program is allowed to create or remove modules
- The program is allowed to create or remove dynamic prompts
- The program is allowed to add rerankers, provided the final document count remains the same - 21 final outputted documents
- There is no limit on the number of search results to retrieve per query (same cost)
- Available services: wikipedia colbert-server (Via dspy.Retrieve)

#### Constraints:
- Do NOT search more than three queries per claim. This is a hard requirement.
- Again, do not search more than three queries per claim. This is a hard requirement
- The final output of the system CANNOT return more than 21 documents. This is a hard requirement.
- Do NOT use any external websearch services.


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

Additional notes:
- If you increase k retrieved per query, do not increase it greater than 25 per query as it may overload the Colbert server.
"""

OPTIMIZE_CONFIG = {
    "repo_url": "https://github.com/julianghadially/LangProBe-CodeEvolver",
    "program": "langProBe.hover.hover_pipeline.HoverMultiHopPipeline",
    "metric": "langProBe.hover.hover_utils.discrete_retrieval_eval",
    "trainset_path": "data/hoverBench_train.json", # data/FacTool_QA_train_normalized.jsonl
    "valset_path": "data/hoverBench_val.json", # data/FacTool_QA_train_normalized.jsonl
    "input_keys": ["claim"],
    "reflection_lm": "openai/gpt-4.1-mini",
    "max_metric_calls": 2000,  # 60 examples × 30 full evals = 1800 (with subsampled valset)
    "num_threads": 8,  # Increased from 5 to 20 for better parallelization
    "max_valset_size": 60,  # Subsample validation set to 150 examples (from 300) for faster evaluation
    "seed": 42,
    "additional_instructions": additional_instructions,
    "initial_branch": "hover",
    "decay_rate": 30,
    "subsample_size": 20,
    "subsample_eval_timeout": 2400,  # 40 min (in seconds) - subsample evals
    "valset_eval_timeout": 14400,  # 4 hours (in seconds) - reranker architectures need more time for full valset
    # Using default CodeFrequencyComponentSelector (initial=1, decay_rate=25)
    # This does 1:1 code/prompt ratio initially, then increases prompts per code every 25 iterations
}
