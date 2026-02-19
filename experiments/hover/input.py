'''
Controls
- search count this is hard
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
Hover is designed to retrieve information from 2017 Wikipedia Abstracts (5.9M).

#### Task Type: Document Retrieval
This is a document retrieval task. The metric measures whether the correct supporting documents are retrieved — it does NOT measure claim verification or fact-checking accuracy. Do not add claim verification, fact-checking, or classification modules. Focus on improving retrieval recall and precision through better queries, re-ranking, or retrieval strategies.

#### What's Allowed
- The program is allowed to create or remove modules
- The program is allowed to create or remove dynamic prompts
- The program is allowed to add rerankers, provided the final document count remains the same - 21 total documents
- There is no limit on the number of search results to retrieve per query (same cost)
- Available services: wikipedia colbert-server (Via dspy.Retrieve)

#### Constraints:
- Do NOT search more than three times per question. This is a hard requirement.
- Do NOT return more than 21 documents in the final output. This is a hard requirement.
- Do NOT use the hover dataset as context. 
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
- Consider increasing the k retrieved, and then reranking the final results to 21 (max 21 documents allowed)
- Consider different context retrieval pipelines, including query + different kinds of rerankers (list rerankers, score-based reranking, sliding-window rerankers, etc.)
- Utility-oriented rerankers: https://arxiv.org/abs/2110.09059
- LLM enhanced rerankers: https://arxiv.org/html/2406.12433v2
- FIRST Faster Improved Listwise Reranking with Single Token Decoding rerankers: https://arxiv.org/abs/2406.15657
- Do not attempt pairwise re-ranking as this takes too much time.
- Consider a gap analysis before generating queries. This worked in a past GEPA optimization run.
"""

OPTIMIZE_CONFIG = {
    "repo_url": "https://github.com/julianghadially/LangProBe-CodeEvolver",
    "program": "langProBe.hover.hover_pipeline.HoverMultiHopPipeline",
    "metric": "langProBe.hover.hover_utils.discrete_retrieval_eval",
    "trainset_path": "data/hoverBench_train.json", # data/FacTool_QA_train_normalized.jsonl
    "valset_path": "data/hoverBench_val.json", # data/FacTool_QA_train_normalized.jsonl
    "input_keys": ["claim"],
    "reflection_lm": "openai/gpt-4.1-mini",
    "max_metric_calls": 7500,  # 150 examples × 50 full evals (with subsampled valset)
    "num_threads": 20,  # Increased from 5 to 20 for better parallelization
    "max_valset_size": 150,  # Subsample validation set to 150 examples (from 300) for faster evaluation
    "seed": 42,
    "additional_instructions": additional_instructions,
    "initial_branch": "hover",
    "decay_rate": 30,
    "subsample_size": 20,
    # Using default CodeFrequencyComponentSelector (initial=1, decay_rate=25)
    # This does 1:1 code/prompt ratio initially, then increases prompts per code every 25 iterations
}
