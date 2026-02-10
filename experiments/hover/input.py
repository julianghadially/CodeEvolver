additional_instructions = """
## Hover
Hover is designed to retrieve information from 2017 Wikipedia Abstracts (5.9M). 

#### What's Allowed
- The program is not required to stay on Wikipedia only.
- The program is allowed to create or remove modules, dynamic prompts, tool calls, etc.
- The program is allowed to change the module types (e.g., dspy.ReAct for tool calling, dspy.ChainOfThought, dspy.Predict, etc.)
- The program is allowed to add rerankers, provided the final remains the same - 21 total documents
- There is no limit on the number of search results to display per query
- Available services: wikipedia colbert-server (Via dspy.Retrieve), Firecrawl, serper.dev. 

#### Constraints:
- Do NOT search more than three times per question. This is a hard requirement.
- Do NOT return more than 21 documents. This is a hard requirement.
- Do NOT use the hover dataset as context. 



### Available Services

The following external services are available with API keys already configured in the environment:

### Wikipedia colbert-server (Via dspy.Retrieve)
- **Documentation**: https://github.com/julianghadially/colbert-server
- **Use cases**:
  - Retrieve information from Wikipedia abstracts
  - Useful for fact-checking and information retrieval
- **Python usage**: `from dspy import Retrieve`

## Ideas for Optimization
- Consider different context retrieval pipelines, including query + rerankers (pairwise rerankers, list rerankers, score-based reranking, sliding-window rerankers, etc.)
- Text reranker paper here on pairwise: https://arxiv.org/abs/2306.17563
- Utility-oriented rerankers: https://arxiv.org/abs/2110.09059
- LLM enhanced rerankers: https://arxiv.org/html/2406.12433v2
- FIRST Faster Improved Listwise Reranking with Single Token Decoding rerankers: https://arxiv.org/abs/2406.15657
- Consider different search architectures (query-only; query + rerank per query; query + rerank after all queries; increase or decrease k, etc.)
"""

OPTIMIZE_CONFIG = {
    "repo_url": "https://github.com/julianghadially/LangProBe-CodeEvolver",
    "program": "langProBe.hover.hover_program.HoverMultiHopPredict",
    "metric": "langProBe.hover.hover_utils.discrete_retrieval_eval",
    "trainset_path": "data/hoverBench_train.json", # data/FacTool_QA_train_normalized.jsonl
    "valset_path": "data/hoverBench_val.json", # data/FacTool_QA_train_normalized.jsonl
    "input_keys": ["claim"],
    "reflection_lm": "openai/gpt-4-mini",
    "max_metric_calls": 1000,
    "num_threads": 5,
    "seed": 42,
    "additional_instructions": additional_instructions,
    "initial_branch": "hover",  # Start from the 'simple' branch
    # Using default round_robin selector (no initial specified)
    # This lets GEPA's ReflectionComponentSelector handle component selection
}
