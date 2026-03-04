'''
Controls
- search count
- Limited to existing Wikipedia web service
- reasoning variable controlled by allowing chain of thought reasoning

Not controlled
- module creation, removal, and modification
- # of database documents retrieved per search (same cost)

Future:
- Consider different search architectures (iterative search, Evidence workspace, Searching specific websites (e.g., Wikipedia))
'''

additional_instructions = """
HotpotQA is designed to contain information from Wikipedia. 
#### What's Allowed
- The program is allowed to create or remove modules
- The program is allowed to create or remove dynamic prompts
- The program is allowed to add rerankers
- There is no limit on the number of search results to retrieve per query (same cost)
- Available services: wikipedia colbert-server (Via dspy.Retrieve)

#### Constraints:
- Do NOT search more than two times per question. This is a hard requirement.
- Do NOT use any external websearch services.
- Do NOT use the HotpotQA dataset as context. 
- The program is limited to Wikipedia documents.

### Available Services

The following external services are available with API keys already configured in the environment:

### Wikipedia colbert-server (Via dspy.Retrieve)
- **Documentation**: https://github.com/julianghadially/colbert-server
- **Use cases**:
  - Retrieve information from Wikipedia abstracts
  - Useful for fact-checking and information retrieval
- **Python usage**: `from dspy import Retrieve`

## Ideas for Optimization
- Consider increasing the number of k retrieved
- Consider different context retrieval pipelines, including query + rerankers (list rerankers, score-based reranking, sliding-window rerankers, etc.)
- Consider removing question summaries, and providing raw evidence instead (This performed well in a prior GEPA run)
- Consider replacing Generate Answer with Extract Answer - i.e., extracting the exact short factoid answer from the passages (This performed well in a prior GEPA run)
"""

additional_instructions_resource_tradeoffs = """
HotpotQA is designed to contain information from Wikipedia. The goal of this program is to maximize the accuracy of question-answering while managing resource use (queries) appropriately.

#### What's Allowed
- The program is allowed to create or remove modules
- The program is allowed to create or remove dynamic prompts
- The program is allowed to add rerankers
- The program is allowed to add additional retrieval queries.
- There is no limit on the number of search results to retrieve per query (although the system tends to perform worse after more than 30 results per query)
- Available services: wikipedia colbert-server (Via dspy.Retrieve)

#### Constraints:
- Do NOT use any external websearch services.
- Do NOT use the HotpotQA dataset as context. 
- The program is limited to Wikipedia documents.

### Available Services

The following external services are available with API keys already configured in the environment:

### Wikipedia colbert-server (Via dspy.Retrieve)
- **Documentation**: https://github.com/julianghadially/colbert-server
- **Use cases**:
  - Retrieve information from Wikipedia abstracts
  - Useful for fact-checking and information retrieval
- **Python usage**: `from dspy import Retrieve`

## Ideas for Optimization
- Consider increasing the number of k retrieved per query
- It will likely be beneficial to increase the number of retrieval queries above two queries per question. This comes with a Resource Use Trade-Off (see below). 
- Consider performing iterative retrieval, where if there is not enough information, the system can perform as many additional queries as it wants.
- Consider the correct number of documents to pass to a module. Passing raw documents helps avoid any loss of information, while passing too many documents can exceed the maximum context window. Context rot is also experienced as the context window starts to approach being full...
- Consider different context retrieval pipelines, including query + rerankers (list rerankers, score-based reranking, sliding-window rerankers, etc.)
- Consider removing question summaries, and providing raw evidence instead (This performed well in a prior GEPA run)
- Consider replacing Generate Answer with Extract Answer - i.e., extracting the exact short factoid answer from the passages (This performed well in a prior GEPA run)

## Resource use
The goal of this program is to maximize the accuracy of question-answering while managing resource use appropriately. Each additional query (beyond 2) comes with a cost. This cost is roughly equivalent to a 2% improvement in accuracy. In other words, an additional query should increase the accuracy by more than 2% on average in order to be net positive. This will be reflected in the score for the program. 

"""


OPTIMIZE_CONFIG = {
    "repo_url": "https://github.com/julianghadially/LangProBe-CodeEvolver",
    "program": "langProPlus.hotpotGEPA.hotpot_pipeline.HotpotMultiHopPipeline",
    "metric": "langProPlus.hotpotGEPA.hotpot_metric_resource.hotpot_accuracy_with_resource_penalty_feedback", #langProPlus.hotpotGEPA.__init__.exact_match_metric
    "trainset_path": "data/HotpotQABench_train.json", # data/FacTool_QA_train_normalized.jsonl
    "valset_path": "data/HotpotQABench_val.json", # data/FacTool_QA_train_normalized.jsonl
    "input_keys": ["question"],
    "reflection_lm": "openai/gpt-4.1-mini",
    "max_metric_calls": 7500, # 150 examples × 40 full evals (with subsampled valset)
    "num_threads": 8,
    "max_valset_size": 150, # Subsample validation set to 150 examples (from 300) for faster evaluation
    "seed": 42,
    "additional_instructions": additional_instructions_resource_tradeoffs,
    "initial_branch": "hotpotGEPA",
    "subsample_size": 20,
    # Using default CodeFrequencyComponentSelector (initial=1, decay_rate=25)
    # This does 1:1 code/prompt ratio initially, then increases prompts per code every 25 iterations
}
