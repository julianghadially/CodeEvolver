'''
Controls
see additional instructions

Future:
- Consider different search architectures (iterative search, Evidence workspace, Searching specific websites (e.g., Wikipedia))
'''

additional_instructions = """
#### Task Type: Document Retrieval
This is a document retrieval + Question Answering task. The metric measures F1 score - which includes the precision of predicted answers as well as the recall of all available answers. Some questions have multiple correct answers.

To score high, the system requires being able to reason over multiple retrieval steps. Additionally, some questions have multiple correct answers, so the system must be able to reason over multiple chains of logic.

#### What's Allowed
- The program is allowed to create or remove modules, dynamic prompts, tool calls, reasoning steps, etc.
- The program is allowed to change the module types (e.g., dspy.ReAct for tool calling, dspy.RLM for managing large context reasoning, dspy.ChainOfThought, dspy.Predict, etc.)
- There is no limit on the number of search results to display per query or the number of searches to make

#### Constraints:
- Do not change the retriever

#### Ideas
- Try adding reasoning steps and/or structured thinking and/or logic guidance before providing an answer.
- Try formal reasoning / gap analyses / entity mapping in between iterative search steps.
- Try creating a secondary workspace to jot down persistent reasoning logic that agents can add to or remove as they interact with more documents
- RLMs are really good at long-context reasoning and needle-in-the-haystack problems. 
- Try iterative search where retrieval settings are maxed out -  retrieval settings  (high page retrieval value and high query count) and processing it with dspy.RLM.
- Try iterative search methods.
- Try increasing the maximum number of retrieval steps
- Try modifying the number of documents returned per query.

### Available Services
The following external services are available with API keys already configured in the environment:
- PhantomWiki colbert-server (Via dspy.Retrieve)

### Additional notes
- Many answers are surfaced over multiple steps of reasoning and retrieving (i.e., Pure parallel querying is unlikely to work without multiple retrieval steps).
- Different questions require different numbers of retrieval steps.


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
