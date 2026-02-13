additional_instructions = """
HotpotQA is designed to contain information from Wikipedia. 
#### What's Allowed
- The program is not required to stay on Wikipedia only.
- The program is allowed to create or remove modules, dynamic prompts, tool calls, etc.
- The program is allowed to change the module types (e.g., dspy.ReAct for tool calling, dspy.ChainOfThought, dspy.Predict, etc.)
- There is no limit on the number of search results to display per query
- Available services: Firecrawl and serper.dev. 

#### Constraints:
- Do NOT search more than two times per question. This is a hard requirement.
- Do NOT visit more than one page per query
- Do NOT use the HotpotQA dataset as context. 

### Available Services

The following external services are available with API keys already configured in the environment:

### Wikipedia colbert-server (Via dspy.Retrieve)
- **Documentation**: https://github.com/julianghadially/colbert-server
- **Use cases**:
  - Retrieve information from Wikipedia abstracts
  - Useful for fact-checking and information retrieval
- **Python usage**: `from dspy import Retrieve`
### Firecrawl (Web Scraping & Page Fetching)
- **API Key**: Available as `FIRECRAWL_API_KEY` environment variable
- **Documentation**: https://docs.firecrawl.dev/
- **Use cases**:
  - Fetch and parse web page content
  - Convert web pages to clean markdown
  - Crawl websites for structured data extraction
- **Python usage**: `from firecrawl import FirecrawlApp`

### Serper.dev (Web Search)
- **API Key**: Available as `SERPER_API_KEY` environment variable
- **Documentation**: https://serper.dev/
- **Use cases**:
  - Search the web for real-time information
  - Get search results with snippets, URLs, and metadata
  - Useful for fact-checking and information retrieval
- **Python usage**: HTTP requests to `https://google.serper.dev/search`

## Ideas for Optimization
- Consider adding web search capability to verify claims against current information
- Web scraping could help retrieve source documents for fact verification
- These services can augment the context pipeline with real-time data
- Consider different search architectures (iterative search, Evidence workspace, Searching specific websites (e.g., Wikipedia))
- Consider different search architectures (query-only; query + rerank per query; query + rerank after all queries; increase or decrease k, etc.)
- Consider different context retrieval pipelines, including query + rerankers (pairwise rerankers, list rerankers, score-based reranking, sliding-window rerankers, etc.)
- Text reranker paper here on pairwise: https://arxiv.org/abs/2306.17563
- Utility-oriented rerankers: https://arxiv.org/abs/2110.09059
- LLM enhanced rerankers: https://arxiv.org/html/2406.12433v2
- FIRST Faster Improved Listwise Reranking with Single Token Decoding rerankers: https://arxiv.org/abs/2406.15657
"""

OPTIMIZE_CONFIG = {
    "repo_url": "https://github.com/julianghadially/LangProBe-CodeEvolver",
    "program": "langProPlus.hotpotGEPA.hotpot_pipeline.HotpotMultiHopPredictPipeline",
    "metric": "langProPlus.hotpotGEPA.__init__.exact_match_metric",
    "trainset_path": "data/HotpotQABench_train.json", # data/FacTool_QA_train_normalized.jsonl
    "valset_path": "data/HotpotQABench_val.json", # data/FacTool_QA_train_normalized.jsonl
    "input_keys": ["question"],
    "reflection_lm": "openai/gpt-4.1-mini",
    "max_metric_calls": 6000, # 150 examples × 40 full evals (with subsampled valset)
    "num_threads": 20,
    "max_valset_size": 150, # Subsample validation set to 150 examples (from 300) for faster evaluation
    "seed": 42,
    "additional_instructions": additional_instructions,
    "initial_branch": "hotpotGEPA"
    # Using default CodeFrequencyComponentSelector (initial=1, decay_rate=25)
    # This does 1:1 code/prompt ratio initially, then increases prompts per code every 25 iterations
}
