additional_instructions = """## Available Services

The following external services are available with API keys already configured in the environment:

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
"""

OPTIMIZE_CONFIG = {
    "repo_url": "https://github.com/julianghadially/FactChecker",
    "program": "src.factchecker.modules.judge_module.JudgeModule",
    "metric": "src.codeevolver.metric.metric",
    "trainset_path": "data/FactChecker_news_claims_normalized.csv", # data/FacTool_QA_train_normalized.jsonl
    "input_keys": ["statement"],
    "reflection_lm": "openai/gpt-5-mini",
    "max_metric_calls": 1000,
    "num_threads": 5,
    "seed": 42,
    "additional_instructions": additional_instructions,
    "initial_branch": "simple",  # Start from the 'simple' branch
    # Using default round_robin selector (no initial specified)
    # This lets GEPA's ReflectionComponentSelector handle component selection
}
