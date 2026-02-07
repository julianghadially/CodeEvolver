# CodeEvolver Changes
https://github.com/julianghadially/FactChecker/compare/simple...codeevolver-20260206200043-722583

## Added an Evidence Retriever
```python
"""Evidence retriever module for gathering web evidence via search and scraping."""

import dspy
from src.services.serper_service import SerperService
from src.services.firecrawl_service import FirecrawlService


class EvidenceRetrieverModule(dspy.Module):
    """Module that retrieves web evidence by searching and scraping content.
    Takes search queries as input and:
    1. Searches the web using SerperService (5 results per query)
    2. Scrapes the top 3-5 results using FirecrawlService
    3. Collects markdown content from successful scrapes
    4. Returns combined evidence with source attribution
    This is the second stage of the evidence-aware fact-checking pipeline.
    """

    def __init__(self, max_results_per_query: int = 3, max_evidence_length: int = 15000):
        """Initialize the evidence retriever module.
        Args:
            max_results_per_query: Maximum number of URLs to scrape per query (default 3).
            max_evidence_length: Maximum total characters of evidence to return (default 15000).
        """
        super().__init__()
        self.serper = SerperService()
        self.firecrawl = FirecrawlService()
        self.max_results_per_query = max_results_per_query
        self.max_evidence_length = max_evidence_length

    def forward(self, queries: list[str]) -> dspy.Prediction:
        """Retrieve evidence from the web for given search queries.
        Args:
            queries: List of search query strings.
        Returns:
            dspy.Prediction with:
                - evidence: Combined markdown content from all scraped pages (with source attribution)
                - sources: List of dicts with {url, title, success} for each attempted scrape
        """
        all_evidence = []
        all_sources = []

        for query in queries:
            try:
                # Search for results
                results = self.serper.search(query, num_results=5)

                # Scrape top N results
                for result in results[:self.max_results_per_query]:
                    try:
                        scraped = self.firecrawl.scrape(result.link)

                        if scraped.success and scraped.markdown:
                            # Add evidence with clear source attribution
                            evidence_chunk = f"## Source: {result.title}\nURL: {result.link}\n\n{scraped.markdown}\n\n---\n\n"
                            all_evidence.append(evidence_chunk)
                            all_sources.append({
                                "url": result.link,
                                "title": result.title,
                                "success": True
                            })
                        else:
                            # Track failed scrapes
                            all_sources.append({
                                "url": result.link,
                                "title": result.title,
                                "success": False
                            })
                    except Exception as scrape_error:
                        # Handle individual scrape failures gracefully
                        print(f"Failed to scrape {result.link}: {scrape_error}")
                        all_sources.append({
                            "url": result.link,
                            "title": result.title,
                            "success": False
                        })

            except Exception as search_error:
                # Handle search failures gracefully
                print(f"Failed to search for '{query}': {search_error}")
                continue

        # Combine all evidence and truncate if needed
        combined_evidence = "".join(all_evidence)

        if len(combined_evidence) > self.max_evidence_length:
            combined_evidence = combined_evidence[:self.max_evidence_length] + "\n\n[Evidence truncated due to length...]"

        # If no evidence was gathered, provide informative message
        if not combined_evidence.strip():
            combined_evidence = "No evidence could be retrieved from web sources."

        return dspy.Prediction(
            evidence=combined_evidence,
            sources=all_sources,
        )
```

## Added a Search Query Generator
```python
"""Search query generator module for creating targeted web search queries."""

import dspy
from src.factchecker.signatures.search_query_generator import SearchQueryGenerator


class SearchQueryGeneratorModule(dspy.Module):
    """Module that generates targeted search queries for fact-checking.
    Takes a statement as input and uses an LLM to generate 1-3 specific,
    diverse search queries that will help gather evidence to verify or refute
    the claims in the statement.
    This is the first stage of the evidence-aware fact-checking pipeline.
    """

    def __init__(self):
        """Initialize the search query generator module."""
        super().__init__()
        self.generator = dspy.ChainOfThought(SearchQueryGenerator)

    def forward(self, statement: str) -> dspy.Prediction:
        """Generate search queries for a statement.
        Args:
            statement: The statement to generate queries for.
        Returns:
            dspy.Prediction with:
                - queries: List of 1-3 search query strings
                - reasoning: Explanation of the query strategy
        """
        result = self.generator(statement=statement)

        return dspy.Prediction(
            queries=result.queries,
            reasoning=result.reasoning,
        )
```

## Changed Judge to an Evidence Aware Judge, with confidence and reasoning
**Signature:**
```python

"""Evidence-aware judge signature for evaluating statements with web evidence."""

from dspy import Signature, InputField, OutputField
from typing import Literal

class EvidenceAwareJudge(Signature):
    """Evaluate a statement's factual correctness using gathered web evidence.

    Assess whether the statement is factually accurate by analyzing the provided
    evidence gathered from authoritative web sources. Compare the claims in the
    statement against the evidence to determine if they are supported, refuted,
    or cannot be verified.

    Output one of three verdicts:
    - SUPPORTED: The statement is factually correct according to the evidence
    - CONTAINS_REFUTED_CLAIMS: The statement contains information contradicted by evidence
    - CONTAINS_UNSUPPORTED_CLAIMS: Evidence is insufficient to verify the claims

    Your reasoning should:
    - Cite specific evidence that supports or contradicts the statement
    - Reference source URLs when making claims about what evidence shows
    - Explain any discrepancies or contradictions found
    - Acknowledge when evidence is insufficient or ambiguous
    """

    statement: str = InputField(desc="The statement to evaluate for factual correctness")
    evidence: str = InputField(desc="Markdown content from web sources with source attribution")

    reasoning: str = OutputField(desc="Explanation citing specific evidence and sources that led to this verdict")
    verdict: Literal["SUPPORTED", "CONTAINS_UNSUPPORTED_CLAIMS", "CONTAINS_REFUTED_CLAIMS"] = OutputField(
        desc="The factual correctness verdict based on evidence"
    )
    confidence: float = OutputField(desc="Confidence score between 0.0 and 1.0")


"""Evidence-aware judge module - fact checker with web search and evidence retrieval."""

import dspy
from src.factchecker.signatures.evidence_aware_judge import EvidenceAwareJudge
from src.factchecker.modules.search_query_generator_module import SearchQueryGeneratorModule
from src.factchecker.modules.evidence_retriever_module import EvidenceRetrieverModule
```
**Module:**
``` python

class JudgeModule(dspy.Module):
    """Evidence-aware fact checker with web search and evidence retrieval pipeline.

    This module implements a multi-stage fact-checking pipeline:
    1. SearchQueryGenerator: Generates 1-3 targeted search queries for the statement
    2. EvidenceRetriever: Searches the web and scrapes content to gather evidence
    3. EvidenceAwareJudge: Evaluates the statement using the gathered evidence

    This allows the system to verify recent events and specific claims beyond
    the LLM's knowledge cutoff by consulting authoritative web sources.
    """

    def __init__(self):
        """Initialize the evidence-aware judge module pipeline."""
        super().__init__()
        self.query_generator = SearchQueryGeneratorModule()
        self.evidence_retriever = EvidenceRetrieverModule()
        self.judge = dspy.ChainOfThought(EvidenceAwareJudge)

    def forward(self, statement: str) -> dspy.Prediction:
        """Evaluate a statement for factual correctness using web evidence.

        Args:
            statement: The statement to evaluate.

        Returns:
            dspy.Prediction with:
                - statement: The input statement
                - overall_verdict: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
                - confidence: Float between 0.0 and 1.0
                - reasoning: Explanation of the verdict citing evidence
                - queries: List of search queries used (for transparency)
                - sources: List of source URLs consulted (for transparency)
        """
        # Stage 1: Generate search queries
        query_result = self.query_generator(statement=statement)

        # Stage 2: Retrieve evidence from web sources
        evidence_result = self.evidence_retriever(queries=query_result.queries)

        # Stage 3: Judge the statement using evidence
        judgment = self.judge(statement=statement, evidence=evidence_result.evidence)

        # Return prediction with original format plus transparency fields
        return dspy.Prediction(
            statement=statement,
            overall_verdict=judgment.verdict,
            confidence=judgment.confidence,
            reasoning=judgment.reasoning,
            # Additional fields for transparency and debugging
            queries=query_result.queries,
            sources=evidence_result.sources,
        )
```