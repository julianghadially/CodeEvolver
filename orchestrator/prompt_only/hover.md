## Hover
Hover is designed to retrieve information from 2017 Wikipedia Abstracts (5.9M) using multi-hop retrieval.

#### Task Type: Document Retrieval
This is a document retrieval task. The metric measures whether the correct supporting documents are retrieved — it does NOT measure claim verification or fact-checking accuracy. Do not add claim verification, fact-checking, or classification modules. Focus on improving the primary metric - i.e., retrieval recall and precision.

#### Prompt-only optimization
The program's prompts live in `langProBe/hover/prompts/hover_signatures.py`: one DSPy Signature class per LM call (summarize hop 1, create the hop 2 query, summarize hop 2, create the hop 3 query). Each class docstring is the instruction the LM sees, and each field's `desc` is the description shown next to that field. Improve the program by improving these prompts.

#### Constraints:
- You are only allowed to edit prompts (Including signature docstrings, field descriptions, etc.). Do not under any circumstances edit code outside the prompts Or your changes will be rejected as cheating.
- This constraint applies to the program only. Updating your memory (memory.md) and `codeevolver.md` is allowed and expected.
- The final output of the system CANNOT return more than 21 documents. This is a hard requirement.
- Do NOT use any external websearch services.
- Do not change the language model used for the task. It must remain DeepSeek v4 Flash
- `langProBe/lm_provider.py`: Do not edit, delete, replace or bypass it: it owns the pinned model, the provider routing table, `reasoning_effort`, and the cross-provider fallback.

### Available Services

The following external services are available with API keys already configured in the environment:

### Wikipedia colbert-server (Via dspy.Retrieve)
- Wikipedia colbert-server (Via dspy.Retrieve), already used by the program.
- **Documentation**: https://github.com/julianghadially/colbert-server
- **Use cases**:
  - Retrieve information from Wikipedia abstracts
  - Useful for fact-checking and information retrieval
- **Python usage**: `from dspy import Retrieve`