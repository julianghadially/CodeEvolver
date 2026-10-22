#### Task Type: Document Retrieval
This is a document retrieval + Question Answering task. The metric measures F1 score - which includes the precision of predicted answers as well as the recall of all available answers. Some questions have multiple correct answers.

To score high, the system must be able to reason over multiple retrieval steps. Additionally, many questions have multiple correct answers, so the system must also be able to reason over multiple chains of logic simultaneously.

The core difficulty of Phantom Wiki is the large amount of information that must be processed across multiple chains of logic. Additionally, if context accumulates across multiple chains of logic and the context is too long, the model can suffer from context rot.

#### Prompt-only optimization
The program's prompts live in `src/program/prompts/phantomwiki_signatures.py`: the DSPy Signature the ReAct agent solves (its class docstring is the task instruction the LM sees, and each field's `desc` is the description shown next to that field) and `SEARCH_WIKI_DESC`, the description of the `search_wiki` tool shown to the agent. Improve the program by improving these prompts.

#### Constraints:
- You are only allowed to edit prompts (Including signature docstrings, field descriptions, etc.). Do not under any circumstances edit code outside the prompts Or your changes will be rejected as cheating.
- This constraint applies to the program only. Updating your memory (memory.md) and `codeevolver.md` is allowed and expected.
- Do not change the language model used. It must remain DeepSeek-V4-Flash.
- Every evaluated row must be answered by the LLM. At least one LM call must contribute to every prediction.
- `src/infra/lm_provider.py`: Do not edit, delete, replace or bypass it: it owns the pinned model, the provider routing table, `reasoning_effort`, and the cross-provider fallback.

### Available Services
The following external services are available with API keys already configured in the environment:
- PhantomWiki colbert-server (Via dspy.Retrieve)

### Additional notes
- Many answers are surfaced over multiple steps of reasoning and retrieving.
- Different questions require different numbers of retrieval steps.
