"""Test data for GEPA state history tracking unit tests.

Represents a realistic 4-candidate optimization trajectory:

  idx 0  (seed)   → Initial candidate, base prompts, main branch
  idx 1  (prompt) → Improved create_query_hop2 instruction
  idx 2  (prompt) → Improved summarize1 instruction
  idx 3  (code)   → Added ExtractKeyFacts module + new git branch
"""

import json

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
MAIN_BRANCH = "codeevolver-20260213100437-main"
CODE_BRANCH = "codeevolver-20260213100437-0c5479"
PROGRAM_PATH = (
    "langProPlus.hotpotGEPA.hotpot_pipeline.HotpotMultiHopPredictPipeline"
)

# ---------------------------------------------------------------------------
# _code JSON payloads (as dicts — candidates store them as JSON strings)
# ---------------------------------------------------------------------------
SEED_CODE = {
    "git_branch": MAIN_BRANCH,
    "parent_module_path": PROGRAM_PATH,
    "change_request": "",
    "last_change_summary": "Initial state",
}

CODE_MUTATION_CODE = {
    "git_branch": CODE_BRANCH,
    "parent_module_path": PROGRAM_PATH,
    "change_request": (
        '{"change_request": "In `langProPlus/hotpotGEPA/hotpot_program.py`, '
        "replace the final answer generation architecture with a two-stage "
        "approach: (1) Add a new `ExtractKeyFacts` signature that takes "
        "question + summary_1 + summary_2 and outputs 3-5 key factoids as a "
        "bulleted list, then (2) Replace the `GenerateAnswer` module with "
        "`dspy.ChainOfThought(GenerateAnswer)` and update the GenerateAnswer "
        "signature's answer field description to: 'A single short factoid "
        "answer with no articles, qualifiers, or extra words'. This "
        "architecture change ensures the LM reasons through extracted facts "
        'before producing a concise answer."}'
    ),
    "last_change_summary": (
        "[TIMER] Timer started - Loading .env\n"
        "[AGENT] Looking for .env at: /workspace/.env\n"
        "[AGENT] Found .env file, loading...\n"
        "[AGENT] Starting code mutation..."
    ),
}

# ---------------------------------------------------------------------------
# Default prompt texts (shared across seed and early candidates)
# ---------------------------------------------------------------------------
DEFAULT_CREATE_QUERY_HOP2 = (
    "Given the fields `question` and `summary_1`, produce the fields `query`."
)

IMPROVED_CREATE_QUERY_HOP2 = (
    "Task Description:\n"
    "Given two input fields: `question` and `summary_1`, your objective is to "
    "generate a concise `query` that effectively extracts or focuses on the "
    "key answer or entity requested by the question, based on the supporting "
    "information provided in `summary_1`.\n\n"
    "Key Details and Guidelines:\n"
    "1. The `query` should be an answer-focused or answer-equivalent phrase, "
    "word, or named entity extracted or synthesized from the combination of "
    "`question` and `summary_1`.\n\n"
    "2. The `query` is not a reformulation or restatement of the question, "
    "but rather the succinct key fact, name, title, or term that answers the "
    "question or settles the question\u2019s focus, as supported by the summary.\n\n"
    "3. The `query` should directly address the expected final answer to the "
    "question rather than restating or paraphrasing the question structure.\n\n"
    "4. The `query` should not include extraneous details, full sentences, or "
    "explanations. Only provide the minimal necessary name, date, label, or "
    "phrase that correctly answers or identifies the main focus of the question."
)

DEFAULT_SUMMARIZE1 = (
    "Given the fields `question`, `passages`, produce the fields `summary`."
)

IMPROVED_SUMMARIZE1 = (
    "You are a summarization assistant. Given a `question` and `passages`, "
    "produce a focused `summary` that extracts only the facts relevant to "
    "answering the question. Omit tangential information. Keep the summary "
    "under 3 sentences and ensure it preserves exact names, dates, and figures."
)

DEFAULT_SUMMARIZE2 = (
    "Given the fields `question`, `context`, `passages`, produce the fields `summary`."
)

DEFAULT_GENERATE_ANSWER = "Answer questions with a short factoid answer."

# ---------------------------------------------------------------------------
# Candidates — each is a dict[str, str] as GEPA stores them
# ---------------------------------------------------------------------------

# idx 0: Seed candidate — base prompts, main branch
CANDIDATE_0_SEED = {
    "program.create_query_hop2": DEFAULT_CREATE_QUERY_HOP2,
    "program.summarize1": DEFAULT_SUMMARIZE1,
    "program.summarize2": DEFAULT_SUMMARIZE2,
    "program.generate_answer": DEFAULT_GENERATE_ANSWER,
    "_code": json.dumps(SEED_CODE),
}

# idx 1: Prompt change — improved create_query_hop2, same branch
CANDIDATE_1_PROMPT_QUERY = {
    "program.create_query_hop2": IMPROVED_CREATE_QUERY_HOP2,
    "program.summarize1": DEFAULT_SUMMARIZE1,
    "program.summarize2": DEFAULT_SUMMARIZE2,
    "program.generate_answer": DEFAULT_GENERATE_ANSWER,
    "_code": json.dumps(SEED_CODE),  # same _code as seed — no code change
}

# idx 2: Prompt change — improved summarize1, same branch
CANDIDATE_2_PROMPT_SUMMARIZE = {
    "program.create_query_hop2": IMPROVED_CREATE_QUERY_HOP2,  # kept from idx 1
    "program.summarize1": IMPROVED_SUMMARIZE1,
    "program.summarize2": DEFAULT_SUMMARIZE2,
    "program.generate_answer": DEFAULT_GENERATE_ANSWER,
    "_code": json.dumps(SEED_CODE),  # still same _code — no code change
}

# idx 3: Code change — added ExtractKeyFacts module, new branch
CANDIDATE_3_CODE_CHANGE = {
    "program.create_query_hop2": IMPROVED_CREATE_QUERY_HOP2,
    "program.summarize1": IMPROVED_SUMMARIZE1,
    "program.summarize2": DEFAULT_SUMMARIZE2,
    "program.generate_answer": DEFAULT_GENERATE_ANSWER,
    "_code": json.dumps(CODE_MUTATION_CODE),  # new branch + change_request
    # New modules added by the code change:
    "program.extract_key_facts": (
        "Extract key facts from summaries to answer the question."
    ),
    "program.generate_answer.predict": (
        "Answer questions with a short factoid answer."
    ),
}

# ---------------------------------------------------------------------------
# Full trajectory arrays (parallel lists matching GEPAStateRecord fields)
# ---------------------------------------------------------------------------
PROGRAM_CANDIDATES = [
    CANDIDATE_0_SEED,
    CANDIDATE_1_PROMPT_QUERY,
    CANDIDATE_2_PROMPT_SUMMARIZE,
    CANDIDATE_3_CODE_CHANGE,
]

CANDIDATE_SCORES = [0.45, 0.48, 0.50, 0.52]

PARENT_PROGRAMS = [
    [None],  # idx 0: seed — no parent
    [0],     # idx 1: parent is seed
    [1],     # idx 2: parent is idx 1
    [2],     # idx 3: parent is idx 2
]
