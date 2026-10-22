# Prompt-lesson catalog: ragqa

43 lessons in 6 themes. By label: Task 1, Domain 2, Tool 2, Procedure 6, Decision 14, Output 18.

## Lessons

### T1: Search query formulation

#### L1: Keyword-style queries
Label: Tool
Description: A search query should be a short keyword phrase of the most distinctive searchable terms, like a web search, not a full natural-language sentence or question.

#### L2: Retrieval corpus matches technical wording
Label: Tool
Description: The retrieval corpus is Q&A-forum style and matches on technical wording, so a single phrasing of the question often misses the most relevant passages.

#### L3: Multiple diverse queries from different angles
Label: Procedure
Description: When writing several or additional queries, make them differ from one another: rephrase the concept, use alternative or more specific or technical terminology, or target a key sub-aspect. Do not repeat the first query, because this retrieves passages the first would miss.

#### L4: Name the likely interpretation of ambiguous questions
Label: Decision
Description: If the question is ambiguous, colloquial or idiomatic, include a query that states its most likely concrete interpretation.

#### L5: Follow-up query targets missing information
Label: Procedure
Description: When context has already been retrieved, the next query should seek the most important information it does not yet cover, rather than rephrasing or repeating what was already found.

#### L6: Follow-up may use a complementary aspect or alternative method
Label: Decision
Description: To broaden coverage, the follow-up query can pick a complementary aspect of the task or a different method, tool or approach to the same goal.

#### L7: Follow-up stays on topic and keeps the intent
Label: Decision
Description: A follow-up query must stay on the original question's topic and intent. Do not write queries about niche edge cases or tangential sub-questions.

#### L8: Stop searching with an empty query when context suffices
Label: Decision
Description: If the gathered context already suffices to answer the question, emit an empty query instead of searching further.

### T2: Role of retrieved context versus own knowledge

#### L9: Context is main evidence, own knowledge supplements
Label: Task
Description: Treat retrieved context as the primary evidence, and add the model's own knowledge where the context is thin or incomplete. The answer is not limited to or dependent on the context.

#### L10: Irrelevant or silent context falls back to own knowledge
Label: Decision
Description: If the passages do not address the question or are only tangentially related, answer from well-established knowledge. Do not force an answer out of unrelated context.

#### L11: Defer to context over prior expectation
Label: Decision
Description: When the context directly addresses the question, follow it even if it conflicts with what the model would expect.

#### L12: Trust context for version- or platform-specific details
Label: Decision
Description: For version-specific or platform-specific technical details, prefer what the context states over the model's own, possibly outdated, recall. Incorrect specifics cost more than missing ones.

#### L13: Never refuse or declare context insufficient
Label: Output
Description: Always give a complete best-effort answer. Do not refuse, and do not describe what the context lacks.

#### L14: Synthesize across passages
Label: Procedure
Description: Combine the relevant passages rather than fixating on the framing of the single most specific one, without introducing claims the evidence does not state.

#### L15: Base the answer on the provided passages
Label: Output
Description: Ground the answer in the provided passages and avoid speculation beyond them.

#### L16: Consider all passages including limiting or contradicting ones
Label: Procedure
Description: Take all retrieved passages into account, including those that limit or contradict the main point, instead of using only the supporting ones.

### T3: Faithfulness and hedging of claims

#### L17: No fabricated specifics
Label: Output
Description: Do not invent sources, citations, URLs, exact numbers, version availability, product or feature names that cannot be verified.

#### L18: Only supported claims, no speculation
Label: Output
Description: State only what the evidence or well-established fact supports. Avoid speculative, unverified, dubious or low-confidence claims, and do not state claims more strongly than the evidence does.

#### L19: Assert established facts confidently
Label: Output
Description: State well-established facts confidently even if the context omits them. Reserve hedging for claims that go beyond both the context and established fact.

#### L20: Include standard commands and tools
Label: Output
Description: Always include the well-known standard commands and tools relevant to the question, stated confidently.

#### L21: Do not harden hedged evidence into absolutes
Label: Output
Description: When evidence is qualified, conditional, mixed or uncertain, keep that qualification. Avoid unsupported absolutes such as 'always' or 'never' and do not present hedged claims as confirmed.

#### L22: Include caveats and conditions the answer depends on
Label: Output
Description: Mention the important caveats, limitations and conditions from the evidence, such as when the answer depends on a specific factor, and be balanced on tradeoffs or debated topics.

#### L23: Scope single-case or anecdotal evidence
Label: Output
Description: Present a detail or suggestion that rests on one case, a single user's anecdote or a hedged source as applying to that case, or as an option to consider, not as a general rule or guaranteed approach.

#### L24: Do not invent properties for merely named items
Label: Decision
Description: If a passage only names a product or rule without describing it, do not attribute properties or compatibility to it that the passage does not state.

#### L25: Keep distinct figures separate
Label: Output
Description: When the evidence gives several distinct figures or estimates, state the main ones directly and do not merge them into one rounded number.

#### L26: Keep distinct causes separate
Label: Output
Description: When the evidence attributes the same observation to different underlying conditions, state each condition separately rather than merging them.

### T4: Versions, platforms and applicability

#### L27: Prefer the most current version information
Label: Domain
Description: When the question or context involves versions, dates, or features that were removed or changed, prefer the most current information and state it clearly.

#### L28: Do not conflate platforms, versions or eras
Label: Domain
Description: If a passage concerns a different platform, product, version or era than the question's subject, note the distinction. Do not carry over its commands, numbers, specs or caveats to the subject.

#### L29: Unspecified platform: lead with the general method
Label: Decision
Description: If the question names no platform or device but the context is platform-specific, lead with the general platform-agnostic method and treat the platform-specific approach as a secondary supplement.

#### L30: Unspecified platform: mention likely alternatives
Label: Decision
Description: If the question names no platform but the context is platform-specific, briefly mention the most likely alternatives for other platforms from the model's own knowledge.

#### L31: Frame the question's scope first
Label: Procedure
Description: Before answering, briefly establish what the question asks and whether it is specific to a platform, tool or situation or more general.

### T5: Answer content, scope and depth

#### L32: Lead with the direct standard solution
Label: Decision
Description: Begin with the most relevant, direct, standard answer, and only afterwards add alternatives, edge cases or platform detail. Prefer clear direct solutions over lengthy hacks or workarounds unless those are genuinely best.

#### L33: Stay focused, no padding or tangents
Label: Output
Description: Answer the specific question, cover its key points and stop. State each point once, and do not add unrelated topics, peripheral specs, exhaustive enumerations, or caveats the question did not raise.

#### L34: Match depth to question scope
Label: Output
Description: Give a direct, concise answer to simple factual or yes/no questions. Give a thorough, broader answer to multi-part, nuanced, open-ended or how-to questions.

#### L35: Cover multiple valid approaches
Label: Output
Description: When several valid approaches, options or perspectives exist, include them rather than only the first or most prominent one.

#### L36: Explain rather than give a bare answer
Label: Output
Description: Explain the answer instead of giving only a bare command or a single phrase.

#### L37: Opinion or architecture questions: give trade-offs
Label: Decision
Description: For opinion-based, architectural or 'is it a good idea' questions, present the trade-offs and lead with the key condition that decides the answer. Do not give a bare yes/no unless the evidence clearly favors one side.

#### L38: Answer in English
Label: Output
Description: Write the answer in English.

### T6: Answer presentation and draft revision

#### L39: No citation markers or passage numbers
Label: Output
Description: Do not include bracketed reference numbers, passage numbers or source citations such as [1] in the answer. Present clean readable prose.

#### L40: Do not refer to the context or its sources
Label: Output
Description: Address the reader directly and present the information as self-contained. Do not mention 'the context', 'the passages' or paragraphs, or where a fact came from.

#### L41: Verify each claim against the passages
Label: Procedure
Description: In revision, check each factual claim in the draft against the passages, record whether a passage directly supports it, and remove claims that are not directly supported.

#### L42: Revision only removes, never adds
Label: Decision
Description: When revising a draft, do not add new information or claims that are not in the passages.

#### L43: Keep supported or uncertain claims during revision
Label: Decision
Description: Keep claims that are supported, and when unsure whether a claim is supported, keep it rather than removing it. The revised answer should stay clear, organized and concise.

## Notes from discovery (ignored by the parser)

- Some lessons pull in opposite directions and are kept separate on purpose. 'Include caveats and conditions' and 'Stay focused, no padding' differ in scope: the first covers caveats from the evidence that the answer depends on, the second covers caveats the question did not raise. 'Base the answer on the provided passages' is stricter than 'Context is main evidence, own knowledge supplements' and 'Only supported claims'.
- The two follow-up query lessons, 'Follow-up query targets missing information' and 'Multiple diverse queries from different angles', are kept apart. The first is about seeking what earlier retrieval did not cover. The second is about making parallel or alternative queries differ from each other.
- 'Answer in English' is borderline as pure style. It is kept because the language of the answer plausibly affects scoring.
- 'Never refuse or declare context insufficient' and 'Irrelevant or silent context falls back to own knowledge' are closely related. The first is the output prohibition, and the second is the decision rule about what to use instead.
- 'Include standard commands and tools' is kept apart from 'Assert established facts confidently'. It is a rule about what to include, not about how firmly to state it.
- The two 'Unspecified platform' lessons are opposite-adjacent but different decisions: what to lead with, and whether to add alternatives.
- (validator) possible provenance wording 'earlier' in: The two follow-up query lessons, 'Follow-up query targets missing information' and 'Multiple diverse queries from differ
