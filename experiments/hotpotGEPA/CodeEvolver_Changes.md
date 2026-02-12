# CodeEvolver Changes
https://github.com/julianghadially/LangProBe-CodeEvolver/compare/hotpotGEPA...julianghadially:LangProBe-CodeEvolver:codeevolver-20260212001600-4c7d53

## Summary of Changes
1. Removed both summary dspy.predict modules, opting instead for raw evidence
2. Changed primary answering module directive to extract answer (select from passage) instead of generate answer
3. Changed from dspy.predict to dspy.ChainofThought
4. Changed prompts

## Code changes attempted (13)
1. Wasted 3 of 13 code mutations on a broken services module
2. Attempted reranking, without increasing retrieve document size, "in order to reduce context noise" (I.E., no impact)
3. Attempted reranking based on score
4. Attempted a two stage pipeline of extracting answers, and then generating answers
5. Attempted a two stage answer-refinement workflow, that attempted to refine the answers over multiple calls
6. Attempt an information gap analysis between the first and second retrieval, to inform the second retrieval model call
7. Attempted a question decomposition step before the first retrieval call
8. Twice, attempted two variations of answer normalizer modules to address "the core issue where semantically correct answers fail exact match due to missing middle names or incorrect specificity level"

- GenerateAnswer module was changed to ExtractAnswer module, Involving:
- Updated prompts in tandem