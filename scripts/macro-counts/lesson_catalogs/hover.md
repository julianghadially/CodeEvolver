# Prompt-lesson catalog: hover

40 lessons in 6 themes. By label: Task 2, Domain 8, Tool 4, Procedure 5, Decision 17, Output 4.

## Lessons

### T1: Task framing and scoring

#### L1: Retrieve supporting articles, do not verify
Label: Task
Description: The job is to retrieve the Wikipedia articles (entity pages, plus connecting and context articles) that would serve as supporting documents for a claim. It is not to decide whether the claim is true, or which entity is 'the answer'.

#### L2: One missing document fails the claim; favor recall
Label: Task
Description: A claim usually needs several supporting articles, and missing a single one fails the whole claim. Keep or select every plausibly relevant candidate, and include marginal ones when unsure, because an over-included article costs less than a missed one.

#### L3: Supporting documents are of diverse types
Label: Domain
Description: Supporting articles are frequently works, fictional characters, people, places, events, concepts, theories, formations or taxa. They are often only mentioned in passing inside another article, so do not filter them out for seeming peripheral or for not being the answer.

#### L4: Every named entity may have its own article
Label: Domain
Description: Every distinct proper noun or named thing in the claim or in retrieved text may have its own supporting article, even a minor one such as a product, song, technique, award or descriptor. List and query all of them rather than only the main subject, and do not dismiss a name as irrelevant before retrieval has had a chance to decide.

### T2: Query formulation

#### L5: Short keyword-phrase queries that look like article titles
Label: Tool
Description: The retriever works best on short, entity-focused noun phrases (about 2-7 words) that look like Wikipedia article titles. Do not use questions, full sentences, verbose natural language or the raw claim text as queries.

#### L6: No generic descriptor words in queries
Label: Decision
Description: Do not add generic descriptor words such as 'population', 'common name', 'born in' or 'author of' to a query. They are not part of article titles and do not help target an article.

#### L7: One entity per query
Label: Decision
Description: Each query targets a single person, place, organization, work or concept. Do not combine several entities or hops into one query, and decompose a claim into separate entity queries.

#### L8: Use names exactly as written, in full
Label: Output
Description: Use entity names exactly as they appear in the claim or passage, in full and with any qualifiers such as a region. Quote multi-word proper names. Do not shorten them to an ambiguous form or swap in a broader class or family the text does not state.

#### L9: Add a disambiguating qualifier
Label: Procedure
Description: When a name could title several articles, or an earlier search returned the wrong one, append a short parenthetical type, role, year, season or version taken from the claim or passage context, such as (film), (band), (actor) or a year. Leave names that are already specific unchanged.

#### L10: Query every plausible variant of a central ambiguous entity
Label: Decision
Description: If a central entity's name titles several different articles, issue a query for each plausible variant. The chain may depend on the less obvious one.

#### L11: Query the group or affiliation an entity is described by
Label: Decision
Description: When the claim describes an entity by its role or membership in a group (a committee, board and so on), query that group. Alternatively, combine the entity name with the group context word, so the query hits the specific article.

#### L12: Bounded number of queries per step
Label: Decision
Description: Generate only a few queries per step: roughly 3-6 when first decomposing a claim and 2-4 per follow-up, rather than a long exhaustive list.

### T3: Mining retrieved evidence for next hops

#### L13: Follow names in passages that lack their own article
Label: Procedure
Description: Entities named inside retrieved passages (people, places, organizations, works, species, languages, concepts) usually have their own article that has not been retrieved. Compare the claim's entities and the passage's names against the titles already held, and query each one that is still missing.

#### L14: Read passage bodies, not only titles
Label: Tool
Description: Passages have the form 'Title | abstract'. The entities worth following are named in the body text after the separator, so read the body of every passage.

#### L15: Summaries must list candidate entities
Label: Procedure
Description: When summarising passages, explicitly list every named person, place, organization, work or concept that could have its own article, such as the band behind an album or the full named cast and crew. Later steps use these names as queries, and an omitted name breaks the chain.

#### L16: Summaries state supported and unsupported claim parts
Label: Procedure
Description: Alongside the entity list, state which parts of the claim the passages support or fail to support, and which entities' articles are still missing.

#### L17: Use a mechanically extracted name list as a checklist
Label: Tool
Description: A list of names extracted directly from passage text is more complete and less lossy than summaries. Cross-check it against the retrieved titles, treat names not yet retrieved as strong candidates to query by exact name, and add any other names seen in the text.

#### L18: Order candidate sources: passage-name list before summaries
Label: Decision
Description: Handle missing-entity candidates from the passage-extracted name list first, and then those found only in the summaries.

#### L19: Resolve unnamed subjects and role descriptions from passage text
Label: Procedure
Description: When the claim refers to an entity by pronoun, role or descriptor ('it', 'the lead vocalist', 'the territory', 'a genus containing X'), find the entity that fills that reference in the retrieved text. Query or select that name, even though the claim does not name it.

#### L20: Negated or contrasted entities still count
Label: Domain
Description: An entity named in the claim is worth retrieving even when the claim negates or contrasts it ('not X', 'rather than Y'). A supporting article exists for the entity itself, whatever role the claim gives it.

#### L21: Include closely associated people, such as full cast and crew
Label: Domain
Description: People closely tied to a retrieved entity (spouse, founder, director, star, successor, replacement) are likely needed pages. For a film, book or other work, list all named cast and crew, not only the most famous.

#### L22: Add the genus for species names
Label: Domain
Description: When a passage gives a species by its Latin binomial, also output the genus name on its own. The genus often has its own article and is the more likely title.

#### L23: Do not skip an entity because a related one is listed
Label: Decision
Description: Do not omit a needed entity just because a related entity is already listed. List every relevant kind.

#### L24: Trust the passage text over the claim's wording
Label: Domain
Description: The claim may be imprecise or contain errors. Treat the retrieved passage text as authoritative when choosing entity names.

#### L25: Order extracted entities by relevance and unretrieved status
Label: Decision
Description: Rank extracted entities by relevance to the claim, putting those whose own page has not yet been retrieved ahead of those already retrieved.

### T4: Bridge, concept and context articles

#### L26: Infer unnamed bridge entities from world knowledge
Label: Decision
Description: Multi-hop claims depend on entities they do not name and passages may not mention (the writer of a song, the director or star of a film, an organization's founder or location, a parent category). Suggest such entities from your own knowledge when confident they have an article. Prefer the specific instance over a generic type, use full title-style names, put the most relevant first, and cap the number.

#### L27: Use only names stated in the evidence; no speculative guesses
Label: Decision
Description: Take entity names verbatim from the claim, summaries or passages. Do not guess entities that are merely referred to without a name, do not substitute entities you think should be the answer, and do not generate hypothesis queries for entities absent from the evidence.

#### L28: Limit speculative queries to central entities
Label: Decision
Description: Only query entities clearly central to the claim or the most relevant new subjects. Do not generate broad tangential queries or try to enumerate every conceivable subject.

#### L29: Concept, category and type articles are targets
Label: Domain
Description: When the claim or a passage refers to a concept, category, genre, technique, classification or type (a voice range, a territory type, a peninsula, a glam metal band), the article for that concept itself is often needed. Query or select it even if it seems obvious or already covered, and add broader hub pages alongside the specific entity.

#### L30: Search for list, overview and event articles
Label: Decision
Description: Besides entity pages, query for list, overview and event articles that tie entities together. Examples are discographies, filmographies, season, election, tournament or battle pages, regional or airline overviews, and 'Death of <person>' pages. Events and works mentioned in passing are frequently the missing connecting hop.

#### L31: Do not return an empty bridge list when candidates exist
Label: Decision
Description: Return an empty list of missing bridge entities only if no passage names any missing entity. Otherwise propose the most relevant ones, and generously include names mentioned in passing (a player in a match, a show an actor appeared on, an album by a band) even when unsure they are the bridge.

### T5: Search economy and re-querying

#### L32: Do not repeat queries or target already-retrieved articles
Label: Decision
Description: Follow-up queries must be new. Do not reissue earlier queries, make initial queries overlap, or propose entities whose articles are already among the retrieved titles.

#### L33: Re-query exact titles of relevant documents that have been displaced
Label: Tool
Description: The retrieval result set is capped, and a relevant article held at a weak rank, or cut by the cap, may be displaced by similarly named or newly added articles. Re-querying its exact title (with disambiguator, copying the spelling even if the claim spells it differently) restores it to the top rank. Do this only for articles relevant to the claim, prioritising its subjects and the works or places it names.

### T6: Candidate selection and reranking

#### L34: Rank by joint relevance across passages
Label: Domain
Description: A multi-hop claim's supporting articles are often relevant only together. A passage explicitly named by a central passage, or one that ties several retrieved passages or claim entities together, is supporting even if it looks unrelated on its own. Prefer candidates that the claim or snippets connect to a claim entity, including through a chain.

#### L35: Exclude list, disambiguation and index pages when selecting
Label: Decision
Description: Do not select or secure disambiguation or index pages, list articles, or articles with no connection to the claim.

#### L36: Exclude same-name articles about a different subject
Label: Decision
Description: An article that only shares a name with a relevant entity but is about a different subject is not relevant. Prefer the variant whose disambiguator matches the claim's domain over a bare-name one.

#### L37: Do not chase peripheral people
Label: Decision
Description: Do not query or select purely peripheral figures (other bands' members, producers, tangential persons) unless they are the claim's subject or link claim entities.

#### L38: Select at most the cap, as a short complete list
Label: Output
Description: When selecting passages, output only the indices of the top 21 most relevant (all of them if fewer), and do not rank the rest. A short, complete list is better than a long one that is truncated.

#### L39: Select a minimum number, most relevant first
Label: Output
Description: The selection should contain at least a minimum number of candidates (about 8). Order the chosen indices from most to least relevant.

#### L40: Choose only from the supplied candidates
Label: Output
Description: When choosing from a supplied candidate list, copy names from it and do not invent new ones. The only permitted edit is adding a disambiguating parenthetical.

## Notes from discovery (ignored by the parser)

- Several rules look opposed but concern different situations, so they are kept separate. 'Infer unnamed bridge entities from world knowledge' contrasts with 'Use only names stated in the evidence'. 'Search for list, overview and event articles' contrasts with 'Exclude list, disambiguation and index pages when selecting'. 'Do not chase peripheral people' contrasts with the 'every named entity' and 'closely associated people' lessons.
- The cap-cut and weak-rank re-query rules are merged into one lesson, as are the 'skip already-retrieved entities' and 'do not repeat a query' rules.
- Adding a year, season or version to a query is treated as a refinement of the disambiguating-qualifier lesson.
- The entity-extraction lessons (names from passages, resolving unnamed subjects, passage-name checklist) describe a program that extracts entities and writes summaries. A program with different modules may realise them in different places.
- (validator) possible provenance wording 'earlier' in: When a name could title several articles, or an earlier search returned the wrong one, append a short parenthetical type
- (validator) possible provenance wording 'Later' in: When summarising passages, explicitly list every named person, place, organization, work or concept that could have its 
- (validator) possible provenance wording 'earlier' in: Follow-up queries must be new. Do not reissue earlier queries, make initial queries overlap, or propose entities whose a
