# Prompt-lesson catalog: phantomwiki

53 lessons in 6 themes. By label: Task 3, Domain 14, Tool 7, Procedure 13, Decision 11, Output 5.

## Lessons

### T1: Answer-set semantics and scoring

#### L1: Most questions have several correct answers
Label: Task
Description: The answer is the complete set of valid entities or values, scored as a set. A missing valid answer costs as much as a wrong one. "The X of Y" does not imply a single entity.

#### L2: Chains branch at every hop
Label: Procedure
Description: Each hop of a relation chain can fan out to several entities. Follow every branch to the end and keep only entities at the end of the complete chain, not intermediate ones.

#### L3: Aggregated counts are a set of distinct per-person counts
Label: Task
Description: When a count question's subject matches several people, count the relation for each person separately. Return the set of distinct counts. Never return a sum or a single person's count.

#### L4: Descriptive subjects can match many people; a proper name is one
Label: Task
Description: Besides attribute-defined subjects, a subject phrased as a relation of a named person ("the grandson of X") can also resolve to several people, which makes a count question an aggregation. A bare proper name is one person and gets a single count, returned as the number rather than the list of relatives.

#### L5: Answer type must match the question type
Label: Output
Description: "How many" questions return counts only, never names. "Who/What" questions return names or attribute values, never a bare "0" placeholder. Any wrong-type item should be dropped.

#### L6: Value formatting and deduplication
Label: Output
Description: Write full names exactly as they appear in the corpus and counts as strings. Deduplicate answers, since the metric compares sets. Count distinct entities, not repeated mentions.

### T2: Zero, empty and partial answers

#### L7: Return partial verified answers rather than an empty list
Label: Output
Description: An empty answer always scores zero. If the full set cannot be completed, return every answer actually verified. One dead-end person or branch must not empty the whole answer.

#### L8: Zero means the entity exists and has none
Label: Output
Description: Answer "0" only when the counted entity genuinely exists and has none of the relation. If an intermediate link of the chain does not exist, there is no entity to count, so the path contributes nothing rather than "0".

#### L9: An unfinished chain is not a zero
Label: Decision
Description: If steps run out before the chain is resolved, do not default to "0" or guess. Report only counts confirmed at real final entities, or nothing for that path, because a guessed zero pollutes the aggregate.

#### L10: Dead-end branches do not end the search
Label: Procedure
Description: If one branch or parent line lacks the needed links, it contributes nothing (or zero for an upward count). Keep following the other branches and use the other parent's line.

### T3: Kinship derivation and relation definitions

#### L11: Only direct relations are stored; derive the rest by chaining
Label: Domain
Description: Articles list only mother, father, son, daughter, sister, brother, husband, wife, friends, and the attributes date of birth, occupation, hobby and gender. Every other relation (aunt, cousin, grandparent, in-law, great-, second-, once removed) must be computed by chaining across articles. Searching for the relation word itself finds nothing; search people by name instead.

#### L12: Compound relations are defined by chaining basic ones
Label: Domain
Description: Definition table: parent = mother and father, child = son and daughter, sibling = sister and brother, spouse = husband and wife. A grandparent is a parent of a parent. Each "great-" adds one hop. Aunt/uncle is a sibling of a parent. Nephew/niece is a child of a sibling. Cousin is a child of an aunt or uncle. Mother/father-in-law is a parent of the spouse, and son/daughter-in-law is the spouse of a child. Male/female variants are the same relation filtered by the gender attribute.

#### L13: Generation depth must be exact
Label: Domain
Description: Count generations exactly, upward and downward. "Second" aunts and uncles are siblings of great-grandparents, one generation deeper than "great-". The common error is stopping one generation short. Stop at the depth the relation names, because going one hop further gives a different relation.

#### L14: Great-/second-/once-removed are generational prefixes, not ordinals
Label: Domain
Description: Words like "second", "first" or "third" in a relation name are not list positions or birth order. "Second aunt" is not the second-born aunt, and the answer is not to pick the Nth item.

#### L15: Sister/brother-in-law is the spouse's sibling only
Label: Domain
Description: Brother/sister-in-law means the sibling of one's spouse, NOT the spouse of one's sibling. A sister's husband is not a brother-in-law, and including such people adds wrong answers.

#### L16: Sister/brother-in-law covers both paths
Label: Domain
Description: Brother/sister-in-law includes both the sibling of the spouse and the spouse of a sibling, and plural forms collect across all spouses or siblings.

#### L17: First cousin once removed is downward only
Label: Domain
Description: First cousin once removed is the child of a first cousin, one generation below. It is not a parent's first cousin and not a second cousin.

#### L18: First cousin once removed can run either direction
Label: Domain
Description: "Once removed" means one generation apart from a first cousin, so it covers both a cousin's child and a parent's cousin.

#### L19: Second cousin is a 5-hop chain
Label: Domain
Description: A second cousin is the child of a parent's first cousin (parent, parent, sibling, child, child), sharing great-grandparents. It is a different, longer chain than first cousin once removed.

#### L20: Spouses of blood relatives are in-laws, not the blood relation
Label: Domain
Description: An uncle's wife is an aunt-in-law, not an aunt, and does not count toward counts of aunts. Include a spouse only when the question literally names an in-law relation.

#### L21: Friends are not relatives
Label: Domain
Description: The friend relation is separate from family. Do not climb it as a family link; read the friend's own article only for what the chain asks about that friend.

#### L22: Ancestor relations span both parental lines
Label: Procedure
Description: Going up the tree, follow both mother and father at every parent hop, so the set doubles each hop (2 parents, 4 grandparents, 8 great-grandparents). "The parent of X" returns both parents even when X was reached through one parent. Counts of ancestors add across both lines, and following only the paternal line misses answers.

#### L23: Up-then-down chains need all descendants of the ancestor
Label: Procedure
Description: When a chain climbs to an ancestor and then asks for a descendant relation of that ancestor, enumerate all descendants at the required depth across every child line. The original person is only one of them.

#### L24: Single-valued versus multi-valued relations
Label: Domain
Description: A person has one mother, one father and normally one spouse. Siblings, children, friends, cousins, nephews and the like are multi-valued.

#### L25: Missing listed relations mean none exist
Label: Domain
Description: If a person's article lists no parents, siblings or children, none are recorded. That person is a root and has no ancestors, so move on to other branches instead of re-querying.

#### L26: Read right to left from the anchor
Label: Procedure
Description: Parse a relation chain starting from the innermost anchor (the named person or the person with the given attribute) and apply each outer relation in turn.

#### L27: Finish one chain before starting the next
Label: Procedure
Description: Trace one full chain to its end (depth-first) before opening the next branch or the next matching person. Do not spend all steps enumerating anchors upfront while leaving chains unfinished.

#### L28: Read whole articles and record all relatives
Label: Procedure
Description: Read each retrieved article for every listed relation, not just the first one. Record all relatives revealed before moving to the next hop, because a single article often answers a whole step.

#### L29: Counting a direct relation means counting listed names
Label: Procedure
Description: For a direct field (friends, sons, daughters, siblings, parents, spouse), the count is the number of distinct names listed in that field. Union relations combine their fields, and an absent field gives 0.

#### L30: Read on-page values directly when no chain is needed
Label: Decision
Description: If the requested value is printed in the matching person's own article (an attribute, a direct-relation list or its count), read it straight from the gathered articles without further searching.

### T4: Attribute-anchored multi-person questions

#### L31: Attribute values are shared by many people
Label: Domain
Description: "The person whose date of birth, occupation or hobby is V" typically matches dozens to thousands of people. This is the full set of anchors, not an ambiguity, and all of them matter.

#### L32: Attribute matching is exact
Label: Decision
Description: A person matches only if the attribute equals the stated value exactly. A hobby of "ballet dancing" is not "dancing", a "soil scientist" is not a "scientist", and a near-miss date is not a match.

#### L33: Same-attribute question has a trivial answer
Label: Decision
Description: When the attribute asked about is the same one used to pick the person, with no relation in between, every matching person has that value. The answer is just that value, with no retrieval needed. The shortcut does not apply if the attributes differ or a relation sits in between.

#### L34: Evaluate the whole question per anchor and union the results
Label: Procedure
Description: For multi-anchor "Who/What" questions, run the chain for each matching anchor and take the union of all names or values. For counts, collect each anchor's count.

#### L35: Completeness over large sets
Label: Decision
Description: Do not stop after one or a few answers. Do not assume unexamined people give the same result as the sampled ones. Rare values hide in the tail of a large set. Not finding more is not proof there are none. Re-scan before finishing.

#### L36: Sample diversely when the set is too large
Label: Decision
Description: Only when the matching population is far too large to trace within the budget, trace a subset spread across the whole list rather than the first few, to maximise distinct values found.

#### L37: Breadth versus depth depends on chain depth
Label: Decision
Description: For shallow questions (a direct relation of the anchor), look at as many anchors as possible, since distinct counts spread across the population and rare ones are on later pages. For deep or derived chains, thorough tracing of fewer anchors beats widening the sample, because more pages overwhelm reasoning.

#### L38: Delegate each anchor to a fresh focused investigation
Label: Procedure
Description: Tracing many anchors in one long trajectory exhausts steps and loses context. Resolve each anchor's chain separately in a fresh, isolated investigation and then merge.

#### L39: Use per-anchor delegation only for attribute-anchored questions
Label: Decision
Description: Delegation or batch tools apply to questions whose subject is an attribute-defined group. Questions about a single named person are resolved directly.

#### L40: Earlier findings are probably incomplete
Label: Procedure
Description: If a previous attempt has already found answers, treat them as likely incomplete. Look for additional answers through branches or people not yet examined, skip people whose answer is already known, and return the merged complete list.

### T5: Retriever and tool behaviour

#### L41: Semantic retrieval is capped and ranks near-misses above exact matches
Label: Tool
Description: A plain semantic search returns only a handful of passages, with per-query server caps. It buries exact attribute matches under similar ones, so one call cannot enumerate everyone sharing a value. Use a dedicated exact or attribute lookup that filters for the value verbatim.

#### L42: Query with the bare attribute value, not a sentence
Label: Tool
Description: For attribute lookups, pass the specific value (optionally with its attribute type), not a full question or relation phrase. Including the type biases results toward the Attributes field and away from lookalikes such as an occupation "microbiologist" for a hobby "microbiology".

#### L43: Dates are matched poorly and need varied phrasings
Label: Tool
Description: Dates are tokenised into fragments, so near-dates outrank exact ones and true matches fall beyond the cap. Use several phrasings (with a type prefix, bare, first hyphen replaced by a space) and verify the full date. Some matches may be unreachable.

#### L44: The retriever is nondeterministic across calls
Label: Tool
Description: Repeating the same exact-match query can surface additional matching people, so a further call may reveal people missed earlier.

#### L45: Large match sets come back as a bounded sample or pages
Label: Tool
Description: The attribute tool may show only a sample, a page, or names-only entries for large sets. The stated total can exceed what is shown, so page through or look up names separately to read their articles.

#### L46: Look people up by full name
Label: Tool
Description: Retrieval by exact full name returns that person's own article (a few results suffice). Other returned articles are relatives who mention them, which can reveal parents or siblings that the person's own article omits.

#### L47: Batch lookups save steps
Label: Tool
Description: When a hop yields several people, fetch all their articles (or a person plus their immediate family) in one call instead of one search per name, so more branches fit in the step budget.

### T6: Grounding, precision and finalization

#### L48: Ground answers only in retrieved observations
Label: Decision
Description: Facts and answer values must come from articles actually retrieved. Never simulate or predict tool results, never invent values, and omit any entity whose article was not retrieved rather than guess.

#### L49: Precision on Who/What chains: no padding from invalid branches
Label: Decision
Description: Include only entities reached by a complete valid chain. Do not add people from a spouse's family, a wrong generation or an unlisted link. Complete fewer chains correctly rather than many wrongly, since wrong answers cost as much as misses.

#### L50: Preserve every verified finding to the end
Label: Procedure
Description: Record confirmed answers as they are found, and state the complete list of distinct answer values before finishing. Findings recorded along the way should be merged into the final answer, not lost to a shorter final summary.

#### L51: Extraction should carry forward the reasoning's conclusion and re-add dropped values
Label: Output
Description: The final answer should start from the values the agent concluded and add any value found in earlier thoughts or observations that the final summary dropped. A value is dropped only if a later step explicitly rejected it. Never invent values absent from the trajectory.

#### L52: Do not re-interpret the question at extraction
Label: Procedure
Description: The reasoning steps already interpreted the question and chain. Extraction should not re-parse it differently or re-derive the chain from scratch. It may still exclude a candidate that observations show is invalid.

#### L53: Spend steps on new lookups
Label: Decision
Description: Steps are limited. Do not repeat an identical or merely rephrased search, since it returns the same top articles, and do not re-read what is already in hand. Look up a new relative, or follow the other branch. At the same time, missing answers cost more than a few extra steps.

## Notes from discovery (ignored by the parser)

- Some lessons have opposite variants because the prompts disagree. In-laws are defined as spouse's sibling only versus both spouse's sibling and sibling's spouse. First cousin once removed is downward-only versus either direction. Repeating a query is useful because the retriever is nondeterministic, versus wasteful because the same top results come back. Empty-versus-zero is stated both as 'missing intermediate link gives empty' and as 'existing subject with none gives 0, empty only if the subject does not exist'. Each variant is listed separately so a reviewer can detect either wording.
- Definition tables are merged into one lesson (compound relations from basic ones). Contrastive definitions (in-law spouse's sibling only, once removed downward only, second cousin versus once removed, blood versus in-law, friend not a relative, ordinal misreading) are kept as separate lessons, per the definition-table rule.
- Code-only behaviour is excluded: local indices, thread pools, retry loops, count sanitisation, answer flattening, finish-tool schema changes, caps and worker counts, and regexes.
- Step roles and plumbing prompts are excluded: filter-only identification steps, per-anchor sub-question templates, 'you merge the lists' reconcilers, and JSON output formats. Their task-specific content is captured only in the lessons above.
- The instruction to pass the answer list to the finish tool was treated as plumbing, not as a lesson.
- The 'earlier findings are probably incomplete' lesson comes from continuation prompts that list already-found answers. It is kept as a procedure because it teaches that a first pass tends to under-enumerate.
- (validator) possible provenance wording 'later' in: For shallow questions (a direct relation of the anchor), look at as many anchors as possible, since distinct counts spre
- (validator) possible provenance wording 'Earlier' in: Earlier findings are probably incomplete
- (validator) possible provenance wording 'earlier' in: Repeating the same exact-match query can surface additional matching people, so a further call may reveal people missed 
- (validator) possible provenance wording 'earlier' in: The final answer should start from the values the agent concluded and add any value found in earlier thoughts or observa
- (validator) possible provenance wording 'earlier' in: The 'earlier findings are probably incomplete' lesson comes from continuation prompts that list already-found answers. I
