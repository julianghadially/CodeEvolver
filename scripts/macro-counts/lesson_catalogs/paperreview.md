# Prompt-lesson catalog: paperreview

44 lessons in 6 themes. By label: Task 4, Domain 6, Tool 0, Procedure 6, Decision 23, Output 5.

## Lessons

### T1: Venue calibration and accept/reject threshold

#### L1: Calibrate to a roughly 50% acceptance rate
Label: Task
Description: The venue accepts about half of submissions, so the accept/reject threshold should match that balanced prior. Do not default to blanket harshness or blanket leniency.

#### L2: Decide borderline papers by balancing merits
Label: Decision
Description: Accept when contributions or strengths clearly outweigh flaws, and reject when flaws clearly outweigh contributions. Do not default to either outcome for a borderline paper.

#### L3: Acceptance rate does not excuse incremental work
Label: Decision
Description: The venue's roughly 50% acceptance rate must not be used as a reason to accept a paper whose own contribution is incremental. The paper must meaningfully advance the field.

#### L4: Accept bar: clearly strong, novel, well-validated contribution
Label: Decision
Description: Accept only when the core contribution is clearly strong: a novel, well-validated advance that others are likely to build on. Technical soundness is necessary but not sufficient.

#### L5: Independent adjudication of disagreeing reviews
Label: Decision
Description: When two prior reviews disagree, read the paper and judge which verdict the evidence better supports. Do not default to either reviewer and do not average their scores.

### T2: Score scales, probabilities and output consistency

#### L6: Overall score maps to the decision
Label: Output
Description: The accept/reject boundary sits between Overall 5 and 6: 6 or above means Accept, and 5 or below means Reject (5 is borderline-below, 6 is lean accept). The Decision field must agree with the Overall score.

#### L7: Bimodal use of the Overall scale
Label: Decision
Description: Most papers should receive a clear below-threshold score (4 or lower) or a clear above-threshold score (7 or higher). Keep 5 and 6 for genuinely undecided borderline papers.

#### L8: Overall score anchored to significance bands
Label: Output
Description: Tie the scale to significance. 1-3 is a clear reject (unsupported or flawed), 4-5 is a reject (adequate but incremental), and 6 is borderline (significant with material limitations). 7-8 is accept (significant, well-supported) and 9-10 is strong accept or award. A well-executed incremental paper scores 4-5.

#### L9: Acceptance probability is calibrated, not Overall rescaled
Label: Output
Description: Give an integer 0-100 estimate of the chance the paper is accepted, judged against the full submission pool. It must not simply be Overall times 10. A median or borderline paper gets about 50, a clearly strong paper 70 or more, and a clearly weak paper 30 or less.

#### L10: Probability bands tied to the critical weakness
Label: Decision
Description: If the most serious weakness undermines the central claims, the acceptance probability should be 45 or below even when there are other strengths. If strengths clearly outweigh weaknesses, it should be 65 or above.

#### L11: Record the critical weakness and an undermining-claims flag
Label: Output
Description: State the single most serious weakness and give an explicit true/false judgment on whether it undermines the central claims. The probability and decision must be consistent with that flag.

#### L12: Always emit a verdict, even from truncated prior reasoning
Label: Output
Description: If a previous review lost its verdict to truncation or malformed output, produce the verdict fields anyway (overall, confidence, accept/reject). If the reasoning was cut off, decide from what was written rather than leaving the fields empty.

#### L13: Magnitude-of-advance rubric
Label: Domain
Description: Advance over prior work is graded on 4 levels. 4 opens a new direction or solves an unsolved problem. 3 is a significant, non-obvious advance. 2 is a meaningful improvement whose core ideas are already known. 1 is a minor variation or straightforward application of existing techniques.

#### L14: Evidence-sufficiency rubric
Label: Domain
Description: Support for the central claims is graded separately from novelty, on experiments, theory and ablations. 4 is fully supported, 3 is mostly supported with minor gaps, 2 is partially supported with significant gaps, and 1 is unsupported or contradicted.

### T3: Review reasoning procedure

#### L15: Structured reasoning order before scoring
Label: Procedure
Description: Reason in a fixed sequence before assigning scores: summarize the contribution and claims, name the main strength, name the single most serious weakness, weigh whether that weakness undermines the central claims or the strength outweighs it, then give the overall assessment.

#### L16: Steelman both sides before deciding
Label: Procedure
Description: Before committing to a decision, write out the concrete evidence for acceptance and the concrete flaws supporting rejection, then weigh them. The order or format of the two lists does not matter.

#### L17: Compare the single strongest arguments
Label: Procedure
Description: Identify the single most compelling argument for acceptance and the single most compelling argument for rejection. Decide which is more decisive for the paper's overall merit.

#### L18: Check central claims against the evidence
Label: Procedure
Description: Identify the paper's central claims and check whether the evidence actually supports them. Classify each weakness as disqualifying or merely a limitation.

#### L19: Two-question final decision check
Label: Procedure
Description: Ask whether the core contribution is significant enough for a selective venue once fixable weaknesses are set aside. Also ask whether any inherent, contribution-undermining weakness remains. Accept only if the first is yes and the second is no.

#### L20: End reasoning with an explicit significance verdict
Label: Procedure
Description: Finish the reasoning with an explicit verdict: a significant advance others will build on, or adequate but incremental. Let that verdict determine the overall rating and the decision.

### T4: Weighing flaws: fatal, fixable, substantive

#### L21: Judge contribution, not presentation polish
Label: Task
Description: The decision rests on the scientific contribution (significance, originality, technical soundness and adequacy of evidence), not on writing or presentation quality. Clear writing does not rescue a weak contribution, and rough writing does not sink a strong one.

#### L22: Fatal flaws versus non-fatal limitations
Label: Decision
Description: Separate disqualifying flaws, which invalidate the contribution (a wrong central claim, fundamentally unsound methodology, results that do not support conclusions), from non-fatal limitations. Reject for the former. A significant advance with non-fatal limitations is acceptable.

#### L23: Separate substantive concerns from fixable presentation concerns
Label: Decision
Description: Before scoring, sort concerns into substantive ones (method, claims, evaluation that undermine the contribution) and fixable presentation ones (typos, unclear writing, missing details, formatting, incomplete sections). Base the decision on the substantive ones. Fixable issues alone must not push a sound, significant paper below the threshold.

#### L24: Accept sound genuine contributions despite fixable presentation flaws
Label: Decision
Description: A paper with a real, sound, non-incremental contribution and no fundamental flaw should be accepted even with typos, formatting problems, awkward phrasing, broken references or truncated text.

#### L25: Revision-fixable versus inherent weaknesses
Label: Decision
Description: Distinguish weaknesses the authors could plausibly fix in a revision (incomplete proof, underspecified details, minor presentation gaps) from inherent ones (incrementality, lack of understanding, narrow scope by design). Fixable issues are common in accepted papers and alone do not warrant rejection.

#### L26: Contribution-undermining versus peripheral weaknesses; the single worst weakness decides
Label: Decision
Description: Identify the central claimed advance and the single most serious weakness. If it undermines the central claim, including a missing or inadequate evaluation of that claim, lean toward rejection regardless of other strengths. Peripheral flaws alone should not cause rejection, and many minor strengths do not rescue a paper with one critical flaw.

#### L27: Do not explain away substantive concerns
Label: Decision
Description: Do not dismiss an identified substantive concern as minor, fixable in revision, reasonable for a new task, or not undermining the core. Narrow evaluation, a missing key baseline, overstated claims and an unsound method are substantive by definition, so you cannot list them and then declare there are no substantive concerns.

#### L28: Cumulative weight of substantive flaws
Label: Decision
Description: Several individually non-fatal substantive flaws compound. A paper is not clearly above the threshold just because each flaw is non-fatal, and the top scores are reserved for papers that are genuinely strong on substance.

#### L29: Incorrect derivations and violated assumptions are substantive flaws
Label: Domain
Description: A derivation that is legible but mathematically or logically wrong, a violated assumption, or a result that contradicts the central claim is a serious soundness flaw. It often merits rejection even in an otherwise competent paper, and it is not a rendering artifact.

#### L30: Non-fatal technical weaknesses lower Soundness but do not force rejection
Label: Decision
Description: A non-fatal technical weakness (missing ablation, acknowledged limitation, arguable approximation) should lower the Soundness score. It should not by itself turn a genuine contribution into a reject.

#### L31: Evaluation completeness is not unsoundness
Label: Decision
Description: Gaps in evaluation completeness (missing baselines, error bars, ablations, broad datasets) are weaknesses to weigh, not evidence the method is incorrect. Do not lower Soundness for them. Rate Soundness 3 or above when the method appears correct, and give 2 only when a specific incorrect method or unsupported claim is identified.

#### L32: Clarity problems that block evaluation are substantive
Label: Decision
Description: Clarity problems that prevent a reader from understanding the method or evaluating the results are real weaknesses, unlike cosmetic glitches.

### T5: Novelty, significance and adequacy of evidence

#### L33: Judge on significance plus adequate evidence
Label: Task
Description: Accept a paper whose contribution is significant and backed by adequate evidence. Reject one whose central claims are not well supported or whose contribution is not significant.

#### L34: Incremental contributions warrant rejection even if well executed
Label: Decision
Description: A minor extension or recombination of known techniques, with no new insight and no clear advance, is a reason to reject even when carefully executed and thoroughly tested. Incrementality is inherent, so it cannot be repaired in revision. Narrow results that fail to establish a significant advance can also justify rejection without any single fatal flaw.

#### L35: Incrementality rejects only when paired with an over-narrow evaluation
Label: Decision
Description: An incremental contribution is a reason to reject mainly when the paper also makes a superiority claim that its narrow evaluation cannot support. Incrementality alone is not decisive.

#### L36: Weak, missing or outdated baselines and narrow validation leave claims unestablished
Label: Domain
Description: Missing, unfair, outdated or non-state-of-the-art baselines, evaluation limited to narrow settings, or gains within noise mean the claimed improvements are not established. This is a serious weakness.

#### L37: Superiority headline resting on weak baselines or narrow evaluation leans reject
Label: Decision
Description: If the headline claim is state-of-the-art performance or a first result on a new task and it rests only on the authors' own variants or weak baselines with key competitors absent, lean toward reject. A gap this large can pull a borderline paper below the threshold even if the method is sound.

#### L38: Standalone contributions may be accepted despite narrow evaluation
Label: Decision
Description: A paper whose main contribution is a sound theoretical framework or analysis, a benchmark, dataset or tool, or a substantially new method or task formulation can be accepted on that value even with a narrow empirical evaluation. This holds when the narrowness does not undermine its central claim.

#### L39: Overclaiming: significant versus minor
Label: Decision
Description: Significant overclaiming, where the stated contribution exceeds what the experiments or theory show, justifies rejection. Minor or optimistic framing is common and is not grounds for rejection when the underlying contribution is valid and significant.

#### L40: Apply contribution-type-specific standards
Label: Task
Description: Classify the contribution as theory, method, empirical or system, and apply the matching standard. Theory is judged on the significance and correctness of the result. A method is judged on whether it demonstrably works and advances over prior approaches. An empirical paper is judged on the breadth and rigor of its experiments. A system is judged on usefulness and the research it enables.

### T6: Text-extraction artifacts in the paper

#### L41: Extraction artifacts are not paper defects
Label: Domain
Description: The paper text is machine-extracted, so garbled equations, tables, pseudocode, symbols, base64 strings, broken references, missing figures and extraction placeholders (for example an empty algorithm field) come from the pipeline, not the authors. Judge the science from the intelligible content. Do not reject, or lower Clarity, Presentation or Soundness, because of them.

#### L42: Unreadable evidence is not unsupported evidence
Label: Decision
Description: A claim counts as unsupported only when the legible evidence fails to support or contradicts it, not when the supporting table or figure could not be read.

#### L43: Garbled-table number mismatches are misreads
Label: Domain
Description: A numeric mismatch that comes from a garbled table is a reading artifact. It is not evidence that the authors exaggerated or misreported.

#### L44: Leniency for illegibility never waives substantive concerns
Label: Decision
Description: Forgiving unreadable content does not cancel substantive problems visible in the legible parts, such as too-narrow evaluation, missing key baselines or exaggerated claims. Illegibility must not rescue a paper whose visible evidence is weak, and it should not raise the bar either.

## Notes from discovery (ignored by the parser)

- Several extraction-artifact statements (garbled text, base64, placeholders for missing equations or algorithms, missing figures) were merged into one lesson because they lead to the same behaviour. The separate lessons in that theme are the ones that add a distinct rule about unreadable evidence, garbled-table mismatches, and non-waiver of substantive concerns.
- Some entries are opposite or conflicting variants and are kept separate on purpose. 'Non-fatal technical weaknesses lower Soundness' conflicts with 'Evaluation completeness is not unsoundness'. 'Revision-fixable versus inherent weaknesses' lists a missing baseline as fixable, while the substantive-concern lessons treat missing key baselines as substantive. 'Incremental contributions warrant rejection even if well executed' is the opposite of 'Incrementality rejects only when paired with an over-narrow evaluation'.
- The two advance/evidence rubrics are listed as separate scale definitions because each grades a different thing (novelty versus support). They are labelled Domain because they define vocabulary, though they also affect how scores are written.
- 'Decide borderline papers by balancing merits' and 'Calibrate to a roughly 50% acceptance rate' are kept apart. The first is a decision rule for weighing strengths and weaknesses, and the second is a belief about the venue's base rate.
