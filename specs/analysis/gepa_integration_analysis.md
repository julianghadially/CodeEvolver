# GEPA Integration Analysis

## Goal of integration
We need to be able to integrate with other AI frameworks besides DSPY, so we can optimize the AI systems that our clients are already using. The most important AI frameworks we need to integrate with include Langchain, LangGraph, and OpenAI API. 

To determine how to integrate, we must first identify exactly what information we need. One of the powerful components of GEPA is being able to observe traces for every module, including the tool calls it makes and any sub modules it calls. A lot of simple, wrapping methods fall short – they track inputs and outputs, but they do not track tool calling or subagent calling.

Technically, because we have a coding agent, we can embed deep observability tracing inside an AI system. However, there may be a more elegant solution that does not result in providing code back to a user that is heavily tagged, and potentially messy. Especially when a AI framework already handles tracing (which is the case with the DSPy) why would I want to trace twice?

One such solution is to use existing tracing providers such as Langfuse and Langsmith. Another solution is to wrap the open AI client. Thoroughly investigate each option and their advantages and disadvantages before selecting the chosen solution.

When GEPA runs, it requires a prompt candidate JSON and tracing data. The prompts in the candidate JSON are not the final prompt, they are instruction text that is ready to compile alongside any number of variables into a final prompt. DSPy handles this prompt templating strategy. They picked a prompt template strategy so that the input fields always get included no matter how the optimizer edits the prompt.

Considerations
1. F strings or placeholder charcter strategy ${} or {{}} could be used as a prompt template
2. Clients will upload their prompts blindly, so they may already use some of these characters {{}}.
3. Optimizers must be able to make changes to prompt text without accidentally dropping important context variables. Of course, if one iteration leaves out an important context, variable and performs poorly, it's not the end of the world.
4. A prompt templating strategy could work, including using DSP Y. signatures as our core logic, since the prompt templating is already built in.
5. We want it to be easy for users to share a path to their AI system pipeline. If they do not have to redefine prompts, that would be amazing, but if they do, it might be necessary. Especially if it helps the results..


## Analysis
### What GEPA Needs: Core Data Requirements                                       
                                                                                
  GEPA operates on exactly 4 data structures, regardless of framework:          

  1. Candidate (dict[str, str])                                                 
                                                                                
  {
      "_code": '{"git_branch": "...", "parent_module_path": "...", ...}',
      "module_1.predict": "instruction text...",   # optimizable prompt
      "module_2.predict": "instruction text...",   # optimizable prompt
  }
  Each key (except _code) is a named LLM call point whose instruction text GEPA
  can mutate.

  2. Evaluation Traces (per example)

  {
      "trace": [
          {
              "signature_key": "fingerprint_identifying_which_module",
              "inputs": {"field": "value"},
              "output": {"field": "value"},     # or {"__failed__": True}
          },
          ...  # one entry per LLM call in the pipeline
      ],
      "example": {"claim": "..."},              # original input
      "prediction": {"verdict": "true"},        # final output
      "score": 0.8
  }
  The signature_key is how traces get matched back to candidate keys. In DSPy
  this is a fingerprint of sorted input/output field names.

  3. Reflective Dataset (dict[str, list[dict]])

  {
      "query_generator": [
          {"Inputs": {...}, "Generated Outputs": {...}, "Feedback": "Score:
  0.0\n..."},
      ],
      "verifier": [
          {"Inputs": {...}, "Generated Outputs": {...}, "Feedback": "Score:
  0.5\n..."},
      ],
      "_code": [
          {"input": {...}, "output": {...}, "score": 0.0, "exception": "..."},
      ]
  }
  Maps each component to its low-scoring examples with feedback, so the
  reflection LM knows what went wrong.

  4. Scores (list[float])

  One metric score per example. This is framework-agnostic already.

### Orchestration Needs
- **Module discovery:** Must identify which LLM calls ("modules") exist and must extract system prompts
    - DSPy Concept: program.named_predictors(): 
    - What it provides: Discovers all LLM call points and their instruction text.
- **Trace capture:** Must intercept LLM calls and capture all inputs/outputs/tool-calls
    - DSPy Concept: bootstrap_trace_data()
    - What it provides: Captures per-call inputs/outputs/tool-calls during execution 
- **component key fingerprinting**: Must have a stable way to identify which call pertains to which module
    - DSPy: signature_key
    - What it provides: Matches trace entries to candidate keys

