# How GEPA Candidate JSON Becomes a DSPy Program

## High-level flow

- **Inside GEPA (the `gepa` library)**, candidates are plain dicts:
  - `candidate: dict[str, str]` mapping **component names** → **text blobs**.
  - For normal predictors: `{"predictor_name": "new instruction text", ...}`.
  - For tool-optimized ReAct modules: special entries like  
    `{"tool_module:<extract_predictor_name>": "<JSON string with instructions + tools>"}`.

- **In DSPy's GEPA wrapper** (`dspy/teleprompt/gepa/gepa.py`):
  - Seed candidate for GEPA is built from the original `student` via `_build_seed_candidate`.
  - GEPA runs and returns a `GEPAResult` with `candidates` and `best_candidate`.
  - DSPy then **post-processes** the best candidate:

    ```python
    new_prog = adapter.build_program(gepa_result.best_candidate)
    ```

  - If `track_stats=True`, it also calls `DspyGEPAResult.from_gepa_result`, which maps **all** raw candidates to actual DSPy programs via `adapter.build_program(c)`.

So: **`Result.best_candidate` is turned into a DSPy program by `DspyAdapter.build_program` as a final post-processing step.**

---

## What `build_program` actually does

In `dspy/teleprompt/gepa/gepa_utils.py`:

- **Clone original student**: `new_prog = self.student.deepcopy()`.
- **Predictor text**:
  - All non-`tool_module` keys become `predictor_candidates[name] = instruction_string`.
  - For tool-optimized entries (`tool_module:...`), the JSON is parsed; any string-valued entries keyed by predictor name are merged into `predictor_candidates`.
- **Signatures**:
  - For each predictor in `new_prog.named_predictors()`, it calls  
    `pred.signature = pred.signature.with_instructions(...)`.
  - This **only swaps the instructions field**; the rest of the signature (input/output fields, types, and any stored metadata) is preserved.
- **Tools** (when `enable_tool_optimization=True`):
  - `_update_tool_descriptions` finds all `Tool` objects in the module tree and updates:
    - `tool.desc`
    - `tool.args[arg_name]["description"]` when provided.

After `build_program`, you have a normal **DSPy `Module`** with full signatures, any pre-existing demos, and updated instructions/tool descriptions.

### What build_program actually does
In `dspy/teleprompt/gepa/gepa_utils.py`:
```python
def build_program(self, candidate: dict[str, str]):
    new_prog = self.student.deepcopy()

    # 1) Start with plain string instructions
    predictor_candidates = {k: v for k, v in candidate.items() if not k.startswith(TOOL_MODULE_PREFIX)}

    tool_candidates = {}
    if self.enable_tool_optimization:
        for key, value in candidate.items():
            if not key.startswith(TOOL_MODULE_PREFIX):
                continue

            config = json.loads(value)

            for pred_name, instruction in config.items():
                if isinstance(instruction, str):
                    predictor_candidates[pred_name] = instruction

            tool_candidates.update(config.get("tools", {}))

    # 2) Update predictor instructions
    for name, pred in new_prog.named_predictors():
        if name in predictor_candidates:
            pred.signature = pred.signature.with_instructions(predictor_candidates[name])

    # 3) Update tool descriptions (ReAct tool config)
    if tool_candidates:
        self._update_tool_descriptions(new_prog, tool_candidates)

    return new_prog
```

---

## How this becomes "DSPy-style JSON" and how `program.load` fits in

The **saving/loading** mechanism is generic DSPy, not GEPA-specific:

- `Module.save(path)` (from `BaseModule.save`) persists **state** (parameters, including instructions and demos) to:
  - `path.json` or `path.pkl` when `save_program=False`.
- `Module.save(path, save_program=True)` persists a **full program** (architecture + state) as:
  - Directory `path/` containing `program.pkl` and `metadata.json`.

- **For state-only JSON**:
  - You create a base program with the same architecture, then call `program.load("state.json")` to load its saved instructions/demos/etc.
- **For full-program directories**:
  - You use `dspy.utils.saving.load("dir", allow_pickle=True)` to get back the exact module.
- `teleprompt/utils.save_candidate_program` uses `program.save(...)` to dump candidate programs, and later code calls `trial_program.load(trial["program_path"])` to restore them.

In the GEPA integration, **nothing special is added** on top of this: once `new_prog = adapter.build_program(...)` exists, you can call `.save()` yourself, and later `load()` behaves exactly like with any other DSPy program.

---

## Where do DEMOS come from?

**Short answer:** GEPA **does not create demos at all.** It preserves whatever demos the `student` already has; it only evolves text instructions (and optionally tool descriptions).

Details:

- GEPA's adapter builds a **reflective dataset** for the instruction proposer (`make_reflective_dataset`) using trajectories. These `ReflectiveExample` entries are **temporary supervision** fed into the reflection LLM (via `InstructionProposalSignature` or `ToolProposer`) to generate new instructions; they are **not stored as `.demos`** on the final program.

- The **only place demos are systematically created** in this codebase is in the few-shot / bootstrap utilities (e.g. `create_n_fewshot_demo_sets` in `dspy/teleprompt/utils.py`), which is separate from GEPA.

**Therefore:**

- If your `student` was produced by something like `BootstrapFewShot` (or already had `.demos` set), those demos are **carried through** because `build_program` deep-copies the student and only touches `signature.instructions` and tools.
- GEPA's optimization itself **does not synthesize or modify demos**; it only uses reflective datasets internally to improve instructions.

---

## Summary

- **GEPA candidate JSON** = dict mapping component names to text (or JSON for tool modules).
- At the end, **`gepa_result.best_candidate` is turned into a DSPy `Module`** via `DspyAdapter.build_program`, which:
  - Clones the original student,
  - Rewrites `signature.instructions` per predictor,
  - Optionally updates tool descriptions.
- To get a **DSPy-style JSON file**, you then call `new_prog.save("program.json")` or `new_prog.save("prog_dir", save_program=True)`. Later you restore with either `program.load("program.json")` (state-only) or `dspy.load("prog_dir", allow_pickle=True)` (full program).
- **DEMOS**:
  - Are **not created by GEPA**.
  - Whatever demos were present on the input `student` remain on the optimized program.
  - GEPA's internal reflective datasets are separate and not persisted as demos.
