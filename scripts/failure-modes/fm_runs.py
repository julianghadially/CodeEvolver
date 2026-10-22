"""fm_runs.py -- shared, self-contained readers of the core runs for ledgers.py, build_packs.py and python_metrics.py.

WHY a module of its own: the ledgers (Python decides), the packs (the judge reads) and the metrics (Python counts) must all see
ONE reading of a run: the same evals, the same tree identities, the same start states, the same memory replay. Everything here
was copied from the analysis helpers of jg-full (codesig, t2mem_lib, t2glm_common, rescue_metrics, extract_iterations) and made
independent of them (no pandas, no import from jg-full, no precomputed analysis csv): it reads only the raw run folders
(history.json, history/<job>/*.jsonl, logs, architect traces, repo.bundle). Standard library only.

Read-only on the run folders. The only thing written is a scratch BARE git repository per run (fetch_bundle), under the
fm-private/git folder; nothing in jg-full or in the repository is touched.

What is in here, in order:
  1. the 24 core runs and the constants of the study (set sizes, tau table, model-specific SDs)
  2. low-level readers (chunked-string JSON, jsonl)
  3. Repo: persistent `git cat-file --batch` with two tree identities (sig = file content, ast_sig = normalised Python AST)
  4. the log parser (one record per iteration)
  5. the transcript reader (turn files -> steps)
  6. memory.md classification and replay
  7. RunData: everything of one run, loaded lazily and cached
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------------------------------------------
# 1. the core runs and the constants of the study
# ---------------------------------------------------------------------------------------------------------------
RUNS_ROOT_DEFAULT = Path(os.environ.get("CE_RUNS_DIR") or Path.home() / "ce-runs").expanduser()
PRIVATE_DEFAULT = Path.home() / "ce-analysis" / "fm-private"

MODELS = ("Qwen3", "GLM-5.2")
ARMS = ("freeform", "pareto")          # the names of unit_key.csv; the run folders say "pareto_evolution"
BENCHES = ("hover", "phantomwiki")
SEEDS = (1, 2, 3)

SET_ROWS = {"hover": 200, "phantomwiki": 100}   # size of the fixed comparison set
TRAIN_ROWS = 50                                 # the train subsample of every iteration

# tau = the expected spread (SD of the DIFFERENCE) between two evals of IDENTICAL code = sqrt(2) x the SD of one eval.
# Benchmark table of the debate (SD_C hover 0.022 / phantomwiki 0.013; SD_S hover 0.043 / phantomwiki 0.024), rounded as agreed.
TAU = {"hover": {"train": 0.061, "comparison": 0.031}, "phantomwiki": {"train": 0.034, "comparison": 0.018}}
# Model-specific pooled SDs of ONE train eval (sensitivity of the debate). The comparison-set noise is a property of the
# benchmark, not of the agent, and has no model-specific estimate: python_metrics keeps the benchmark SD there.
MODEL_SD_TRAIN = {("Qwen3", "hover"): 0.0496, ("GLM-5.2", "hover"): 0.0317,
                  ("Qwen3", "phantomwiki"): 0.0261, ("GLM-5.2", "phantomwiki"): 0.0207}


def tau_of(bench: str, which: str) -> float:
    """which: 'train' or 'comparison' (anything else, e.g. 'variance', counts as train: same 50-row samples)."""
    return TAU[bench]["comparison" if which == "comparison" else "train"]


@dataclass(frozen=True)
class RunSpec:
    run: str        # folder name below the runs root
    model: str      # "Qwen3" | "GLM-5.2"
    arm: str        # "freeform" | "pareto"
    bench: str      # "hover" | "phantomwiki"
    seed: int

    @property
    def arm_full(self) -> str:
        return "pareto_evolution" if self.arm == "pareto" else "freeform"


def run_name(model: str, arm: str, bench: str, seed: int) -> str:
    """Folder name of a core run. Qwen3: <arm>-<bench>-s<seed>-qwen3c. GLM-5.2: <arm>-<bench>-s<seed>, and the pareto runs
    were redone: ...-redo."""
    arm_full = "pareto_evolution" if arm == "pareto" else "freeform"
    if model == "Qwen3":
        return f"{arm_full}-{bench}-s{seed}-qwen3c"
    return f"{arm_full}-{bench}-s{seed}" + ("-redo" if arm == "pareto" else "")


def core_runs() -> list[RunSpec]:
    """The 24 core runs, ordered model, arm, bench, seed (the order of report.py)."""
    return [RunSpec(run_name(m, a, b, s), m, a, b, s) for m in MODELS for a in ARMS for b in BENCHES for s in SEEDS]


def spec_of(run: str) -> RunSpec:
    for s in core_runs():
        if s.run == run:
            return s
    raise KeyError(f"not a core run: {run}")


def unit_id(run: str, iteration: int, salt: str = "") -> str:
    """Opaque unit id "U-" + 8 hex digits of sha1("<run>|<iteration>") (the scheme of the prototype; `salt` is prepended when
    given). It is deterministic, so the same run and iteration always get the same id and a rebuild reproduces unit_key.csv.
    TRADE-OFF: without a salt it is NOT secret (anybody who knows the public run names can recompute it); it only keeps the id
    free of run, model, arm and seed text. build_packs.py takes --salt for a keyed variant."""
    return "U-" + sha1_hex((salt + "|" if salt else "") + f"{run}|{iteration}")[:8]


# ---------------------------------------------------------------------------------------------------------------
# 2. low-level readers
# ---------------------------------------------------------------------------------------------------------------
def _chunk_hook(d):
    """json object_hook: {"__chunked_string__": [...]} -> the joined string, anywhere in the value. The hook sees the
    innermost objects first, so a chunk wrapper is already a string when its parent is built."""
    if "__chunked_string__" in d:
        return "".join(d["__chunked_string__"])
    return d


def loads_flat(raw):
    """json.loads with chunked strings joined (about 20x faster than a recursive flatten over 50,000 turn files)."""
    return json.loads(raw, object_hook=_chunk_hook)


def read_json(path, flat: bool = False):
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    return loads_flat(raw) if flat else json.loads(raw)


def read_jsonl(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def sha1_hex(text: str | bytes) -> str:
    return hashlib.sha1(text if isinstance(text, bytes) else text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# 3. git: scratch bare repository per run, and the tree identities
# ---------------------------------------------------------------------------------------------------------------
# TREE IDENTITY, the exact rule. A tree (or a commit: its tree) is identified by the sha1 (first 12 hex digits) of the
# sorted lines "<path> <blob sha>" of EVERY non-directory entry of the tree (mode 40000 = directory, walked), leaving out each
# entry whose path matches EXC: any path component venv, env, .venv, artifacts, .codeevolver, .dspy_cache, node_modules or
# __pycache__ (the whole subtree), any *.pyc, and the top-level file codeevolver.md (the architecture description that a
# separate analysis agent rewrites after every iteration; it is not program code). File modes are not part of the identity.
# Two different commits can have the same identity (same files). Raw tree hashes are NOT an identity: the venv directory is
# committed and differs between nearly every pair of commits.
EXC = re.compile(r"(^|/)(venv|env|\.venv|artifacts|\.codeevolver|\.dspy_cache|node_modules|__pycache__)(/|$)|\.pyc$|^codeevolver\.md$")


def norm_py(src: bytes) -> str:
    """Normalised-AST hash of one Python file (10 hex digits): parse, drop docstrings of the module, classes and functions,
    dump the tree without field names and positions. Comments, blank lines, formatting and docstrings do not change it. A file
    that does not parse falls back to its raw bytes (prefix RAW:). It depends on the Python version (3.11 here)."""
    try:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = getattr(node, "body", None)
                if (body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    node.body = body[1:] or [ast.Pass()]
        return hashlib.sha1(ast.dump(tree, annotate_fields=False, include_attributes=False).encode("utf-8")).hexdigest()[:10]
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return "RAW:" + hashlib.sha1(src).hexdigest()[:10]


class Repo:
    """One persistent `git cat-file --batch` on a bare repository (a process start costs about 1 s on this machine)."""

    def __init__(self, path):
        self.path = str(path)
        self.p = subprocess.Popen(["git", "-C", self.path, "cat-file", "--batch"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.cache: dict[str, str] = {}      # tree sha -> sig
        self.ast_cache: dict[str, str] = {}  # tree sha -> ast sig
        self.tcache: dict[str, tuple] = {}   # object name -> (type, bytes, resolved sha)
        self.blob_ast: dict[str, str] = {}   # blob sha -> normalised AST hash

    def _read(self, obj: str):
        if obj in self.tcache:
            return self.tcache[obj]
        self.p.stdin.write((obj + "\n").encode())
        self.p.stdin.flush()
        hdr = self.p.stdout.readline().decode().strip()
        if hdr.endswith("missing") or hdr.endswith("ambiguous") or not hdr:
            self.tcache[obj] = (None, b"", None)
            return self.tcache[obj]
        sha, typ, size = hdr.split()
        data = self.p.stdout.read(int(size))
        self.p.stdout.read(1)
        self.tcache[obj] = (typ, data, sha)
        return self.tcache[obj]

    def has(self, obj: str) -> bool:
        return self._read(obj)[0] is not None

    def tree_of(self, obj: str):
        """Resolve a commit or tree object name to its tree sha (None when missing)."""
        r = self._read(obj + "^{tree}")
        return r[2] if r[0] == "tree" else None

    def commit_of(self, obj: str):
        r = self._read(obj + "^{commit}")
        return r[2] if r[0] == "commit" else None

    def _entries(self, tree: str):
        typ, data, _ = self._read(tree)
        i, out = 0, []
        while i < len(data):
            j = data.index(b" ", i)
            mode = data[i:j].decode()
            k = data.index(b"\0", j)
            out.append((mode, data[j + 1:k].decode("utf-8", "replace"), data[k + 1:k + 21].hex()))
            i = k + 21
        return out

    def _walk(self, tree: str, prefix: str, parts: list):
        for mode, name, sha in self._entries(tree):
            path = prefix + name
            if EXC.search(path):
                continue
            if mode == "40000":
                self._walk(sha, path + "/", parts)
            else:
                parts.append((path, sha))

    def files(self, obj: str) -> dict[str, str] | None:
        """{path: blob sha} of the identity-relevant files of a commit or tree (None when missing)."""
        t = self.tree_of(obj)
        if t is None:
            return None
        parts: list = []
        self._walk(t, "", parts)
        return dict(parts)

    def sig(self, obj: str):
        """File-content identity (rule above); None when the object is missing."""
        t = self.tree_of(obj)
        if t is None:
            return None
        if t not in self.cache:
            parts: list = []
            self._walk(t, "", parts)
            self.cache[t] = sha1_hex("\n".join(sorted(f"{p} {s}" for p, s in parts)))[:12]
        return self.cache[t]

    def ast_sig(self, obj: str):
        """Identity with every .py file replaced by its normalised-AST hash (other files by blob sha); same exclusions."""
        t = self.tree_of(obj)
        if t is None:
            return None
        if t not in self.ast_cache:
            parts: list = []
            self._walk(t, "", parts)
            lines = []
            for path, sha in parts:
                if path.endswith(".py"):
                    if sha not in self.blob_ast:
                        typ, data, _ = self._read(sha)
                        self.blob_ast[sha] = norm_py(data) if typ == "blob" else "MISSING"
                    lines.append(f"{path} {self.blob_ast[sha]}")
                else:
                    lines.append(f"{path} {sha}")
            self.ast_cache[t] = sha1_hex("\n".join(sorted(lines)))[:12]
        return self.ast_cache[t]

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def scratch_name(run: str) -> str:
    """Opaque folder name of a run's scratch repository (the run name must not appear in a path a judge could ever see)."""
    return sha1_hex("bundle|" + run)[:12] + ".git"


def git_run(args: list[str], cwd=None, check=True) -> subprocess.CompletedProcess:
    r = subprocess.run(["git"] + args, cwd=cwd, capture_output=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])} failed: {r.stderr.decode('utf-8', 'replace')[:300]}")
    return r


def fetch_bundle(bundle: Path, bare: Path) -> Path:
    """`git init --bare` + `git fetch <bundle> '+refs/codeevolver/*:refs/codeevolver/*'` into `bare` (idempotent: an existing
    repository is fetched into again; forced refs make the second fetch a no-op). Writes only inside `bare`."""
    bare = Path(bare)
    if not (bare / "HEAD").exists():
        bare.mkdir(parents=True, exist_ok=True)
        git_run(["init", "--bare", "-q", str(bare)])
    git_run(["-C", str(bare), "fetch", "-q", "--no-tags", str(bundle), "+refs/codeevolver/*:refs/codeevolver/*"])
    return bare


def list_refs(bare: Path) -> dict[str, str]:
    r = git_run(["-C", str(bare), "for-each-ref", "--format=%(refname) %(objectname)"])
    out = {}
    for line in r.stdout.decode().splitlines():
        if line.strip():
            name, sha = line.rsplit(" ", 1)
            out[name] = sha
    return out


# ---------------------------------------------------------------------------------------------------------------
# 4. the log parser
# ---------------------------------------------------------------------------------------------------------------
LOG_LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ \| (\w+) \| ([\w\.]+) \| (.*)$")
IT_MSG = re.compile(r"^iteration (\d+)(?: token usage by model)?: ?(.*)$")
RE_EVAL_DONE = re.compile(r"run_eval done \(id=(\w+), sample=(\w+), suffix=(\w+), primary_score=([-\d.eE]+|None), "
                          r"sha=(\w+), tree=(\w+) (first eval|changed since last eval|UNCHANGED since last eval)\)")
RE_FORK = re.compile(r"forked (\S+) from candidate iteration=(\d+) \(branch=(\S+), sha=(\w+)\)")
RE_PRESELECT = re.compile(r"pareto preselect — parent candidate=(\d+) \(exploit p=([\d.]+)\)")
RE_ADMIT = re.compile(r"pareto_evolution close — admitted=(\w+) \((\w+)\), comparison_state=(\w+), "
                      r"comparison_set_score=([-\d.eE]+|None), population=(\d+), best=candidate (\d+)")
RE_PROGRESS = re.compile(r"progress — evals (\d+)/(\d+) rows")
RE_CLOSED = re.compile(r"closed \(branch=(\S+), close_sha=(\w+), subsample baseline ([-\d.eE]+|None) -> best ([-\d.eE]+|None)")
RE_REVERT = re.compile(r"closing sha (\w+) scored ([-\d.eE]+) < best ([-\d.eE]+) \(sha (\w+)\); firing revert-check watchdog")
RE_CLOSE_CALL = re.compile(r"close_iteration \((head|loop) branch\) branch=(\S+) memory_updated=(\w+)")
RE_LOOP_DEBIT = re.compile(r"loop comparison eval debited (\d+) credit \(balance=(\d+)\)")
RE_GATE_PASS = re.compile(r"constraint gate: pass for (\w+)\.\.(\w+)")
RE_GATE_FAIL = re.compile(r"constraint gate: FAIL for (\w+)\.\.(\w+) — (.*)$")
RE_GATE_FIRED = re.compile(r"constraint gate fired \((.*)\) — force-reverting (\S+) to parent (\w+)")


def new_iter_log() -> dict:
    return dict(start_ts=None, end_ts=None, preselect_parent=None, exploit_p=None, exploit_trigger=False, fork_from=None,
                fork_branch=None, fork_sha8=None, evals=[], revert_check_fired=False, revert_check_text=None,
                revert_check_suppressed=False, close_skill_watchdog=False, close_missing_watchdog=False,
                close_calls=[], no_change_close=False, closed_by_fallback=None, gate_lines=[], gate_fired=None,
                admitted=None, admission_reason=None, comparison_state=None, comparison_set_score=None,
                final_post_budget=False, loop_debit=None, close_check_waived=0, progress=None, closed=None,
                autocommit_before_comp=0, autocommit_at_close=0, memory_diff_notice=False, variance_calls=0)


def parse_log(path) -> dict[int, dict]:
    """{iteration: facts of that iteration taken from the engine log}. Lines of other loggers that carry no 'iteration N:'
    (the constraint gate's pass/FAIL lines) belong to the iteration that started last."""
    out: dict[int, dict] = {}
    cur = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            m = LOG_LINE.match(raw.rstrip("\n"))
            if not m:
                continue
            ts, level, logger, msg = m.groups()
            mi = IT_MSG.match(msg)
            if mi:
                it, rest = int(mi.group(1)), mi.group(2)
                rec = out.setdefault(it, new_iter_log())
                rec["end_ts"] = ts
                if rest.startswith("starting"):
                    rec["start_ts"] = ts
                    cur = it
                _apply(rec, ts, level, rest)
            elif cur is not None:
                mg = RE_GATE_PASS.search(msg) or RE_GATE_FAIL.search(msg)
                if mg:
                    out[cur]["gate_lines"].append(msg)
    return out


def _apply(rec: dict, ts: str, level: str, rest: str) -> None:
    mm = RE_EVAL_DONE.search(rest)
    if mm:
        eid, sample, suffix, score, sha8, tree8, flag = mm.groups()
        rec["evals"].append(dict(eval_id=eid, sample=sample, suffix=suffix, score=None if score == "None" else float(score),
                                 sha8=sha8, tree8=tree8, flag=flag, ts=ts))
        return
    mm = RE_FORK.search(rest)
    if mm:
        rec["fork_branch"], rec["fork_from"], _, rec["fork_sha8"] = mm.group(1), int(mm.group(2)), mm.group(3), mm.group(4)
        return
    mm = RE_PRESELECT.search(rest)
    if mm:
        rec["preselect_parent"], rec["exploit_p"] = int(mm.group(1)), float(mm.group(2))
        return
    if "selecting best candidate due to exploit phase trigger" in rest:
        rec["exploit_trigger"] = True
        return
    mm = RE_ADMIT.search(rest)
    if mm:
        rec["admitted"] = mm.group(1) == "True"
        rec["admission_reason"], rec["comparison_state"] = mm.group(2), mm.group(3)
        rec["comparison_set_score"] = None if mm.group(4) == "None" else float(mm.group(4))
        return
    mm = RE_PROGRESS.search(rest)
    if mm:
        rec["progress"] = (int(mm.group(1)), int(mm.group(2)))
        return
    mm = RE_CLOSED.search(rest)
    if mm:
        rec["closed"] = dict(branch=mm.group(1), close_sha8=mm.group(2), no_change=False)
        return
    if rest.startswith("closed with no program change"):
        rec["no_change_close"] = True
        mf = re.search(r"closed_by_fallback=(\w+)", rest)
        rec["closed_by_fallback"] = (mf.group(1) == "True") if mf else None
        return
    mm = RE_REVERT.search(rest)
    if mm:
        rec["revert_check_fired"], rec["revert_check_text"] = True, rest
        return
    if "revert check suppressed" in rest:
        rec["revert_check_suppressed"] = True
        return
    if "firing close-skill watchdog" in rest:
        rec["close_skill_watchdog"] = True
        return
    if "firing close-missing watchdog" in rest:
        rec["close_missing_watchdog"] = True
        return
    mm = RE_CLOSE_CALL.search(rest)
    if mm:
        rec["close_calls"].append(dict(kind=mm.group(1), branch=mm.group(2), memory_updated=mm.group(3) == "True", ts=ts))
        return
    mm = RE_GATE_FIRED.search(rest)
    if mm:
        rec["gate_fired"] = mm.group(1)
        return
    mm = RE_LOOP_DEBIT.search(rest)
    if mm:
        rec["loop_debit"] = dict(credits=int(mm.group(1)), balance=int(mm.group(2)))
        return
    if "spent the run's final (post-budget) comparison eval" in rest:
        rec["final_post_budget"] = True
    elif "auto-committed dirty tree before comparison eval" in rest:
        rec["autocommit_before_comp"] += 1
    elif "auto-committed dirty tree at close" in rest:
        rec["autocommit_at_close"] += 1
    elif "run_variance_estimate drew" in rest:
        rec["variance_calls"] += 1


# ---------------------------------------------------------------------------------------------------------------
# 5. the transcript reader
# ---------------------------------------------------------------------------------------------------------------
NOTICE_OPENERS = (  # same openers as src/engine/optimizer/v3/architect_trace_markdown.py (_WATCHDOG_OPENERS)
    ("You are about to end iteration", "no-change"),
    ("The session is about to exit but you have not called", "close-missing"),
    ("You called the `close_iteration` tool", "close-skill"),
    ("Warning: the current git sha", "revert-check"),
    ("It appears that you did real work", "memory-diff"),
)


def strip_prompt_wrapper(text: str) -> str:
    """The harness quotes the prompt and prefixes the skill marker: '"/s\\n\\n<text>"'."""
    t = text.strip()
    if t.startswith('"'):
        t = t[1:]
    if t.endswith('"'):
        t = t[:-1]
    t = t.lstrip()
    if t.startswith("/s"):
        t = t[2:].lstrip()
    return t.strip()


def notice_type(text: str) -> str:
    t = strip_prompt_wrapper(text)
    for opener, label in NOTICE_OPENERS:
        if t.startswith(opener):
            return label
    return "other"


def load_turns(trace_dir) -> list[tuple[int, dict]]:
    """[(turn number, turn dict)] of one iteration, chunked strings joined. Turn numbers are those of the file names."""
    out = []
    for p in sorted(Path(trace_dir).glob("turn_*.json")):
        out.append((int(p.name[5:-5]), read_json(p, flat=True)))
    return out


def build_steps(turns: list[tuple[int, dict]]) -> list[dict]:
    """Chronological step list. kinds: prompt | notice | assistant | results | meta (the final harness `result` turn)."""
    names: dict = {}
    steps: list[dict] = []
    first_user = False
    for tn, t in turns:
        role = t.get("role")
        blocks = t.get("blocks") or []
        if role == "result":
            steps.append(dict(kind="meta", turn=tn))
            continue
        if role == "user":
            if blocks and all(b.get("kind") == "text" for b in blocks):
                text = "\n\n".join(str(b.get("text")) for b in blocks)
                if not first_user:
                    first_user = True
                    steps.append(dict(kind="prompt", turn=tn, text=text))
                else:
                    steps.append(dict(kind="notice", turn=tn, text=text, ntype=notice_type(text)))
                continue
            first_user = True
            results, extra = [], []
            for b in blocks:
                if b.get("kind") == "tool_result":
                    nm, inp, ctn, cidx = names.get(b.get("tool_use_id"), ("?", {}, None, None))
                    c = b.get("content")
                    if not isinstance(c, str):
                        c = json.dumps(c, ensure_ascii=False)
                    results.append(dict(id=b.get("tool_use_id"), name=nm, input=inp, call_turn=ctn, call_idx=cidx,
                                        is_error=bool(b.get("is_error")), truncated=bool(b.get("truncated")), content=c))
                elif b.get("kind") == "text":
                    extra.append(str(b.get("text")))
            steps.append(dict(kind="results", turn=tn, results=results, extra=extra))
        elif role == "assistant":
            texts, thinks, calls, patches = [], [], [], []
            for b in blocks:
                k = b.get("kind")
                if k == "text":
                    tx = str(b.get("text"))
                    if tx.startswith("[patch] "):
                        patches.append(tx)  # synthetic line added by the harness after ce_coding_agent
                    else:
                        texts.append(tx)
                elif k == "thinking":
                    thinks.append(str(b.get("text")))
                elif k == "tool_use":
                    inp = b.get("input") or {}
                    idx = len(calls)
                    names[b.get("id")] = (b.get("name"), inp, tn, idx)
                    calls.append(dict(id=b.get("id"), name=b.get("name"), input=inp, turn=tn, idx=idx))
            steps.append(dict(kind="assistant", turn=tn, texts=texts, thinks=thinks, calls=calls, patches=patches))
    return steps


def results_by_id(steps: list[dict]) -> dict:
    d = {}
    for s in steps:
        if s["kind"] == "results":
            for r in s["results"]:
                d[r["id"]] = dict(r, result_turn=s["turn"])
    return d


def all_calls(steps: list[dict]) -> list[dict]:
    return [c for s in steps if s["kind"] == "assistant" for c in s["calls"]]


def parse_json_obj(text):
    """A tool result that is a JSON object -> dict, else None."""
    try:
        p = json.loads(text)
        return p if isinstance(p, dict) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------------------------
# 6. memory.md: classification and replay
# ---------------------------------------------------------------------------------------------------------------
MEMBASE = "memory.md"
READ_VERBS = re.compile(r"\b(cat|head|tail|less|more|grep|egrep|awk|sed\s+-n|nl|bat|python3?|diff|cp)\b")


def is_memfile(path) -> bool:
    p = (path or "").replace("\\", "/")
    return bool(p) and os.path.basename(p) == MEMBASE and "/memory/" in p


def parse_read(res):
    """opencode read result -> text (line numbers stripped)."""
    if res is None:
        return None
    m = re.search(r"<content>\n(.*?)(?:\n\n?\(End of file|\n\n?\(Showing|\n</content>|</content>|$)", res, re.S)
    body = m.group(1) if m else res
    lines = []
    for ln in body.split("\n"):
        mm = re.match(r"^(\d+): ?(.*)$", ln)
        if mm:
            lines.append(mm.group(2))
    return "\n".join(lines)


def classify(x: dict):
    """Classify a tool item that touches memory.md: 'read' | 'write_overwrite' | 'append' | 'edit' | 'sed_inplace' |
    'other_file_write' | None. x = {name, input}."""
    n, i = x["name"], x["input"]
    fp = i.get("filePath") or i.get("path") or ""
    if n == "read":
        return "read" if is_memfile(fp) else None
    if n == "grep":
        return "read" if "memory.md" in json.dumps(i) else None
    if n == "write":
        return "write_overwrite" if is_memfile(fp) else "other_file_write"
    if n == "edit":
        return "edit" if is_memfile(fp) else "other_file_write"
    if n == "bash":
        c = i.get("command", "")
        m = re.search(r"(>>?)\s*(\S*/)?memory\.md\b", c)
        if m:  # 'cat memory.md > other' is not a write onto memory.md: the target follows '>'
            return "append" if m.group(1) == ">>" else "write_overwrite"
        if re.search(r"tee\s+(-a\s+)?\S*memory\.md", c):
            return "append" if "tee -a" in c else "write_overwrite"
        if re.search(r"sed\s+-i", c) and "memory.md" in c and "old_memory" not in c.split("sed -i")[0][-0:]:
            return "sed_inplace"
        if re.search(r"\b(mv|cp)\b[^|;&]*\S*memory\.md\s*($|[;&|])", c) and "git" not in c.split("mv")[0][-5:]:
            return "write_overwrite"
        if "memory.md" in c and READ_VERBS.search(c) and not re.search(r"git\s+(add|commit)", c):
            return "read"
        return None
    return None


def long_lines(s, minlen: int = 25) -> set:
    """Set of stripped lines of at least `minlen` characters (the 'long lines' of the memory metrics)."""
    return {ln.strip() for ln in (s or "").split("\n") if len(ln.strip()) >= minlen}


def norm_line(ln: str) -> str:
    return " ".join(ln.split())


def long_lines_norm(s, minlen: int = 25) -> set:
    """Whitespace-normalised, de-duplicated long lines (length measured on the stripped line)."""
    return {norm_line(ln) for ln in (s or "").split("\n") if len(ln.strip()) >= minlen}


def rstrip_text(s) -> str:
    return (s or "").rstrip()


def tolerant_replace(state: str, old: str, nw: str):
    """Line-trimmed match (opencode's edit tool tolerates whitespace differences)."""
    NL = chr(10)
    sl = state.split(NL)
    ol = old.strip(NL).split(NL)
    st = [ln.strip() for ln in sl]
    ot = [ln.strip() for ln in ol]
    n = len(ot)
    for a in range(0, len(sl) - n + 1):
        if st[a:a + n] == ot:
            return NL.join(sl[:a] + nw.strip(NL).split(NL) + sl[a + n:])
    return None


def mem_items(steps: list[dict]) -> list[dict]:
    """The tool calls of an iteration that touch memory.md or the memory folder, as replay items in call order:
    {name, input, result, is_error, turn, mem}. `mem` = the call text names memory.md, /memory/ or old_memory."""
    res_by = results_by_id(steps)
    items = []
    for c in all_calls(steps):
        s = json.dumps(c["input"], ensure_ascii=False)
        mem = "memory.md" in s or "/memory/" in s or "old_memory" in s
        r = res_by.get(c["id"])
        items.append(dict(type="tool", turn=c["turn"], name=c["name"], id=c["id"], input=c["input"], mem=mem,
                          result=r["content"] if r else None, is_error=bool(r and r["is_error"]),
                          rturn=r["result_turn"] if r else None))
    return items


def replay_iteration(start, items: list[dict]):
    """Replay the memory edits of one iteration over `start`. -> (events, end state). Event fields: kind, turn, name, before,
    after, read_before (a memory read call came earlier, an errored one included: legacy rule), ok (the replay could apply it)."""
    state = start if start is not None else ""
    out, read_seen = [], False
    for x in items:
        if x["type"] != "tool" or not x["mem"]:
            continue
        k = classify(x)
        if k is None or k == "other_file_write":
            continue
        i = x["input"]
        res = x.get("result") or ""
        ev = dict(kind=k, turn=x["turn"], name=x["name"], before=state, after=state, read_before=read_seen, ok=True,
                  err=bool(x.get("is_error")))
        if k == "read":
            read_seen = True    # ANY read attempt counts, an errored one too (t2mem_lib rule that the legacy 22/144 used)
            ev["view"] = parse_read(res) if x["name"] == "read" else None
            ev["partial"] = x["name"] == "read" and ("offset" in i or "limit" in i)
            ev["complete"] = x["name"] == "read" and (not ev["partial"]) and "(End of file" in res
            if ev["complete"] and ev["view"] is not None and rstrip_text(ev["view"]) != rstrip_text(state):
                ev["resynced"] = True
                state = ev["view"]
                ev["before"] = state
            ev["after"] = state
        elif k == "write_overwrite":
            if x["name"] == "write":
                new = i.get("content", "")
            else:
                m = re.match(r"^\s*(?:cd [^&]*&&\s*)?echo\s+(\"|')(.*)\1\s*>\s*\S*memory\.md\s*$", i.get("command", ""), re.S)
                new = (m.group(2) + "\n") if m else None
            ev["ok"] = (not ev["err"]) and new is not None
            if ev["ok"]:
                state = new
            ev["after"] = state
        elif k == "append":
            c = i.get("command", "")
            m = re.search(r"echo\s+(?:-e\s+)?(\"|')(.*?)\1\s*>>", c, re.S)
            h = re.search(r"<<\s*'?(\w+)'?\n(.*?)\n\1", c, re.S)
            txt = m.group(2) if m else (h.group(2) + "\n" if h else None)
            ev["ok"] = (not ev["err"]) and txt is not None
            if ev["ok"]:
                state = state + ("" if state.endswith("\n") or not state else "\n") + txt + ("" if txt.endswith("\n") else "\n")
            ev["after"] = state
        elif k == "edit":
            old, nw = i.get("oldString", ""), i.get("newString", "")
            ok = (not ev["err"]) and "applied" in res.lower()
            if ok and old and old in state:
                state = state.replace(old, nw) if i.get("replaceAll") else state.replace(old, nw, 1)
            elif ok and old and tolerant_replace(state, old, nw) is not None:
                state = tolerant_replace(state, old, nw)
            elif ok and old == "":
                state = state + nw
            else:
                ev["ok"] = False
            ev["after"] = state
        elif k == "sed_inplace":
            c = i.get("command", "")
            for mm in re.finditer(r"sed -i\s+'(\d+),(\d+)d'", c):
                a, b = int(mm.group(1)), int(mm.group(2))
                ls = state.split("\n")
                state = "\n".join(ls[:a - 1] + ls[b:])
            for mm in re.finditer(r"sed -i\s+'s/([^/]*)/([^/]*)/g?'", c):
                state = state.replace(mm.group(1), mm.group(2))
            ev["after"] = state
        out.append(ev)
    return out, state


# ---------------------------------------------------------------------------------------------------------------
# 7. one run
# ---------------------------------------------------------------------------------------------------------------
class RunData:
    """Everything of one core run, loaded lazily and cached. `runs_root` is a folder with one subfolder per run."""

    def __init__(self, runs_root, spec: RunSpec, private=None):
        self.spec = spec
        self.root = Path(runs_root) / spec.run
        self.private = Path(private) if private else PRIVATE_DEFAULT
        self._c: dict = {}
        self._repo = None

    # ---- identity ------------------------------------------------------------------------------------------
    @property
    def run(self) -> str:
        return self.spec.run

    @property
    def history(self) -> dict:
        if "history" not in self._c:
            self._c["history"] = read_json(self.root / "history.json")
        return self._c["history"]

    @property
    def job(self) -> str:
        return self.history["job_id"]

    @property
    def hdir(self) -> Path:
        return self.root / ".codeevolver" / "history" / self.job

    @property
    def iterations(self) -> list[int]:
        """Iteration numbers with a record in history.json, ascending (0 = the seed program)."""
        ih = self.history["iteration_history"]
        return sorted(int(k) for k in ih if str(k).isdigit())

    def rec(self, it: int) -> dict:
        return self.history["iteration_history"][str(it)]

    def trace_dir(self, it: int) -> Path:
        return self.root / ".codeevolver" / "architect_traces" / self.job / f"iteration_{it}_architect_trace"

    # ---- logs and ledgers of the harness ---------------------------------------------------------------------
    @property
    def log(self) -> dict[int, dict]:
        if "log" not in self._c:
            p = self.root / ".codeevolver" / "logs" / f"{self.job}.log"
            self._c["log"] = parse_log(p) if p.exists() else {}
        return self._c["log"]

    @property
    def evals(self) -> list[dict]:
        """Every logged eval (iteration_N_evals.jsonl of all iterations), in time order. Each record keeps its own
        `iteration` field (a harness-run seed comparison eval has iteration 0)."""
        if "evals" not in self._c:
            out = []
            for p in sorted(self.hdir.glob("iteration_*_evals.jsonl")):
                out.extend(read_jsonl(p))
            out.sort(key=lambda r: (r.get("ts") or 0))
            self._c["evals"] = out
        return self._c["evals"]

    def evals_of(self, it: int) -> list[dict]:
        return [e for e in self.evals if e.get("iteration") == it]

    @property
    def comps(self) -> list[dict]:
        """comparison_set_history.jsonl without the per-row scores."""
        if "comps" not in self._c:
            out = []
            for d in read_jsonl(self.hdir / "comparison_set_history.jsonl"):
                d.pop("row_scores", None)
                out.append(d)
            self._c["comps"] = out
        return self._c["comps"]

    @property
    def snaps(self) -> dict[int, str]:
        """{k: memory.md text at the END of iteration k} from history/<job>.jsonl (later lines overwrite earlier ones;
        empty or missing texts are skipped)."""
        if "snaps" not in self._c:
            snaps: dict[int, str] = {}
            for d in read_jsonl(self.root / ".codeevolver" / "history" / f"{self.job}.jsonl"):
                p = d.get("progress") or {}
                k, mm = p.get("current_iteration"), p.get("memory_md")
                if k is None or mm is None or mm == "":
                    continue
                snaps[int(k)] = mm
            self._c["snaps"] = snaps
        return self._c["snaps"]

    @property
    def budget(self) -> dict:
        """The budget facts of the run from settings.json (only these fields are kept: max_metric_calls, subsample_size,
        comparison_set_size, variance_estimate_samples, credits_initial, credits_per_iteration)."""
        if "budget" not in self._c:
            s = read_json(self.root / "settings.json")
            b = dict(s.get("budget") or {})
            ov = {}
            for o in s.get("overrides") or []:
                if isinstance(o, str) and "=" in o:
                    k, v = o.split("=", 1)
                    ov[k.strip()] = v.strip()
            self._c["budget"] = dict(
                max_metric_calls=int(b.get("max_metric_calls")), subsample_size=int(b.get("subsample_size")),
                comparison_set_size=int(b.get("comparison_set_size")),
                variance_estimate_samples=int(b.get("variance_estimate_samples")),
                credits_initial=int(ov.get("comparison_credits_initial", 1)),
                credits_per_iteration=int(ov.get("comparison_credits_per_iteration", 1)))
        return self._c["budget"]

    def header(self, it: int) -> dict:
        p = self.trace_dir(it) / "header.json"
        return read_json(p) if p.exists() else {}

    def steps(self, it: int) -> list[dict]:
        key = ("steps", it)
        if key not in self._c:
            self._c[key] = build_steps(load_turns(self.trace_dir(it)))
        return self._c[key]

    def drop_steps(self, it: int) -> None:
        self._c.pop(("steps", it), None)

    # ---- git -----------------------------------------------------------------------------------------------------
    @property
    def bare(self) -> Path:
        return self.private / "git" / scratch_name(self.run)

    def ensure_repo(self) -> Repo:
        """Fetch the bundle into the scratch bare repository (once) and open a Repo on it."""
        if self._repo is None:
            fetch_bundle(self.root / "repo.bundle", self.bare)
            self._repo = Repo(self.bare)
        return self._repo

    @property
    def repo(self) -> Repo:
        return self.ensure_repo()

    def close(self) -> None:
        if self._repo is not None:
            self._repo.close()
            self._repo = None
