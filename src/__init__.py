"""CodeEvolver: agentic optimization of AI system code."""
import os

# When set, merge the client repo's src/ into this package's __path__ so
# `src.<client_pkg>` resolves alongside `src.engine`.
#
# CODEEVOLVER_WORKSPACE_SRC: literal-string read of a cross-process env
# var written by src/engine/services/subprocess_service.py::_build_env.
# Registry of cross-process env-var names lives in
# src/engine/paths.py::CROSS_PROCESS_ENV_VARS.
_workspace_src = os.environ.get("CODEEVOLVER_WORKSPACE_SRC")
if _workspace_src and os.path.isdir(_workspace_src) and _workspace_src not in __path__:
    __path__.append(_workspace_src)
