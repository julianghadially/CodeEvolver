"""GEPA-specific client sandbox implementation.

Extends ClientSandbox with exec_prebuilt() for DSPy command execution.
"""

import base64
import json
import logging

from ..core.client_sandbox import ClientSandbox

logger = logging.getLogger(__name__)


class GEPASandbox(ClientSandbox):
    """Modal Sandbox for GEPA DSPy program evaluation.

    Inherits generic sandbox lifecycle from ClientSandbox and implements
    exec_prebuilt() to execute DSPy-specific commands via master.py.

    Usage:
        sandbox = GEPASandbox(app, repo_url, ...)
        sandbox.start()
        try:
            result = sandbox.exec_prebuilt({"command": "evaluate", ...})
        finally:
            sandbox.stop()
    """

    def exec_prebuilt(self, command: dict) -> dict:
        """Execute a prebuilt DSPy command via master.py.

        Writes the command to a temp file, executes master.py,
        and parses the EVAL_RESULT: prefix from stdout.

        Args:
            command: Dict with at minimum a "command" key.
                Supported commands: build_seed_candidate, evaluate, make_reflective_dataset

        Returns:
            Parsed JSON result dict from the handler.

        Raises:
            RuntimeError: If the sandbox is not started.
        """
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started. Call start() first.")

        # Base64-encode the command JSON for safe file transfer
        cmd_json = json.dumps(command)
        cmd_b64 = base64.b64encode(cmd_json.encode()).decode()

        # Write command file via bash (base64 decode for safety)
        write_p = self._sandbox.exec(
            "bash", "-c",
            f"echo '{cmd_b64}' | base64 -d > /tmp/prebuilt_command.json",
        )
        write_p.wait()

        # Execute master.py dispatcher (source .env first to load environment variables)
        p = self._sandbox.exec(
            "bash", "-c",
            f"set -a && source {self._workspace}/.env 2>/dev/null; set +a; "
            f"PYTHONPATH=/app:$PYTHONPATH python /app/sandbox_scripts/master.py "
            f"--workspace {self._workspace} "
            f"--command-file /tmp/prebuilt_command.json",
        )
        p.wait()

        stdout = p.stdout.read()
        stderr = p.stderr.read()

        # Print stderr which contains our debug output (print shows in Modal logs)
        if stderr:
            print(f"[SANDBOX STDERR]\n{stderr}", flush=True)

        if p.returncode != 0:
            print(f"[SANDBOX ERROR] master.py failed (rc={p.returncode}): {stderr}", flush=True)
            return {
                "success": False,
                "error": f"Prebuilt script exit code {p.returncode}: {stderr[:2000]}",
            }

        # Parse EVAL_RESULT: from stdout
        for line in stdout.split("\n"):
            if line.startswith("EVAL_RESULT:"):
                payload = line[len("EVAL_RESULT:"):]
                try:
                    return json.loads(payload)
                except json.JSONDecodeError as e:
                    return {
                        "success": False,
                        "error": f"Failed to parse script output: {e}",
                    }

        # No EVAL_RESULT found
        logger.error(f"No EVAL_RESULT in stdout. stdout={stdout[:2000]}, stderr={stderr[:2000]}")
        return {
            "success": False,
            "error": f"Prebuilt script produced no EVAL_RESULT. stderr: {stderr[:2000]}",
        }
