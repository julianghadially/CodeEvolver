"""Copyright © 2026 440 Labs LLC

Entry script for the simple loop, exec'd inside the sandbox.

The filename must stay `optimization_entry.py`: the research runner detects a
live optimizer by that substring in the process command line.
"""

from __future__ import annotations

from src.engine.optimizer.v3._entry import run_entry


def main() -> None:
    from src.engine.optimizer.simple_loop.simple_loop_optimizer import (
        run_simple_loop_optimization,
    )

    run_entry(run_simple_loop_optimization, method="simple_loop")


if __name__ == "__main__":
    main()
