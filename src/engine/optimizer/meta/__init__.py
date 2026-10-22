"""Copyright © 2026 440 Labs LLC

Post-run analysis agents shared by every optimizer arm.

These are read-only auditors that run on the artifacts of a finished run
(iteration history, architect traces, the final workspace). They never feed
back into optimization: `meta_reflection` proposes engine improvements,
`run_integrity` classifies cheating and run health, and
`simple_meta_reflection` is the simple-loop variant of meta-reflection.
"""
