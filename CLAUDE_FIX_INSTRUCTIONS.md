# Claude Fix Instructions — Difficulty Index

This file is now an index only. The old single long guide is obsolete because it mixed local fixes with architecture-level refactors and made completion claims hard to verify.

Read `PLAN.md` first, then execute these files in order:

```text
1. CLAUDE_FIX_EASY.md
2. CLAUDE_FIX_MEDIUM.md
3. CLAUDE_FIX_HARD.md
```

## Easy

Use `CLAUDE_FIX_EASY.md` for small, local fixes:

```text
replay validation order
missing-advantage handling
cross-worker optimizer-step agreement
pre-commit step verification
monotonic manifest accounting failures
stale comments/config claims
```

These tasks should be implemented as focused commits and must not redesign scheduler or optimizer ownership.

## Medium

Use `CLAUDE_FIX_MEDIUM.md` for cross-function production changes:

```text
reserved-update transaction lifecycle
strict tree_instance_id and derived edge_id
canonical manifest invariant cleanup
real Hydra/dataclass schema validation
```

These tasks require failure-injection tests on production helpers.

## Hard

Use `CLAUDE_FIX_HARD.md` for architecture-level refactors:

```text
total training-step unit
global successful optimizer-step semantics
scheduler placement
non-finite did_step accounting
per-optimizer-batch zero-signal skip
FSDP/FSDP2 production parity
post-refactor checkpoint and interval semantics
```

Do not split the step-unit/scheduler refactor into unrelated one-line patches.

## Completion rule

A task is not complete merely because:

```text
a commit has the task name
a helper unit test passes
a source-string guard finds the expected function name
a config file contains the intended value
```

A task is complete only when its production success path and failure path satisfy `PLAN.md` and the corresponding difficulty guide.
