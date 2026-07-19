"""PLAN.md P0.E: persistent trainer counter state for checkpoint/resume.

The base ``RayPPOTrainer._load_checkpoint`` restores only ``global_steps``
(parsed from the ``global_step_{n}`` folder name). VDRA additionally needs
``rollout_iteration`` — replay ages are ``rollout_iteration -
generation_rollout_iteration``, so resuming with ``rollout_iteration = 0``
while restored edges carry high generation iterations produces negative
ages and edges that never expire.

This module is deliberately engine-free (no verl / torch imports) so the
state contract is unit-testable on CPU without the trainer stack.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

TRAINER_STATE_FILENAME = "gear_tree_trainer_state.json"


@dataclass
class GearTreeTrainerState:
    """Counters that must survive checkpoint/resume exactly (PLAN.md P0.E)."""

    global_step: int = 0
    rollout_iteration: int = 0
    num_optimizer_steps_total: int = 0
    successful_actor_updates: int = 0
    postponed_updates: int = 0
    failed_updates: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def trainer_state_path(checkpoint_dir: str | Path) -> Path:
    return Path(checkpoint_dir) / TRAINER_STATE_FILENAME


def save_trainer_state(
    checkpoint_dir: str | Path, state: GearTreeTrainerState
) -> Path:
    """Write the counter state atomically into the checkpoint directory."""
    path = trainer_state_path(checkpoint_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(state.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def initial_next_threshold(global_step: int, freq: int) -> Optional[int]:
    """PLAN.md P0.E: first save/eval threshold strictly above ``global_step``.

    Derived (not persisted) so resume at ``global_step=400`` with
    ``freq=10`` waits for 410, never re-fires 400.
    """
    freq = int(freq or 0)
    if freq <= 0:
        return None
    return (int(global_step) // freq + 1) * freq


def advance_past_thresholds(
    *,
    previous_step: int,
    current_step: int,
    next_threshold: Optional[int],
    freq: int,
) -> tuple[int, Optional[int]]:
    """PLAN.md P0.E: crossed-threshold trigger semantics.

    ``global_step`` advances by the actual optimizer-step count (e.g.
    ``8 -> 12``), so a modulo check misses thresholds inside the jump. This
    returns ``(crossed_count, new_next_threshold)`` where ``crossed_count``
    is how many thresholds lie in ``(previous_step, current_step]`` — the
    caller fires its action when the count is positive and the counter has
    already advanced past every crossed threshold.
    """
    freq = int(freq or 0)
    if freq <= 0 or next_threshold is None:
        return 0, next_threshold
    crossed = 0
    threshold = int(next_threshold)
    while int(previous_step) < threshold <= int(current_step):
        crossed += 1
        threshold += freq
    return crossed, threshold


def load_trainer_state(
    checkpoint_dir: str | Path,
) -> Optional[GearTreeTrainerState]:
    """Read the counter state from a checkpoint directory.

    Returns ``None`` for a legacy checkpoint that predates the state file —
    the caller must then choose an explicit safe behavior (PLAN.md P0.E:
    reset replay, never continue silently with negative ages).
    """
    path = trainer_state_path(checkpoint_dir)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    known = {f for f in GearTreeTrainerState.__dataclass_fields__}
    cleaned = {k: int(v) for k, v in data.items() if k in known}
    return GearTreeTrainerState(**cleaned)
