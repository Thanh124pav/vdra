"""H1 (CODEX_FIX_HARD.md): REAL FSDP2 distributed semantics, measured.

Two REAL ``torch.distributed`` processes (gloo, CPU) wrap the shared TinyLM
with the ACTUAL production ``verl.utils.fsdp_utils.apply_fsdp2``
(``fully_shard``) and drive the ACTUAL ``DataParallelPPOActor.update_policy``
on 512 canonical replay edges tensorized by the real ``edges_to_dataproto``,
dispatched exactly like production (contiguous ``DataProto.chunk`` across
ranks, ``ppo_mini_batch_size`` divided by world size as in
``fsdp_workers.py``). Rank 0 additionally runs a single-rank REFERENCE
update whose k-th optimizer batch is the UNION of both ranks' k-th local
mini-batches, so every comparison is row-aligned with what the distributed
run jointly processed per step.

Measured cells (loss modes are the CURRENT production paths):

* ``segment_mean``  — the ``batch_slot_mean_ablation`` path
  ``L = sum(rows)/N_B`` with pre-filter ``N_B = len(mini_batch)``. With
  ``segment_token_reduction=mean`` this IS the paper objective
  ``L = (1/M) * sum_u [token-mean of segment u]``. Expected and asserted:
  exact distributed parity (local ``M_local = M/W`` holds by construction).
* ``tree_balanced`` — the currently-canonical ``w = 1/(N_T * N_seg)`` path,
  where dp_actor computes ``N_T`` from the LOCAL rank mini-batch. Expected
  and asserted: parity ONLY when every rank sees ``N_T_union/W`` trees;
  measured mismatch (collapse to the uniform segment mean, or an exact
  halving) otherwise.

HONESTY POLICY: the mismatch tests assert the MEASURED, algebraically
predicted behavior of production code — a green suite means our model of the
distributed semantics is confirmed, NOT that the semantics are acceptable.
The acceptability question is escalated in ``docs/h1_fsdp_parity_report.md``
per the CODEX_FIX_HARD.md approval gate. No xfail is used to hide findings.

FSDP1 (``strategy="fsdp"``) cannot run on CPU under torch 2.11 at all; a
dedicated test documents that limitation loudly instead of silently skipping
(FSDP1 parity evidence therefore remains a GPU-smoke item).
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

try:  # namespace-package import under PYTHONPATH=verl
    from recipe.gear_tree.tests import _test_shims
except ImportError:  # flat import when mp.spawn re-imports this module
    import _test_shims

_test_shims.install()

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")

import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
from torch.distributed.tensor import DTensor  # noqa: E402

try:
    from recipe.gear_tree.tests import _tiny_actor
except ImportError:
    import _tiny_actor

WORLD = 2
N_ROWS = 512
GLOBAL_MINI = 128
LOCAL_MINI = GLOBAL_MINI // WORLD  # fsdp_workers.py divides by dp world size
LOCAL_MICRO = 32
GLOBAL_MICRO = LOCAL_MICRO * WORLD
N_STEPS = N_ROWS // GLOBAL_MINI  # 4 optimizer steps
GRAD_CLIP = 1e9  # no-op clipping keeps SGD param delta == -lr * grad
LR = 0.05
# Same-shape identities (reducer average, collapse) are exact in fp32 because
# both sides run bitwise-identical 32-row micro-batch kernels.
ATOL = 1e-6
# Comparisons against the single-rank reference cross micro-batch SHAPES
# (local 32-row vs reference 64-row kernels): CPU fp32 GEMM tiling and
# summation order differ per shape, producing ~1.5e-6 absolute noise on
# gradients of magnitude ~1e-4..1e-3 (measured). 5e-6 bounds that noise
# while still failing hard on any semantic deviation: the uneven-tree
# mismatch shifts gradients by ~50% of their own norm, orders of magnitude
# above this tolerance.
REF_ATOL = 5e-6

# (scenario, loss_mode, segment_token_reduction)
CELLS = [
    ("balanced", "segment_mean", "mean"),
    ("balanced", "segment_mean", "sum"),
    ("uneven", "segment_mean", "mean"),
    ("interleaved", "segment_mean", "mean"),
    ("token_skew", "segment_mean", "mean"),
    ("balanced", "tree_balanced", "mean"),
    ("uneven", "tree_balanced", "mean"),
    ("interleaved", "tree_balanced", "mean"),
]


def _scenario_edges(name: str) -> list[dict]:
    if name == "balanced":
        # 64 trees x 8 segments, contiguous: each 64-row local mini-batch
        # holds exactly 8 whole trees -> W * N_T_local == N_T_union == 16.
        return _tiny_actor.make_edges([8] * 64)
    if name == "uneven":
        # Contiguous: rank 0 gets only 4-segment trees (N_T_local = 16 per
        # 64-row mini-batch), rank 1 only 32-segment trees (N_T_local = 2);
        # the union optimizer batch has 18 trees.
        return _tiny_actor.make_edges([4] * 64 + [32] * 8)
    if name == "interleaved":
        # Round-robin rows: every 64-row local mini-batch touches all 64
        # trees, so N_T_local == N_T_union == 64 on BOTH ranks.
        return _tiny_actor.make_edges([8] * 64, order="interleaved")
    if name == "token_skew":
        # Balanced trees but rank 0 rows have 1 valid token and rank 1 rows
        # have MAX_RESPONSE: probes token-count asymmetry across ranks.
        return _tiny_actor.make_edges(
            [8] * 64,
            resp_len_for=lambda t, j: 1 if t < 32 else _tiny_actor.MAX_RESPONSE,
        )
    raise ValueError(f"unknown scenario {name!r}")


def _cell_file(cell) -> str:
    return "-".join(cell) + ".pt"


def _full(t: torch.Tensor) -> torch.Tensor:
    if isinstance(t, DTensor):
        t = t.full_tensor()
    return t


def _named_full_params(model) -> dict[str, torch.Tensor]:
    return {n: _full(p.detach()).clone() for n, p in model.named_parameters()}


def _cell_config(mode: str, reduction: str, *, mini: int, micro: int):
    return _tiny_actor.make_actor_config(
        strategy="fsdp2" if mini == LOCAL_MINI else "fsdp",
        mini=mini,
        micro=micro,
        reduction=reduction,
        batch_slot_ablation=(mode == "segment_mean"),
        grad_clip=GRAD_CLIP,
    )


def _run_update_recording(actor, model, batch):
    """Run REAL update_policy, recording full pre-clip grads per step."""
    step_grads: list[dict[str, torch.Tensor]] = []
    real_step = actor._optimizer_step

    def _recording_step():
        step_grads.append(
            {n: _full(p.grad).clone() for n, p in model.named_parameters()}
        )
        return real_step()

    actor._optimizer_step = _recording_step
    p0 = _named_full_params(model)
    metrics = actor.update_policy(batch)
    p1 = _named_full_params(model)
    delta = {n: p1[n] - p0[n] for n in p0}
    return step_grads, p0, delta, metrics


def _reference_run(full, mode: str, reduction: str):
    """Single-rank reference on rank 0: PLAIN TinyLM, union-aligned batches."""
    from verl.protocol import DataProto

    chunks = full.chunk(WORLD)
    minis = [c.split(LOCAL_MINI) for c in chunks]
    ordered = []
    for k in range(N_STEPS):
        for r in range(WORLD):
            ordered.append(minis[r][k])
    ref_batch = DataProto.concat(ordered)
    ref_batch.meta_info.update(full.meta_info)

    model = _tiny_actor.TinyLM()
    actor, model, _ = _tiny_actor.make_actor(
        model=model,
        config=_cell_config(mode, reduction, mini=GLOBAL_MINI, micro=GLOBAL_MICRO),
        lr=LR,
    )
    step_grads, p0, delta, metrics = _run_update_recording(actor, model, ref_batch)
    return {
        "step_grads": step_grads,
        "p0": p0,
        "param_delta": delta,
        "pg_loss": metrics["actor/pg_loss"],
        "grad_norm": metrics["actor/grad_norm"],
        "num_steps": metrics["actor/num_optimizer_steps"],
    }


def _step0_prediction(full, mode: str, reduction: str) -> dict[str, torch.Tensor]:
    """Average of both ranks' LOCAL step-0 grads (fresh identical models).

    Each rank's first 64-row local mini-batch is fed to a PLAIN single-rank
    actor as its own optimizer batch, so the loss uses exactly the LOCAL
    denominators dp_actor would use. Averaging the two gradients predicts
    the distributed step-0 gradient IF the FSDP2 reducer averages.
    """
    grads: list[dict[str, torch.Tensor]] = []
    for r in range(WORLD):
        local_mini0 = full.chunk(WORLD)[r].split(LOCAL_MINI)[0]
        model = _tiny_actor.TinyLM()
        actor, model, _ = _tiny_actor.make_actor(
            model=model,
            config=_cell_config(mode, reduction, mini=LOCAL_MINI, micro=LOCAL_MICRO),
            lr=LR,
        )
        # strategy field is "fsdp" here but the module is a plain nn.Module,
        # so _optimizer_step takes the plain clip path (no collectives).
        step_grads, _, _, _ = _run_update_recording(actor, model, local_mini0)
        grads.append(step_grads[0])
    return {n: (grads[0][n] + grads[1][n]) / WORLD for n in grads[0]}


def _run_cell(rank, mesh, scenario, mode, reduction, out_dir):
    from verl.utils.fsdp_utils import apply_fsdp2

    full = _tiny_actor.build_batch(_scenario_edges(scenario))

    model = _tiny_actor.TinyLM()
    apply_fsdp2(model, {"mesh": mesh}, {})
    actor, model, _ = _tiny_actor.make_actor(
        model=model,
        config=_cell_config(mode, reduction, mini=LOCAL_MINI, micro=LOCAL_MICRO),
        lr=LR,
    )

    local = full.chunk(WORLD)[rank]  # contiguous production dispatch
    step_grads, p0, delta, metrics = _run_update_recording(actor, model, local)

    local_report = {
        "pg_loss": metrics["actor/pg_loss"],
        "grad_norm": metrics["actor/grad_norm"],
        "num_steps": metrics["actor/num_optimizer_steps"],
    }
    gathered: list = [None] * WORLD
    dist.all_gather_object(gathered, local_report)

    if rank == 0:
        payload = {
            "dist": {
                "step_grads": step_grads,
                "p0": p0,
                "param_delta": delta,
                "per_rank": gathered,
            },
            "ref": _reference_run(full, mode, reduction),
            "pred0": _step0_prediction(full, mode, reduction),
        }
        torch.save(payload, os.path.join(out_dir, _cell_file((scenario, mode, reduction))))
    dist.barrier()


def _worker(rank: int, world: int, rdv_file: str, out_dir: str):
    _test_shims.install()
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rdv_file}",
        rank=rank,
        world_size=world,
    )
    try:
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh("cpu", (world,))
        for scenario, mode, reduction in CELLS:
            _run_cell(rank, mesh, scenario, mode, reduction, out_dir)
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def measurements(tmp_path_factory):
    out = tmp_path_factory.mktemp("h1")
    rdv = out / "rdv"
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    # Children must see no CUDA so get_device_id() returns "cpu" and micro
    # batches stay on the CPU device mesh.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        mp.spawn(_worker, args=(WORLD, str(rdv), str(out)), nprocs=WORLD, join=True)
    finally:
        if saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved
    return {
        cell: torch.load(str(out / _cell_file(cell)), weights_only=False)
        for cell in CELLS
    }


def _max_rel_dev(got: dict, want: dict) -> float:
    devs = []
    for n in want:
        denom = want[n].norm().item() + 1e-12
        devs.append((got[n] - want[n]).norm().item() / denom)
    return max(devs)


def _mini_loss(pg_loss: list, k: int, micros_per_mini: int) -> float:
    return float(sum(pg_loss[k * micros_per_mini : (k + 1) * micros_per_mini]))


def _evidence(title: str, lines: list[str]) -> None:
    print(f"\nH1-EVIDENCE | {title}")
    for line in lines:
        print(f"H1-EVIDENCE |   {line}")


pytestmark = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed gloo backend unavailable",
)


def test_initial_params_identical_across_paths(measurements):
    """Seeded TinyLM gives bit-identical initial weights to the FSDP2 wrap
    and the single-rank reference — the parity comparisons are grounded."""
    cell = measurements[("balanced", "segment_mean", "mean")]
    for n, p in cell["ref"]["p0"].items():
        assert torch.equal(cell["dist"]["p0"][n], p), f"init mismatch: {n}"


def test_all_cells_ran_four_real_steps(measurements):
    for cell, data in measurements.items():
        for rank_report in data["dist"]["per_rank"]:
            assert rank_report["num_steps"] == [N_STEPS], cell
        assert data["ref"]["num_steps"] == [N_STEPS], cell
        assert len(data["dist"]["step_grads"]) == N_STEPS, cell


def test_fsdp2_reducer_averages_local_gradients(measurements):
    """MEASURED reducer behavior: in EVERY cell the distributed step-0
    gradient equals the plain average of the two ranks' local-denominator
    gradients — FSDP2's reduce-scatter divides by world size (average)."""
    lines = []
    for cell, data in measurements.items():
        got = data["dist"]["step_grads"][0]
        pred = data["pred0"]
        dev = _max_rel_dev(got, pred)
        lines.append(f"{cell}: |g_dist - avg(g_local)|/|avg| = {dev:.3e}")
        for n in pred:
            assert torch.allclose(got[n], pred[n], atol=ATOL), (cell, n)
    _evidence("reducer = AVERAGE of rank gradients (all cells)", lines)


def test_segment_mean_distributed_parity(measurements):
    """Paper objective (uniform segment mean over the pre-filter optimizer
    batch, current batch_slot path): EXACT distributed parity in every
    scenario, including uneven trees and token skew — M_local = M/W always
    holds because slots are sharded evenly."""
    lines = []
    for cell in CELLS:
        scenario, mode, reduction = cell
        if mode != "segment_mean":
            continue
        data = measurements[cell]
        # Per-step gradient parity after the reducer.
        for k in range(N_STEPS):
            got = data["dist"]["step_grads"][k]
            want = data["ref"]["step_grads"][k]
            for n in want:
                assert torch.allclose(got[n], want[n], atol=REF_ATOL), (cell, k, n)
        # 4-step parameter delta parity.
        for n, want in data["ref"]["param_delta"].items():
            assert torch.allclose(
                data["dist"]["param_delta"][n], want, atol=REF_ATOL
            ), (cell, n)
        # Loss identity: mean of rank losses == union reference loss per step.
        for k in range(N_STEPS):
            local = [
                _mini_loss(r["pg_loss"], k, LOCAL_MINI // LOCAL_MICRO)
                for r in data["dist"]["per_rank"]
            ]
            ref = _mini_loss(data["ref"]["pg_loss"], k, GLOBAL_MINI // GLOBAL_MICRO)
            assert abs(sum(local) / WORLD - ref) < 5e-6, (cell, k)
        # Grad norms match the reference per step.
        for k in range(N_STEPS):
            ref_gn = float(data["ref"]["grad_norm"][k])
            for r in data["dist"]["per_rank"]:
                assert abs(float(r["grad_norm"][k]) - ref_gn) < 1e-5, (cell, k)
        dev = max(
            _max_rel_dev(data["dist"]["step_grads"][k], data["ref"]["step_grads"][k])
            for k in range(N_STEPS)
        )
        lines.append(f"{scenario}/{reduction}: max step-grad rel-dev = {dev:.3e}")
    _evidence("segment_mean (paper objective): distributed PARITY", lines)


def test_tree_balanced_parity_only_when_trees_split_evenly(measurements):
    """tree_balanced parity DOES hold in the balanced scenario, where every
    64-row local mini-batch contains exactly N_T_union / W whole trees."""
    data = measurements[("balanced", "tree_balanced", "mean")]
    for k in range(N_STEPS):
        got = data["dist"]["step_grads"][k]
        want = data["ref"]["step_grads"][k]
        for n in want:
            assert torch.allclose(got[n], want[n], atol=REF_ATOL), (k, n)
    for n, want in data["ref"]["param_delta"].items():
        assert torch.allclose(data["dist"]["param_delta"][n], want, atol=REF_ATOL), n
    _evidence(
        "tree_balanced balanced scenario: parity (W*N_T_local == N_T_union)",
        [f"max step-grad rel-dev = "
         f"{max(_max_rel_dev(data['dist']['step_grads'][k], data['ref']['step_grads'][k]) for k in range(N_STEPS)):.3e}"],
    )


def test_tree_balanced_uneven_collapses_to_uniform_segment_mean(measurements):
    """MEASURED FINDING (not a pass/fail verdict on production): with uneven
    trees under production dispatch, the tree_balanced loss deviates from
    its own single-rank objective and its distributed gradient collapses to
    EXACTLY the uniform segment mean's gradient.

    Algebra: rank0 rows get w = 1/(16*4) = 1/64, rank1 rows 1/(2*32) = 1/64;
    after the averaging reducer every row weighs 1/128 — indistinguishable
    from segment_mean — while the single-rank reference weighs rows
    1/(18*4) vs 1/(18*32). See docs/h1_fsdp_parity_report.md (approval gate).
    """
    tb = measurements[("uneven", "tree_balanced", "mean")]
    sm = measurements[("uneven", "segment_mean", "mean")]
    # Collapse identity, step 0 (identical parameters across cells at step 0).
    got = tb["dist"]["step_grads"][0]
    collapse = sm["dist"]["step_grads"][0]
    for n in collapse:
        assert torch.allclose(got[n], collapse[n], atol=ATOL), n
    # And it is NOT the tree_balanced single-rank objective anymore.
    dev0 = _max_rel_dev(tb["dist"]["step_grads"][0], tb["ref"]["step_grads"][0])
    assert dev0 > 0.10, f"expected a large deviation, measured {dev0:.3e}"
    delta_dev = _max_rel_dev(tb["dist"]["param_delta"], tb["ref"]["param_delta"])
    _evidence(
        "tree_balanced uneven scenario: MISMATCH vs own reference",
        [
            f"step-0 grad rel-dev vs single-rank reference = {dev0:.3e}",
            f"4-step param-delta rel-dev vs reference       = {delta_dev:.3e}",
            "distributed grad == uniform segment_mean grad (collapse identity)",
            "N_T_local: rank0=16, rank1=2; N_T_union=18; W*N_T_local != N_T_union",
        ],
    )


def test_tree_balanced_interleaved_halves_the_gradient(measurements):
    """MEASURED FINDING: when rows interleave trees (every rank sees ALL
    trees), N_T_local == N_T_union on both ranks and the averaging reducer
    yields EXACTLY HALF the single-rank gradient — the naked 1/W factor."""
    data = measurements[("interleaved", "tree_balanced", "mean")]
    got = data["dist"]["step_grads"][0]
    want = data["ref"]["step_grads"][0]
    for n in want:
        assert torch.allclose(got[n], want[n] / WORLD, atol=REF_ATOL), n
    _evidence(
        "tree_balanced interleaved scenario: g_dist == g_ref / W",
        [
            f"W = {WORLD}; N_T_local == N_T_union == 64 on every rank",
            f"step-0 |g_dist| = {sum(g.norm().item()**2 for g in got.values())**0.5:.6e}",
            f"step-0 |g_ref|  = {sum(g.norm().item()**2 for g in want.values())**0.5:.6e}",
        ],
    )


def test_token_mean_local_denominator_deviation_measured():
    """Arithmetic probe on the REAL tensorized batch (no distributed run
    needed beyond the reducer proof above): a future token_mean objective
    that used LOCAL pre-filter token counts as its denominator would NOT be
    distributed-safe, because per-rank token counts are not equal in
    general. Deviation factor d(r,k) = T_union(k) / (W * T_local(r,k))."""
    full = _tiny_actor.build_batch(_scenario_edges("token_skew"))
    mask = full.batch["response_mask"]
    chunks = mask.chunk(WORLD)
    lines = []
    max_dev = 0.0
    for k in range(N_STEPS):
        locals_k = [
            int(c[k * LOCAL_MINI : (k + 1) * LOCAL_MINI].sum()) for c in chunks
        ]
        union_k = sum(locals_k)
        for r, t_local in enumerate(locals_k):
            d = union_k / (WORLD * t_local)
            max_dev = max(max_dev, abs(d - 1.0))
            lines.append(
                f"step {k} rank {r}: T_local={t_local}, T_union={union_k}, "
                f"d = T_union/(W*T_local) = {d:.4f}"
            )
    assert max_dev > 0.5, f"token skew should produce large deviation, got {max_dev}"
    balanced = _tiny_actor.build_batch(_scenario_edges("balanced"))
    b_chunks = balanced.batch["response_mask"].chunk(WORLD)
    b_locals = [int(c[:LOCAL_MINI].sum()) for c in b_chunks]
    lines.append(
        f"(balanced scenario: T_local per rank {b_locals} happen to be equal — "
        "a coincidence of the regular length pattern, not a guarantee)"
    )
    _evidence("token_mean probe: LOCAL token denominators are NOT safe", lines)


def test_fsdp1_cpu_unsupported_documented():
    """FSDP1 (strategy='fsdp') cannot be exercised on this CPU-only rig:
    torch 2.11 requires a non-CPU accelerator for FullyShardedDataParallel.
    This test pins that limitation LOUDLY (no silent skip): FSDP1 parity
    evidence must come from the server GPU smoke. If torch ever gains CPU
    support here, this test fails and H1 must be re-run for FSDP1."""
    code = (
        "import os; os.environ['CUDA_VISIBLE_DEVICES'] = '';\n"
        "import tempfile, torch, torch.distributed as dist\n"
        "rdv = tempfile.mktemp()\n"
        "dist.init_process_group('gloo', init_method=f'file://{rdv}', rank=0, world_size=1)\n"
        "from torch.distributed.fsdp import FullyShardedDataParallel as FSDP\n"
        "try:\n"
        "    FSDP(torch.nn.Linear(4, 4))\n"
        "    print('FSDP1_WRAPPED_OK')\n"
        "except Exception as e:\n"
        "    print(f'FSDP1_CPU_ERROR: {type(e).__name__}: {e}')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    out = proc.stdout + proc.stderr
    assert "FSDP1_CPU_ERROR" in out, (
        "FSDP1 wrapped on CPU — the documented torch 2.11 limitation no "
        f"longer holds; re-run H1 with FSDP1 cells. Output: {out}"
    )
    assert "accelerator" in out, out
    _evidence(
        "FSDP1 on CPU: unsupported (documented limitation, GPU-smoke item)",
        [out.strip().splitlines()[-1]],
    )
