"""Regression coverage for the real Ray training entrypoints."""

import pytest


def _init_ray_for_entrypoint_test(ray):
    ray.shutdown()
    ray.init(
        num_cpus=2,
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
        _temp_dir="/tmp/vdra_ray_entrypoint_test",
    )


def test_main_ppo_imports_without_ray_init():
    import ray

    assert not ray.is_initialized()
    import verl.trainer.main_ppo  # noqa: F401


def test_gear_tree_entrypoint_imports_without_ray_init():
    import ray

    assert not ray.is_initialized()
    import recipe.gear_tree.main_gear_tree  # noqa: F401


def test_task_runner_actor_can_start():
    import ray
    from verl.trainer.main_ppo import TaskRunner

    _init_ray_for_entrypoint_test(ray)
    try:
        runner = TaskRunner.remote()
        assert ray.get(runner.healthcheck.remote()) is True
    finally:
        ray.shutdown()


def test_gear_tree_task_runner_actor_can_start():
    import ray
    from recipe.gear_tree.main_gear_tree import GearTreeTaskRunner

    _init_ray_for_entrypoint_test(ray)
    try:
        runner = GearTreeTaskRunner.remote()
        assert ray.get(runner.healthcheck.remote()) is True
    finally:
        ray.shutdown()


def test_gear_tree_runner_inherits_plain_base():
    from recipe.gear_tree.main_gear_tree import GearTreeTaskRunner
    from verl.trainer.main_ppo import TaskRunnerBase

    plain_cls = GearTreeTaskRunner.__ray_metadata__.modified_class
    assert issubclass(plain_cls, TaskRunnerBase)


def test_gear_tree_import_does_not_raise_actor_inheritance_exception():
    try:
        import recipe.gear_tree.main_gear_tree  # noqa: F401
    except Exception as exc:  # pragma: no cover - only reached on regression
        if exc.__class__.__name__ == "ActorClassInheritanceException":
            pytest.fail("GearTree entrypoint must not subclass a Ray actor class")
        raise
