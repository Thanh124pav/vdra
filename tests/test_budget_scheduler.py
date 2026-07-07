from treetune.gear.budget_scheduler import FlexibleBudgetScheduler


def test_flexible_scheduler_marks_queue_ids_and_floor_allocates():
    nodes = [
        {"gear_segment_id": "a", "gear_reward_variance": 0.0},
        {"gear_segment_id": "b", "gear_reward_variance": 0.5},
        {"gear_segment_id": "c", "gear_reward_variance": 1.0},
    ]
    scheduler = FlexibleBudgetScheduler(queue_count=2, lambda_=0.02)
    summaries = scheduler.allocate(nodes, total_depth_budget=9)

    assert summaries
    assert all("gear_budget_queue_id" in node for node in nodes)
    assert sum(summary.allocated_budget for summary in summaries) <= 9
    assert sum(summary.underallocated_budget for summary in summaries) == 9 - sum(
        summary.allocated_budget for summary in summaries
    )


def test_flexible_scheduler_passes_n_min_to_queue_allocations():
    nodes = [
        {"gear_segment_id": "a", "gear_reward_variance": 0.0},
        {"gear_segment_id": "b", "gear_reward_variance": 0.0},
    ]
    scheduler = FlexibleBudgetScheduler(queue_count=1, lambda_=0.02, n_min=1)

    summaries = scheduler.allocate(nodes, total_depth_budget=4)

    assert summaries[0].allocations == {"a": 1, "b": 1}
