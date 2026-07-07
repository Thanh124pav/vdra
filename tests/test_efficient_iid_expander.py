import asyncio

from treetune.inference_strategies.tree_inference.expansion import EfficientIIDExpander


class FixedBranchFactor:
    def __init__(self, value):
        self.value = value

    def __call__(self, _node):
        return self.value


class ProgramResult:
    def __init__(self, variables, text="prompt response"):
        self._variables = variables
        self.text = text

    def variables(self):
        return self._variables


def make_expander(*, num_expansion_rounds=1, branch_factor=1, logprobs=0):
    return EfficientIIDExpander(
        program=(
            '{{prefix}}{{gen "chain_of_thought" max_tokens={max_tokens} '
            'logprobs={logprobs} n={num_samples}}}'
        ),
        node_text_template="{chain_of_thought}",
        program_kwargs={"max_tokens": 16, "logprobs": logprobs},
        num_expansion_rounds=num_expansion_rounds,
        branch_factor_strategy=FixedBranchFactor(branch_factor),
    )


def test_expand_returns_no_nodes_when_all_rounds_lack_chain_of_thought():
    expander = make_expander(num_expansion_rounds=2)

    async def run_program(_program, *, prefix):
        return ProgramResult({"error": f"no generation for {prefix}"})

    expander.set_run_program(run_program)

    assert asyncio.run(expander.expand({}, "prompt", depth=0)) == []


def test_expand_keeps_valid_round_when_another_round_is_malformed():
    expander = make_expander(num_expansion_rounds=2)
    results = iter(
        [
            ProgramResult({"error": "temporary malformed response"}),
            ProgramResult(
                {
                    "chain_of_thought": "valid response",
                    "chain_of_thought_finish_reason": "stop",
                }
            ),
        ]
    )

    async def run_program(_program, *, prefix):
        return next(results)

    expander.set_run_program(run_program)

    nodes = asyncio.run(expander.expand({}, "prompt", depth=0))

    assert [node["text"] for node in nodes] == ["valid response"]


def test_expand_preserves_and_stores_requested_logprobs():
    expander = make_expander(branch_factor=2, logprobs=1)

    async def run_program(program, *, prefix):
        assert "logprobs=1" in program
        return ProgramResult(
            {
                "chain_of_thought": ["first", "second"],
                "chain_of_thought_finish_reason": ["stop", "length"],
                "chain_of_thought_logprobs": [[-0.5, -1.0], [-2.0]],
            }
        )

    expander.set_run_program(run_program)

    nodes = asyncio.run(expander.expand({}, "prompt", depth=0))

    assert [node["sum_logprobs"] for node in nodes] == [-1.5, -2.0]
    assert [node["num_tokens"] for node in nodes] == [2, 1]
