import asyncio
from types import SimpleNamespace

import pytest

from guidance.library._gen import gen


class FakeParser:
    def __init__(self, response):
        self.response = response
        self.program = SimpleNamespace(
            logprobs=None,
            stream=False,
            _displaying=False,
            caching=False,
            cache_seed=0,
        )
        self.should_stop = False
        self.executing = True
        self.llm_session = self._llm_session

    async def _llm_session(self, prompt, **kwargs):
        self.last_prompt = prompt
        self.last_kwargs = kwargs
        return self.response


def _context(response):
    parser = FakeParser(response)
    variables = {"@raw_prefix": "", "@prefix": "prompt"}
    return parser, variables, {
        "parser": parser,
        "variable_stack": variables,
        "next_node": None,
        "next_next_node": None,
        "prev_node": None,
    }


def test_gen_stores_single_completion_token_logprobs():
    response = {
        "choices": [
            {
                "text": " answer",
                "finish_reason": "stop",
                "logprobs": {
                    "token_logprobs": [-0.2, -0.3],
                    "tokens": [" answer", "</s>"],
                },
            }
        ]
    }
    parser, variables, context = _context(response)

    asyncio.run(gen(name="chain_of_thought", logprobs=1, _parser_context=context))

    assert parser.last_kwargs["logprobs"] == 1
    assert variables["chain_of_thought"] == " answer"
    assert variables["chain_of_thought_finish_reason"] == "stop"
    assert variables["chain_of_thought_logprobs"] == [-0.2, -0.3]
    assert variables["chain_of_thought_tokens"] == [" answer", "</s>"]


def test_gen_stores_batched_completion_token_logprobs():
    response = {
        "choices": [
            {
                "text": " first",
                "finish_reason": "length",
                "logprobs": {
                    "token_logprobs": [-0.1, -0.2],
                    "tokens": [" first", " step"],
                },
            },
            {
                "text": " second",
                "finish_reason": "stop",
                "logprobs": {
                    "token_logprobs": [-0.3],
                    "tokens": [" second"],
                },
            },
        ]
    }
    parser, variables, context = _context(response)

    asyncio.run(gen(name="chain_of_thought", n=2, logprobs=1, _parser_context=context))

    assert parser.last_kwargs["n"] == 2
    assert parser.last_kwargs["logprobs"] == 1
    assert variables["chain_of_thought"] == [" first", " second"]
    assert variables["chain_of_thought_finish_reason"] == ["length", "stop"]
    assert variables["chain_of_thought_logprobs"] == [[-0.1, -0.2], [-0.3]]
    assert variables["chain_of_thought_tokens"] == [[" first", " step"], [" second"]]


def test_gen_fails_fast_when_requested_logprobs_are_missing():
    response = {"choices": [{"text": " answer", "finish_reason": "stop"}]}
    _, _, context = _context(response)

    with pytest.raises(RuntimeError, match="did not include logprobs"):
        asyncio.run(gen(name="chain_of_thought", logprobs=1, _parser_context=context))