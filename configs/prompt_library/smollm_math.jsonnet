local tree_expansion_iid = '{{prefix}}{{gen "chain_of_thought" temperature={temperature} top_p={top_p} max_tokens={max_tokens} seed={seed} logprobs={logprobs} save_stop_text="stop_text" stop={stop} n={num_samples}}}';
local tree_question_template = 'Solve the following math problem. Show your reasoning, then put only the final answer inside \\boxed{}.\n\nProblem:\n{query}\n\nSolution:\n';

{
  prompt_library+: {
    tree+: {
      expansion+: {
        iid: tree_expansion_iid,
      },
      question_template: tree_question_template,
    },
  },
}
