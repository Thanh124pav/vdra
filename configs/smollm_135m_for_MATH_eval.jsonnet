local hf_model_name = 'HuggingFaceTB/SmolLM2-135M';
local tokenizer = {
  type: 'pretrained',
  hf_model_name: hf_model_name,
};

local math_inference_pipeline =
  (import 'prompt_library/smollm_math.jsonnet')
  + (import 'inference_strategies/tree/iid_expander.jsonnet')
  + (import 'inference_strategies/cot.jsonnet')
  + {
    inference_strategy+: {
      max_concurrent_programs: 8,
      max_concurrent_generations: 4,

      node_expander+: {
        type: 'efficient_iid',
        program_kwargs: {
          temperature: 0,
          top_p: 1,
          max_tokens: 512,
          stop: '"\n\n\nProblem:"',
          logprobs: 0,
        },
        node_text_template: '{chain_of_thought}',
        model_context_size: 1024,
        tokenizer: tokenizer,
      },
      answer_extractor+: {
        type: 'identity',
        node_key_name: 'text',
      },
      samples: 1,
      max_depth: 100,
      guidance_llm: (import 'guidance_llms/smollm_135m.jsonnet') + {
        api_base: 'none',
      },
      no_cache: false,
      question_field: 'query',
      seed: 42,
    },
    task: (import 'tasks/math_inplace_no_answer_prefix.jsonnet'),
    analyzers: [(import 'analyzers/task_performance.jsonnet')],
  };

local math_test_inference_pipeline =
  math_inference_pipeline + {
    dataset_split: 'test',
    dataset_portion: 1,
    inference_name: 'math_test',
  };

local math_benchmark_pipelines =
  (import 'evaluation/math_benchmarks.libsonnet')(math_inference_pipeline);

{
  tokenizer: tokenizer,
  inference_pipelines: [
    math_test_inference_pipeline,
  ] + math_benchmark_pipelines,
  evaluation_vllm_server: {
    max_num_seqs: 8,
    max_model_len: 1024,
  },
}
