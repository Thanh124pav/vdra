local hf_model_name = 'Qwen/Qwen2.5-0.5B-Instruct';
local tokenizer = { type: 'pretrained', hf_model_name: hf_model_name };
local guidance_llm = (import '../guidance_llms/openai_vllm.jsonnet') + {
  model: hf_model_name,
  tokenizer_name: hf_model_name,
  api_base: 'none',
};

{
  episode_generator+: {
    initial_model_name_or_path: hf_model_name,
    inference_strategy+: { guidance_llm: guidance_llm },
    value_estimation_inference_strategy+: { guidance_llm: guidance_llm },
  },
  tokenizer: tokenizer,
  trainer+: {
    actor_model+: { hf_model_name: hf_model_name },
    reference_model+: { hf_model_name: hf_model_name },
  },
}
