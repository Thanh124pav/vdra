// Qwen/Qwen3.5-7B is kept as a user-provided/local alias. At the time this
// config was added, public Qwen3.5 HF listings exposed 4B/9B/27B variants
// but not an official 7B repository.
local hf_model_name = 'Qwen/Qwen3.5-7B';
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
