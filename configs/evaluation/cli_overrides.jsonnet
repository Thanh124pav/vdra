// Runtime overrides supplied by scripts/evaluate.sh.
local tokenizer_name = std.extVar('APP_EVAL_TOKENIZER');
local context_length_str = std.extVar('APP_EVAL_CONTEXT_LENGTH');
local max_new_tokens_str = std.extVar('APP_EVAL_MAX_NEW_TOKENS');
local has_tokenizer = tokenizer_name != '';
local has_context_length = context_length_str != '';
local has_max_new_tokens = max_new_tokens_str != '';
local context_length =
  if has_context_length then std.parseInt(context_length_str) else null;
local max_new_tokens =
  if has_max_new_tokens then std.parseInt(max_new_tokens_str) else null;

(
  if has_tokenizer then {
    tokenizer+: {
      hf_model_name: tokenizer_name,
    },
  } else {}
)
+ {
  inference_pipelines: [
    pipeline
    + (
      if has_tokenizer then {
        inference_strategy+: {
          guidance_llm+: {
            tokenizer_name: tokenizer_name,
          },
          node_expander+: {
            tokenizer+: {
              hf_model_name: tokenizer_name,
            },
          },
        },
      } else {}
    )
    + (
      if has_context_length then {
        inference_strategy+: {
          node_expander+: {
            model_context_size: context_length,
          },
        },
      } else {}
    )
    + (
      if has_max_new_tokens then {
        inference_strategy+: {
          node_expander+: {
            program_kwargs+: {
              max_tokens: max_new_tokens,
            },
          },
        },
      } else {}
    )
    for pipeline in super.inference_pipelines
  ],
}
+ (
  if has_context_length then {
    evaluation_vllm_server+: {
      max_model_len: context_length,
    },
  } else {}
)
