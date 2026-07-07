(import 'math_inplace_no_answer_prefix.jsonnet') + {
  load_dataset_dict: true,
  dataset_dict_path: 'data/aime24',
  hf_dataset_args: null,
  problem_field: 'problem',
  answer_field: null,
  solution_field: 'solution',
  normalize_dataset_fields: true,
  use_dataset_answer: true,
}
