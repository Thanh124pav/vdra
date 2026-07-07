(import 'math_inplace_no_answer_prefix.jsonnet') + {
  load_dataset_dict: true,
  dataset_dict_path: 'data/olympiadbench_hf',
  hf_dataset_args: null,
  problem_field: 'question',
  answer_field: 'final_answer',
  solution_field: 'solution',
  normalize_dataset_fields: true,
  use_dataset_answer: true,
}
