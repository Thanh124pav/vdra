// Math evaluations shared by all MATH training configurations.
function(base_pipeline) [
  base_pipeline + {
    task: (import '../tasks/aime24_inplace_no_answer_prefix.jsonnet'),
    dataset_split: 'test',
    dataset_portion: 1,
    inference_name: 'aime24_test',
  },
  base_pipeline + {
    task: (import '../tasks/aime25_inplace_no_answer_prefix.jsonnet'),
    dataset_split: 'test',
    dataset_portion: 1,
    inference_name: 'aime25_test',
  },
  base_pipeline + {
    task: (import '../tasks/amc23_inplace_no_answer_prefix.jsonnet'),
    dataset_split: 'test',
    dataset_portion: 1,
    inference_name: 'amc23_test',
  },
  base_pipeline + {
    task: (import '../tasks/olympiadbench_hf_inplace_no_answer_prefix.jsonnet'),
    dataset_split: 'train',
    dataset_portion: 1,
    inference_name: 'olympiadbench_test',
  },
  base_pipeline + {
    task: (import '../tasks/collegeMath_inplace_no_answer_prefix.jsonnet'),
    dataset_split: 'test',
    dataset_portion: 1,
    inference_name: 'collegeMath_test',
  },
]
