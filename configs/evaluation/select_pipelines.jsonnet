{
  // Filter the final inference pipeline list for scripts/evaluate.sh.
  local requested = std.split(std.extVar('APP_EVAL_PIPELINES'), ','),
  local available = [
    pipeline.inference_name
    for pipeline in super.inference_pipelines
  ],
  local missing = [
    name
    for name in requested
    if !std.member(available, name)
  ],

  assert std.length(missing) == 0 :
    'Unknown evaluation pipeline(s): ' + std.join(', ', missing)
    + '. Available pipelines: ' + std.join(', ', available),

  inference_pipelines: [
    pipeline
    for pipeline in super.inference_pipelines
    if std.member(requested, pipeline.inference_name)
  ],
}
