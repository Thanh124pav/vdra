"""Test-environment compatibility shims for optional dependency drift."""

import transformers

if not hasattr(transformers, "AutoModelForVision2Seq"):
    transformers.AutoModelForVision2Seq = object
