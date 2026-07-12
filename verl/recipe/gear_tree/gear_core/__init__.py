"""Vendored, byte-identical algorithm core from treetune.

Contents (formulas unchanged; only import statements rewritten to be
self-contained):

* ``gear/``            -- GEAR online prune/share + budget allocation + TV math
                          (from ``treetune/gear/``)
* ``tree_update_modes`` -- SPO / TreePO / TreeRL segment-advantage core
                          (``compute_tree_update_values``)
* ``grading/``         -- MATH/GSM8K answer grading (from ``treetune/tasks/``)
* ``logging_utils``    -- logging helper used by the grading modules
"""
