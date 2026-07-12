"""GEAR / tree-family RL recipe for verl.

Ports the treetune (DeepSpeed) SPO/GEAR/TreeRL/TreePO algorithms to verl while
keeping all algorithm logic byte-identical. The pure-math + grading modules live
under ``gear_core`` (vendored verbatim from treetune; only import paths rewritten).
"""
