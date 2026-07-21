"""Test-environment compatibility shims for optional dependency drift.

The shims themselves live in ``_test_shims.py`` so that ``mp.spawn`` worker
modules (which re-import outside pytest and never run this conftest) can
install the exact same shims before touching the verl import chain.
"""

try:  # namespace-package import under PYTHONPATH=verl
    from recipe.gear_tree.tests import _test_shims
except ImportError:  # flat rootdir-relative import
    import _test_shims

_test_shims.install()
