# Top-level conftest to prevent pytest from attempting to collect non-Python
# files that accidentally start with `test_` (e.g. binary logs). This keeps
# existing Python tests collected while ignoring stray files like
# `test_errors.txt` that are not intended as test modules.
collect_ignore = ["test_errors.txt"]
