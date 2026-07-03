# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0.
"""Every registry path constant must resolve to a real file or directory."""
from pathlib import Path

from xdomain_ser.core import registry


def _path_constants():
    return {name: val for name, val in vars(registry).items()
            if isinstance(val, Path) and name.isupper()}


def test_all_path_constants_exist():
    missing = {name: str(p) for name, p in _path_constants().items()
               if not p.exists()}
    assert not missing, f"registry paths that do not resolve: {missing}"


def test_hf_ids_are_namespaced():
    for name in ["HF_EXTRACTOR", "HF_RANKER", "HF_BASE_MODEL", "HF_NLI_MODEL"]:
        val = getattr(registry, name)
        assert isinstance(val, str) and val.count("/") == 1, (name, val)
