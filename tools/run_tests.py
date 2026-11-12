"""Fallback test runner for machines where pytest cannot be installed.

On this machine pip fails with an SSL certificate error (likely HTTPS
interception by antivirus or a proxy). This runner stubs the small part of
the pytest API the suite uses (fixtures by name, parametrize, raises,
approx, skip) and runs every ``tests/test_*.py``. Prefer the real thing
when it is available:

    python -m pytest          # preferred
    python tools/run_tests.py # fallback
"""

from __future__ import annotations

import builtins
import contextlib
import importlib
import inspect
import re
import sys
import tempfile
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class _Skip(Exception):
    pass


class _Raises(contextlib.AbstractContextManager):
    def __init__(self, exc, match=None):
        self.exc, self.match = exc, match

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"DID NOT RAISE {self.exc}")
        if not issubclass(et, self.exc):
            return False
        if self.match and not re.search(self.match, str(ev)):
            raise AssertionError(f"{ev!r} does not match {self.match!r}")
        return True


class _Approx:
    def __init__(self, expected, rel=1e-6, abs=1e-12):  # pytest's defaults
        self.expected, self.rel, self.abs = expected, rel, abs

    def __eq__(self, other):
        if isinstance(self.expected, (list, tuple)):
            return len(other) == len(self.expected) and all(
                _Approx(e, self.rel, self.abs) == o for e, o in zip(self.expected, other))
        tol = max(self.rel * max(builtins.abs(self.expected), 1e-12), self.abs)
        return builtins.abs(other - self.expected) <= tol

    def __repr__(self):
        return f"approx({self.expected!r})"


class _Mark:
    def __getattr__(self, name):
        if name == "parametrize":
            def parametrize(argname, values):
                def deco(fn):
                    fn._params = (argname, list(values))
                    return fn
                return deco
            return parametrize
        return lambda fn: fn


def _install_stub() -> None:
    pt = types.ModuleType("pytest")
    pt.fixture = lambda fn=None, **kw: fn if fn else (lambda g: g)
    pt.skip = lambda msg="": (_ for _ in ()).throw(_Skip(msg))
    pt.raises = _Raises
    pt.approx = _Approx
    pt.mark = _Mark()
    sys.modules["pytest"] = pt


def _resolve(name, modules, tmp, cache):
    """Build fixture ``name`` from the test module or conftest, resolving its own arguments."""
    if name in cache:
        return cache[name]
    if name == "tmp_path":
        value = tmp
    else:
        fn = next((getattr(m, name) for m in modules if hasattr(m, name)), None)
        if fn is None:
            raise LookupError(f"unknown fixture {name!r}")
        args = {p: _resolve(p, modules, tmp, cache) for p in inspect.signature(fn).parameters}
        value = fn(**args)
    cache[name] = value
    return value


def main() -> int:
    _install_stub()
    conftest = importlib.import_module("tests.conftest")
    ok = fail = skip = 0
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        mod = importlib.import_module(f"tests.{path.stem}")
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("test_") or fn.__module__ != mod.__name__:
                continue
            cases = [{}]
            if hasattr(fn, "_params"):
                arg, values = fn._params
                cases = [{arg: v} for v in values]
            for case in cases:
                label = f"{path.stem}::{name}" + (f"[{next(iter(case.values()))}]" if case else "")
                with tempfile.TemporaryDirectory() as td:
                    try:
                        cache = dict(case)
                        kwargs = {p: _resolve(p, (mod, conftest), Path(td), cache)
                                  for p in inspect.signature(fn).parameters}
                        fn(**kwargs)
                        ok += 1
                        print("PASS", label)
                    except _Skip as exc:
                        skip += 1
                        print("SKIP", label, exc)
                    except Exception:
                        fail += 1
                        print("FAIL", label)
                        traceback.print_exc()
    print(f"\n{ok} passed, {fail} failed, {skip} skipped")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
