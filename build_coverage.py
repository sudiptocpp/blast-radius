"""Build a zlib-compressed, bitmask-indexed dependency index from the base checkout.

Run from the base checkout with the scorer's pytest dependencies:
    python /app/build_coverage.py /app/coverage.pkl.zlib
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import enum
import importlib
import importlib.machinery
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
import zlib
from collections import defaultdict
from functools import wraps
from pathlib import Path

import pytest


SCALAR_TYPES = (int, float, bool, str, bytes, enum.Enum, tuple, frozenset, set, type(None))


def canonical_key(k):
    """Serialize hashable literal/enum/type/tuple dict keys deterministically."""
    if isinstance(k, str):
        return f"s:{k}"
    if isinstance(k, bool):
        return f"b:{'1' if k else '0'}"
    if isinstance(k, int):
        return f"i:{k}"
    if k is None:
        return "n:"
    if isinstance(k, enum.Enum):
        cls = type(k)
        return f"e:{cls.__module__}.{cls.__qualname__}.{k.name}"
    if isinstance(k, type):
        return f"c:{k.__module__}.{k.__qualname__}"
    if isinstance(k, tuple):
        parts = []
        for item in k:
            part = canonical_key(item)
            if part is None:
                return None
            parts.append(part)
        return f"t:({','.join(parts)})"
    return None


class _TrackedDict(dict):
    """Dict subclass recording per-key and full-dict accesses per active context."""

    __slots__ = ("_owners", "_attr", "_tracker")

    def __init__(self, owners, attr, initial, tracker):
        super().__init__(initial)
        self._owners = owners  # list[(owner_key, attr_name)]
        self._attr = attr
        self._tracker = tracker

    def _record_key(self, key):
        tr = self._tracker
        if tr.pause_data or not tr.active:
            return
        ctx = tr.context
        if ctx.startswith("collection:") and tr._caller_in_sqlglot_init():
            return
        ck = canonical_key(key)
        if ck is None:
            self._record_all()
            return
        for owner, attr in self._owners:
            tr.dict_key_usage[(owner, attr, ck)].add(ctx)

    def _record_all(self):
        tr = self._tracker
        if tr.pause_data or not tr.active:
            return
        ctx = tr.context
        if ctx.startswith("collection:") and tr._caller_in_sqlglot_init():
            return
        for owner, attr in self._owners:
            tr.dict_all_usage[(owner, attr)].add(ctx)

    def __getitem__(self, key):
        self._record_key(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._record_key(key)
        return super().get(key, default)

    def __contains__(self, key):
        self._record_key(key)
        return super().__contains__(key)

    def __iter__(self):
        self._record_all()
        return super().__iter__()

    def keys(self):
        self._record_all()
        return super().keys()

    def values(self):
        self._record_all()
        return super().values()

    def items(self):
        self._record_all()
        return super().items()

    def copy(self):
        self._record_all()
        return dict(self)

    def __reduce__(self):
        return (dict, (dict(self),))


class _TrackedAttr:
    """Non-data descriptor recording class-attribute reads without leaking memory."""

    __slots__ = ("_attr", "_val", "_tracker")

    def __init__(self, attr, val, tracker):
        self._attr = attr
        self._val = val
        self._tracker = tracker

    def __get__(self, obj, objtype=None):
        tr = self._tracker
        if objtype is not None and not tr.pause_data and tr.active:
            ctx = tr.context
            if not (ctx.startswith("collection:") and tr._caller_in_sqlglot_init()):
                ckey = tr.class_key(objtype)
                if ckey is None:
                    for base in objtype.__mro__:
                        if self._attr in base.__dict__:
                            ckey = tr.class_key(base)
                            if ckey is not None:
                                break
                if ckey is not None:
                    tr.attr_usage[(ckey, self._attr)].add(ctx)
        return self._val


class _InlinePoolExecutor:
    """Drop-in replacement for ProcessPoolExecutor executing inline with pickle round-trips."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kwargs):
        if initializer is not None:
            fn, args = pickle.loads(pickle.dumps((initializer, initargs)))
            fn(*args)

    def submit(self, fn, *args, **kwargs):
        future = concurrent.futures.Future()
        try:
            p_fn, p_args, p_kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
            res = p_fn(*p_args, **p_kwargs)
            res = pickle.loads(pickle.dumps(res))
            future.set_result(res)
        except BaseException as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class CoverageIndex:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root_prefix = self.root.as_posix().rstrip("/") + "/"
        self.context = "collection:*"
        self.active = True
        self.pause_data = 0

        self.collected = set()
        self.passed = set()
        self.unverified = set()
        self.collection_errors = set()
        self.fixture_contexts = {}
        self.fixture_users = defaultdict(set)
        self.resources = defaultdict(set)
        self.module_usage = defaultdict(set)

        # sys.monitoring state
        self.func_lines = defaultdict(lambda: defaultdict(set))
        self.func_calls = defaultdict(lambda: defaultdict(set))
        self.init_lines = defaultdict(lambda: defaultdict(set))
        self.init_collection = defaultdict(lambda: defaultdict(set))
        self.func_calls_collection = defaultdict(lambda: defaultdict(set))
        self._code_cache = {}
        self._class_defs_cache = {}
        self._reading_class_defs = False
        self._test_start_times = {}
        self.test_cpu_ms = {}

        # Fine-grained data dependencies
        self.dict_key_usage = defaultdict(set)
        self.dict_all_usage = defaultdict(set)
        self.attr_usage = defaultdict(set)
        self.expr_usage = defaultdict(set)
        self._seen_exprs = set()
        self._tracked_dicts_by_id = {}  # id(dict) -> _TrackedDict
        self._instrumented_classes = set()
        self._instrumented_typing_modules = set()
        self._synthetic_class_keys = {}
        self.dialect_paths = {}
        self.restore_hooks = []

        self.tool_id = sys.monitoring.COVERAGE_ID

    def relative_path(self, path):
        if not path:
            return None
        path_str = os.fspath(path)
        norm = path_str.replace("\\", "/")
        if not norm.startswith("/"):
            norm = os.path.abspath(path_str).replace("\\", "/")
        if norm.startswith(self.root_prefix):
            return norm[len(self.root_prefix):]
        try:
            return Path(os.path.abspath(path_str)).relative_to(self.root).as_posix()
        except ValueError:
            return None

    def class_key(self, cls):
        syn = self._synthetic_class_keys.get(cls)
        if syn is not None:
            return syn
        mod = sys.modules.get(getattr(cls, "__module__", ""))
        filename = getattr(mod, "__file__", None)
        if not filename:
            return None
        rel = self.relative_path(filename)
        if not rel or not rel.startswith("sqlglot/"):
            return None
        return f"{rel}:{cls.__qualname__}"

    def _caller_in_sqlglot_init(self):
        frame = sys._getframe(2)
        for _ in range(4):
            if frame is None:
                return False
            rel = self.relative_path(frame.f_code.co_filename)
            if rel and rel.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/", "sqlglot/expressions/")):
                name = frame.f_code.co_name
                if name == "<module>" or (name and name[0].isupper()):
                    return True
            frame = frame.f_back
        return False

    def _get_class_defs(self, rel_path):
        cached = self._class_defs_cache.get(rel_path)
        if cached is not None:
            return cached
        full = self.root / rel_path
        defs = set()
        try:
            self._reading_class_defs = True
            try:
                source = full.read_text(encoding="utf-8")
            finally:
                self._reading_class_defs = False
            tree = ast.parse(source, filename=rel_path)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    defs.add((node.lineno, node.name))
        except (OSError, SyntaxError):
            pass
        self._class_defs_cache[rel_path] = defs
        return defs

    def _code_info(self, code):
        cached = self._code_cache.get(code)
        if cached is not None:
            return cached
        rel = self.relative_path(code.co_filename)
        if not rel or not rel.startswith(("sqlglot/", "tests/", "sqlglotc/")):
            info = (None, False, None)
            self._code_cache[code] = info
            return info
        name = code.co_name
        if name == "<module>":
            is_func = False
        elif (code.co_firstlineno, name) in self._get_class_defs(rel):
            is_func = False
        else:
            is_func = True
        info = (rel, is_func, (code.co_firstlineno, name))
        self._code_cache[code] = info
        return info

    def on_py_start(self, code, instruction_offset):
        if not self.active:
            return sys.monitoring.DISABLE
        rel, is_func, func_key = self._code_info(code)
        if rel is None:
            return sys.monitoring.DISABLE
        if is_func:
            ctx = self.context
            if ctx.startswith("collection:"):
                self.func_calls_collection[rel][func_key].add(ctx)
            else:
                self.func_calls[rel][func_key].add(ctx)
        return sys.monitoring.DISABLE

    def on_line(self, code, lineno):
        if not self.active:
            return sys.monitoring.DISABLE
        rel, is_func, _ = self._code_info(code)
        if rel is None:
            return sys.monitoring.DISABLE
        ctx = self.context
        if ctx.startswith("collection:"):
            self.init_collection[rel][lineno].add(ctx)
        elif is_func:
            self.func_lines[rel][lineno].add(ctx)
        else:
            self.init_lines[rel][lineno].add(ctx)
        return sys.monitoring.DISABLE

    def start_monitoring(self):
        sys.monitoring.use_tool_id(self.tool_id, "blast_coverage")
        sys.monitoring.register_callback(self.tool_id, sys.monitoring.events.PY_START, self.on_py_start)
        sys.monitoring.register_callback(self.tool_id, sys.monitoring.events.LINE, self.on_line)
        sys.monitoring.set_events(self.tool_id, sys.monitoring.events.PY_START | sys.monitoring.events.LINE)

    def stop_monitoring(self):
        try:
            sys.monitoring.set_events(self.tool_id, 0)
            sys.monitoring.register_callback(self.tool_id, sys.monitoring.events.PY_START, None)
            sys.monitoring.register_callback(self.tool_id, sys.monitoring.events.LINE, None)
            sys.monitoring.free_tool_id(self.tool_id)
        except ValueError:
            pass

    def switch(self, context):
        previous = self.context
        if context != previous:
            self.context = context
            self._seen_exprs.clear()
            if self.active:
                sys.monitoring.restart_events()
        return previous

    def audit_open(self, event, args):
        if event != "open" or not self.active or self._reading_class_defs:
            return
        raw_path, _, flags = args
        if isinstance(raw_path, int) or flags & (os.O_WRONLY | os.O_RDWR) == os.O_WRONLY:
            return
        path = os.fsdecode(raw_path)
        if path.endswith((".pyc", ".pyo")):
            return
        relative = self.relative_path(path)
        if not relative or relative.startswith((".git/", ".pytest_cache/")):
            return
        if relative.endswith(".py"):
            caller = sys._getframe(1)
            for _ in range(5):
                source = caller.f_code.co_filename.replace("\\", "/")
                if source.startswith("<frozen importlib") or "/_pytest/assertion/" in source:
                    return
                if caller.f_back is None or caller.f_globals.get("__name__", "").startswith("tests."):
                    break
                caller = caller.f_back
        self.resources[relative].add(self.context)

    def _record_expr(self, cls):
        if self.pause_data or not self.active:
            return
        ctx = self.context
        if ctx.startswith("collection:") and self._caller_in_sqlglot_init():
            return
        if cls in self._seen_exprs:
            return
        self._seen_exprs.add(cls)
        ckey = self.class_key(cls)
        if ckey is not None:
            self.expr_usage[ckey].add(ctx)

    def _wrap_dict(self, owner_key, name, val):
        entry = self._tracked_dicts_by_id.get(id(val))
        if entry is not None:
            td, _ = entry
            if (owner_key, name) not in td._owners:
                td._owners.append((owner_key, name))
            return td
        td = _TrackedDict([(owner_key, name)], name, val, self)
        self._tracked_dicts_by_id[id(val)] = (td, val)
        return td

    def instrument_class(self, cls, skip_attrs=()):
        if cls in self._instrumented_classes:
            return
        ckey = self.class_key(cls)
        if ckey is None:
            return
        self._instrumented_classes.add(cls)

        for name, val in list(cls.__dict__.items()):
            if name.startswith("_") or name in skip_attrs:
                continue
            if isinstance(val, _TrackedDict):
                if (ckey, name) not in val._owners:
                    val._owners.append((ckey, name))
                continue
            if isinstance(val, _TrackedAttr):
                continue
            if type(val) is dict:
                setattr(cls, name, self._wrap_dict(ckey, name, val))
            elif isinstance(val, SCALAR_TYPES):
                setattr(cls, name, _TrackedAttr(name, val, self))

    def instrument_expr_class(self, cls):
        if cls in self._instrumented_classes:
            return
        ckey = self.class_key(cls)
        if ckey is None:
            return
        self._instrumented_classes.add(cls)
        val = cls.__dict__.get("arg_types")
        if type(val) is dict:
            setattr(cls, "arg_types", self._wrap_dict(ckey, "arg_types", val))

    def instrument_typing_module(self, mod):
        name = getattr(mod, "__name__", "")
        if not name.startswith("sqlglot.typing.") or mod in self._instrumented_typing_modules:
            return
        self._instrumented_typing_modules.add(mod)
        filename = getattr(mod, "__file__", None)
        rel = self.relative_path(filename)
        if not rel:
            return
        val = mod.__dict__.get("EXPRESSION_METADATA")
        if type(val) is dict:
            mod.__dict__["EXPRESSION_METADATA"] = self._wrap_dict(rel, "EXPRESSION_METADATA", val)

    def instrument_loaded_modules(self):
        from sqlglot.dialects.dialect import Dialect
        from sqlglot.expressions.core import Expr
        from sqlglot.generator import Generator
        from sqlglot.parser import Parser
        from sqlglot.tokens import _TokenizerBase

        def walk_subclasses(base, fn):
            fn(base)
            for sub in list(base.__subclasses__()):
                walk_subclasses(sub, fn)

        walk_subclasses(Dialect, lambda c: self.instrument_class(c, skip_attrs=("tokenizer_class", "jsonpath_tokenizer_class", "parser_class", "generator_class")))
        walk_subclasses(_TokenizerBase, self.instrument_class)
        walk_subclasses(Parser, self.instrument_class)
        walk_subclasses(Generator, self.instrument_class)
        walk_subclasses(Expr, self.instrument_expr_class)

        for mod in list(sys.modules.values()):
            if mod is not None and getattr(mod, "__name__", "").startswith("sqlglot.typing."):
                self.instrument_typing_module(mod)

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_make_collect_report(self, collector):
        path = self.relative_path(collector.path)
        previous = self.switch(f"collection:{path if path and path.endswith('.py') else '*'}")
        try:
            report = yield
            if report.failed:
                self.collection_errors.add(report.nodeid)
            return report
        finally:
            self.switch(previous)

    def pytest_collection_finish(self, session):
        for item in session.items:
            self.collected.add(item.nodeid)
            fixture_info = getattr(item, "_fixtureinfo", None)
            if fixture_info is None:
                continue
            for definitions in fixture_info.name2fixturedefs.values():
                for fixture in definitions:
                    if fixture.scope != "function":
                        self.fixture_users[self.fixture_context(fixture)].add(item.nodeid)

        self._install_runtime_hooks()
        self.instrument_loaded_modules()

    def _install_runtime_hooks(self):
        from sqlglot.dialects.dialect import Dialect, _Dialect
        from sqlglot.expressions.core import Expr, Func
        from sqlglot.generator import Generator
        from sqlglot.parser import Parser
        from sqlglot.tokens import _TokenizerBase

        trace = self

        # 1. Dialect.get_or_raise module tracking
        descriptor = vars(Dialect)["get_or_raise"]

        @wraps(descriptor.__func__)
        def trace_dialect(cls, *args, **kwargs):
            instance = descriptor.__func__(cls, *args, **kwargs)
            dialect_type = type(instance)
            if dialect_type not in trace.dialect_paths:
                trace.dialect_paths[dialect_type] = trace.dialect_modules(dialect_type)
            for path in trace.dialect_paths[dialect_type]:
                trace.module_usage[path].add(trace.context)
            return instance

        Dialect.get_or_raise = classmethod(trace_dialect)
        self.restore_hooks.append((Dialect, "get_or_raise", descriptor))

        # 2. _Dialect.__new__ hook
        orig_dialect_new = _Dialect.__new__

        @wraps(orig_dialect_new)
        def tracked_dialect_new(cls, clsname, bases, attrs):
            trace.pause_data += 1
            try:
                klass = orig_dialect_new(cls, clsname, bases, attrs)
            finally:
                trace.pause_data -= 1
            trace.instrument_class(
                klass,
                skip_attrs=("tokenizer_class", "jsonpath_tokenizer_class", "parser_class", "generator_class"),
            )
            d_key = trace.class_key(klass)
            for attr_name in ("tokenizer_class", "jsonpath_tokenizer_class", "parser_class", "generator_class"):
                sub = getattr(klass, attr_name, None)
                if isinstance(sub, type):
                    if d_key and sub.__module__ == "sqlglot.dialects.dialect" and sub.__qualname__ in ("Tokenizer", "JSONPathTokenizer"):
                        trace._synthetic_class_keys[sub] = f"{d_key}._auto_{sub.__qualname__}"
                    trace.instrument_class(sub)
            return klass

        _Dialect.__new__ = tracked_dialect_new
        self.restore_hooks.append((_Dialect, "__new__", orig_dialect_new))

        # 3. _TokenizerBase.__init_subclass__ hook
        orig_tok_init = vars(_TokenizerBase)["__init_subclass__"].__func__

        @classmethod
        @wraps(orig_tok_init)
        def tracked_tok_init(cls, **kwargs):
            trace.pause_data += 1
            try:
                orig_tok_init(cls, **kwargs)
            finally:
                trace.pause_data -= 1
            trace.instrument_class(cls)

        _TokenizerBase.__init_subclass__ = tracked_tok_init
        self.restore_hooks.append((_TokenizerBase, "__init_subclass__", classmethod(orig_tok_init)))

        # 4. Parser and Generator __init_subclass__ hooks
        for base_cls in (Parser, Generator):
            orig_sub = vars(base_cls).get("__init_subclass__")

            def make_hook(bcls, orig_desc):
                @classmethod
                def hook(cls, **kwargs):
                    if orig_desc is not None:
                        orig_desc.__func__(cls, **kwargs)
                    else:
                        super(bcls, cls).__init_subclass__(**kwargs)
                    trace.instrument_class(cls)
                return hook

            base_cls.__init_subclass__ = make_hook(base_cls, orig_sub)
            self.restore_hooks.append((base_cls, "__init_subclass__", orig_sub))

        # 5. Expr __init_subclass__, __init__, error_messages, Func.from_arg_list
        orig_expr_sub = vars(Expr)["__init_subclass__"].__func__

        @classmethod
        @wraps(orig_expr_sub)
        def tracked_expr_sub(cls, **kwargs):
            orig_expr_sub(cls, **kwargs)
            trace.instrument_expr_class(cls)

        Expr.__init_subclass__ = tracked_expr_sub
        self.restore_hooks.append((Expr, "__init_subclass__", classmethod(orig_expr_sub)))

        orig_expr_init = Expr.__init__

        @wraps(orig_expr_init)
        def tracked_expr_init(self, **args):
            trace._record_expr(type(self))
            return orig_expr_init(self, **args)

        Expr.__init__ = tracked_expr_init
        self.restore_hooks.append((Expr, "__init__", orig_expr_init))

        orig_err = Expr.error_messages

        @wraps(orig_err)
        def tracked_err(self, args=None):
            trace._record_expr(type(self))
            return orig_err(self, args)

        Expr.error_messages = tracked_err
        self.restore_hooks.append((Expr, "error_messages", orig_err))

        orig_from_arg_list = vars(Func)["from_arg_list"].__func__

        @classmethod
        @wraps(orig_from_arg_list)
        def tracked_from_arg_list(cls, args):
            trace._record_expr(cls)
            return orig_from_arg_list(cls, args)

        Func.from_arg_list = tracked_from_arg_list
        self.restore_hooks.append((Func, "from_arg_list", classmethod(orig_from_arg_list)))

        # 6. Wrap SourceFileLoader.exec_module to guard sqlglot module init & wrap typing modules
        orig_exec_module = importlib.machinery.SourceFileLoader.exec_module

        @wraps(orig_exec_module)
        def tracked_exec_module(loader, module):
            name = getattr(module, "__name__", "")
            is_sqlglot = name == "sqlglot" or name.startswith("sqlglot.")
            if is_sqlglot:
                trace.pause_data += 1
            try:
                result = orig_exec_module(loader, module)
            finally:
                if is_sqlglot:
                    trace.pause_data -= 1
            if name.startswith("sqlglot.typing."):
                trace.instrument_typing_module(module)
            return result

        importlib.machinery.SourceFileLoader.exec_module = tracked_exec_module
        self.restore_hooks.append((importlib.machinery.SourceFileLoader, "exec_module", orig_exec_module))

        # 7. Wrap subprocess.run to also trace `python -c ...` invocations (e.g. test_lazy_load)
        orig_subproc_run = subprocess.run
        self._orig_subproc_run = orig_subproc_run

        @wraps(orig_subproc_run)
        def tracked_subproc_run(cmd, *args, **kwargs):
            res = orig_subproc_run(cmd, *args, **kwargs)
            if (
                trace.active
                and isinstance(cmd, (list, tuple))
                and len(cmd) >= 3
                and os.path.basename(str(cmd[0])).startswith("python")
                and cmd[1] == "-c"
                and isinstance(cmd[2], str)
            ):
                trace._trace_python_dash_c(cmd[2], kwargs.get("cwd"), kwargs.get("env"))
            return res

        subprocess.run = tracked_subproc_run
        self.restore_hooks.append((subprocess, "run", orig_subproc_run))

    def _trace_python_dash_c(self, code_str, cwd, env):
        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tf:
            out_path = tf.name
        root_prefix = self.root.as_posix().rstrip("/") + "/"
        wrapper = (
            "import ast, json, os, sys\n"
            f"root = {root_prefix!r}\n"
            "hits = []\n"
            "defs = {}\n"
            "def rel(fn):\n"
            "    p = os.path.abspath(fn).replace('\\\\', '/')\n"
            "    return p[len(root):] if p.startswith(root) else None\n"
            "def cdefs(r):\n"
            "    if r not in defs:\n"
            "        s = set()\n"
            "        try:\n"
            "            for n in ast.walk(ast.parse(open(os.path.join(root, r)).read())):\n"
            "                if isinstance(n, ast.ClassDef): s.add((n.lineno, n.name))\n"
            "        except Exception: pass\n"
            "        defs[r] = s\n"
            "    return defs[r]\n"
            "tid = sys.monitoring.COVERAGE_ID\n"
            "sys.monitoring.use_tool_id(tid, 'sub')\n"
            "def on_s(c, o):\n"
            "    r = rel(c.co_filename)\n"
            "    if r and r.startswith(('sqlglot/', 'tests/')) and c.co_name != '<module>' and (c.co_firstlineno, c.co_name) not in cdefs(r):\n"
            "        hits.append(('S', r, c.co_firstlineno, c.co_name))\n"
            "    return sys.monitoring.DISABLE\n"
            "def on_l(c, ln):\n"
            "    r = rel(c.co_filename)\n"
            "    if r and r.startswith(('sqlglot/', 'tests/')):\n"
            "        isf = c.co_name != '<module>' and (c.co_firstlineno, c.co_name) not in cdefs(r)\n"
            "        hits.append(('L', r, ln, isf))\n"
            "    return sys.monitoring.DISABLE\n"
            "sys.monitoring.register_callback(tid, sys.monitoring.events.PY_START, on_s)\n"
            "sys.monitoring.register_callback(tid, sys.monitoring.events.LINE, on_l)\n"
            "sys.monitoring.set_events(tid, sys.monitoring.events.PY_START | sys.monitoring.events.LINE)\n"
            f"exec(compile({code_str!r}, '<string>', 'exec'), {{'__name__': '__main__'}})\n"
            "sys.monitoring.set_events(tid, 0)\n"
            f"open({out_path!r}, 'w').write(json.dumps(hits))\n"
        )
        try:
            proc = self._orig_subproc_run(
                [sys.executable, "-c", wrapper],
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0 and os.path.exists(out_path):
                ctx = self.context
                for item in json.loads(Path(out_path).read_text(encoding="utf-8")):
                    if item[0] == "S":
                        _, rel, fln, name = item
                        self.func_calls[rel][(fln, name)].add(ctx)
                    else:
                        _, rel, ln, isf = item
                        if isf:
                            self.func_lines[rel][ln].add(ctx)
                        else:
                            self.init_lines[rel][ln].add(ctx)
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass

    def dialect_modules(self, dialect):
        paths = set()
        classes = [dialect]
        for attribute in ("parser_class", "generator_class", "tokenizer_class", "jsonpath_tokenizer_class"):
            cls = getattr(dialect, attribute, None)
            if isinstance(cls, type):
                classes.append(cls)
        for cls in classes:
            for base in cls.__mro__:
                module = sys.modules.get(base.__module__)
                filename = getattr(module, "__file__", None)
                if filename:
                    path = self.relative_path(filename)
                    if path and path.startswith("sqlglot/"):
                        paths.add(path)
        return paths

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_runtest_protocol(self, item, nextitem):
        previous = self.switch(f"test:{item.nodeid}")
        self._test_start_times[item.nodeid] = time.perf_counter()
        try:
            return (yield)
        finally:
            elapsed_ms = round((time.perf_counter() - self._test_start_times.pop(item.nodeid)) * 1000)
            self.test_cpu_ms[item.nodeid] = elapsed_ms
            self.switch(previous)

    def pytest_runtest_logreport(self, report):
        if report.when == "call" and report.passed:
            self.passed.add(report.nodeid)
        if report.failed or report.skipped:
            self.unverified.add(report.nodeid)

    def fixture_context(self, fixturedef):
        if fixturedef not in self.fixture_contexts:
            self.fixture_contexts[fixturedef] = f"fixture:{len(self.fixture_contexts)}"
        return self.fixture_contexts[fixturedef]

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_fixture_setup(self, fixturedef, request):
        if fixturedef.scope == "function":
            return (yield)
        context = self.fixture_context(fixturedef)
        if self.context.startswith("test:"):
            self.fixture_users[context].add(self.context.removeprefix("test:"))
        previous = self.switch(context)
        add_finalizer = request.addfinalizer

        def add_traced_finalizer(finalizer):
            @wraps(finalizer)
            def trace_finalizer():
                previous = self.switch(context)
                try:
                    return finalizer()
                finally:
                    self.switch(previous)

            add_finalizer(trace_finalizer)

        request.addfinalizer = add_traced_finalizer
        try:
            return (yield)
        finally:
            del request.addfinalizer
            self.switch(previous)

    def restore_all_hooks(self):
        for obj, name, orig in reversed(self.restore_hooks):
            if orig is None:
                try:
                    delattr(obj, name)
                except AttributeError:
                    pass
            else:
                setattr(obj, name, orig)
        self.restore_hooks.clear()

    def _collect_static_metadata(self):
        """Import all sqlglot modules and extract class hierarchy, AST dict spreads, enums, and __eq__ metadata."""
        for sub in ("dialects", "parsers", "generators", "expressions", "typing"):
            pkg_dir = self.root / "sqlglot" / sub
            if not pkg_dir.is_dir():
                continue
            for py in sorted(pkg_dir.glob("*.py")):
                if py.name.startswith("__"):
                    continue
                mod_name = f"sqlglot.{sub}.{py.stem}"
                if mod_name not in sys.modules:
                    try:
                        importlib.import_module(mod_name)
                    except Exception:
                        pass

        from sqlglot.dialects.dialect import Dialect
        from sqlglot.expressions.core import Expr
        from sqlglot.generator import Generator
        from sqlglot.parser import Parser
        from sqlglot.tokens import _TokenizerBase

        # 1. Inspect AST of all sqlglot/*.py files for dict spreads, Enum defs, and __eq__ defs
        ast_dict_spreads = {}  # (rel_path, cls_or_mod, attr) -> (list[spread_expr_names], set[explicit_keys])
        enum_classes = {}
        custom_eq_classes = set()
        unsafe_none_eq_classes = {"reverse_key"}

        for py in sorted((self.root / "sqlglot").rglob("*.py")):
            rel = py.relative_to(self.root).as_posix()
            try:
                src = py.read_text(encoding="utf-8")
                tree = ast.parse(src, filename=rel)
            except (OSError, SyntaxError):
                continue

            def _scan_dict_literal(dict_node: ast.Dict):
                spreads = []
                explicit = set()
                for k, v in zip(dict_node.keys, dict_node.values):
                    if k is None:
                        spreads.append(ast.unparse(v))
                    else:
                        try:
                            lit = ast.literal_eval(k)
                            ck = canonical_key(lit)
                            if ck is not None:
                                explicit.add(ck)
                        except Exception:
                            pass
                return spreads, explicit

            # Top-level EXPRESSION_METADATA in typing modules
            if rel.startswith("sqlglot/typing/"):
                for stmt in tree.body:
                    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                        tgt = stmt.targets[0]
                        if isinstance(tgt, ast.Name) and tgt.id == "EXPRESSION_METADATA" and isinstance(stmt.value, ast.Dict):
                            ast_dict_spreads[(rel, "", "EXPRESSION_METADATA")] = _scan_dict_literal(stmt.value)

            def scan_class(cnode: ast.ClassDef, prefix: str):
                qual = f"{prefix}.{cnode.name}" if prefix else cnode.name
                for item in cnode.body:
                    if isinstance(item, ast.ClassDef):
                        scan_class(item, qual)
                    elif isinstance(item, ast.FunctionDef) and item.name in ("__eq__", "__ne__"):
                        custom_eq_classes.add(f"{rel}:{qual}")
                        custom_eq_classes.add(cnode.name)
                    elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                        tgt = item.targets[0] if isinstance(item, ast.Assign) and len(item.targets) == 1 else (
                            item.target if isinstance(item, ast.AnnAssign) else None
                        )
                        if isinstance(tgt, ast.Name) and isinstance(item.value, ast.Dict):
                            ast_dict_spreads[(rel, qual, tgt.id)] = _scan_dict_literal(item.value)

            for stmt in tree.body:
                if isinstance(stmt, ast.ClassDef):
                    scan_class(stmt, "")

        # 2. Collect all relevant classes
        all_classes = set()

        def gather(base):
            if self.class_key(base) is not None:
                all_classes.add(base)
            for sub in base.__subclasses__():
                gather(sub)

        for root_cls in (Dialect, _TokenizerBase, Parser, Generator, Expr):
            gather(root_cls)

        # Also collect Enum subclasses across loaded sqlglot modules
        def gather_enums(base):
            ckey = self.class_key(base)
            if ckey is not None and base is not enum.Enum:
                members = list(base.__members__.keys())
                values = [m.value for m in base.__members__.values()]
                try:
                    distinct = len(set(values)) == len(values)
                except TypeError:
                    distinct = len(values) == len(members)
                is_pure = not issubclass(base, (str, int, float, bytes))
                info = {
                    "key": ckey,
                    "name": base.__name__,
                    "is_pure": is_pure,
                    "distinct_values": distinct,
                    "members": members,
                }
                enum_classes[ckey] = info
            for sub in base.__subclasses__():
                gather_enums(sub)

        gather_enums(enum.Enum)

        # Resolve safe_none_eq_classes: all known sqlglot classes without custom __eq__ (or safe custom __eq__)
        # Note: Expr defines __eq__ (structural AST equality) and Dialect defines __eq__, both compare False against None!
        # Only reverse_key throws AttributeError on `other.obj == self.obj`!
        safe_none_eq_classes = set()
        for mod in list(sys.modules.values()):
            mname = getattr(mod, "__name__", "")
            if not mname.startswith("sqlglot"):
                continue
            for val in list(vars(mod).values()):
                if isinstance(val, type) and getattr(val, "__module__", "").startswith("sqlglot"):
                    if val.__name__ not in unsafe_none_eq_classes:
                        safe_none_eq_classes.add(val.__name__)

        # 3. Build class_hierarchy
        key_of = {cls: self.class_key(cls) for cls in all_classes}
        # Map short/module ref strings to keys for spread resolution
        typing_mod_by_var = {}
        for mod in list(sys.modules.values()):
            mname = getattr(mod, "__name__", "")
            if mname.startswith("sqlglot.typing."):
                rel = self.relative_path(getattr(mod, "__file__", ""))
                if rel:
                    typing_mod_by_var[mname] = rel

        class_hierarchy = {}
        for cls in all_classes:
            ckey = key_of[cls]
            rel, qual = ckey.split(":", 1)
            subs = [key_of[s] for s in cls.__subclasses__() if s in key_of]
            bases = [key_of[b] for b in cls.__bases__ if b in key_of]

            own_attrs = set()
            own_dict_keys = {}
            dict_spreads = {}
            tracked_dict_attrs = set()
            tracked_scalar_attrs = set()

            for name, raw in cls.__dict__.items():
                if name.startswith("__"):
                    continue
                if isinstance(raw, _TrackedDict):
                    tracked_dict_attrs.add(name)
                    keys_set = {ck for k in raw.keys() if (ck := canonical_key(k)) is not None}
                    own_dict_keys[name] = keys_set
                    spread_info = ast_dict_spreads.get((rel, qual, name))
                    if spread_info:
                        spread_exprs, explicit_keys = spread_info
                        resolved_spreads = []
                        for sexpr in spread_exprs:
                            # e.g. "MySQLParser.SHOW_PARSERS", "parser.Parser.FUNCTIONS", "EXPRESSION_METADATA"
                            if "." in sexpr:
                                owner_expr, sattr = sexpr.rsplit(".", 1)
                                resolved_cls = None
                                for mro_cls in cls.__mro__:
                                    if mro_cls in key_of and mro_cls.__name__ == owner_expr.rsplit(".", 1)[-1]:
                                        resolved_cls = key_of[mro_cls]
                                        break
                                if resolved_cls:
                                    resolved_spreads.append((resolved_cls, sattr))
                            elif sexpr == "EXPRESSION_METADATA" and name == "EXPRESSION_METADATA":
                                mod_file = getattr(sys.modules.get(cls.__module__), "__file__", None)
                                mod_rel = self.relative_path(mod_file) if mod_file else None
                                added_typing = False
                                if mod_rel:
                                    try:
                                        mtree = ast.parse((self.root / mod_rel).read_text(encoding="utf-8"))
                                        for mnode in ast.walk(mtree):
                                            if (
                                                isinstance(mnode, ast.ImportFrom)
                                                and mnode.module
                                                and mnode.module.startswith("sqlglot.typing")
                                                and any(al.name == "EXPRESSION_METADATA" for al in mnode.names)
                                            ):
                                                trel = typing_mod_by_var.get(mnode.module)
                                                if trel:
                                                    resolved_spreads.append((trel, "EXPRESSION_METADATA"))
                                                    added_typing = True
                                    except (OSError, SyntaxError):
                                        pass
                                if not added_typing:
                                    for b in cls.__bases__:
                                        if b in key_of:
                                            resolved_spreads.append((key_of[b], "EXPRESSION_METADATA"))
                        if resolved_spreads or spread_exprs:
                            dict_spreads[name] = {
                                "spreads": resolved_spreads,
                                "unresolved": [s for s in spread_exprs] if not resolved_spreads else [],
                                "explicit": explicit_keys,
                            }
                elif type(raw) is dict:
                    own_dict_keys[name] = {ck for k in raw.keys() if (ck := canonical_key(k)) is not None}
                elif isinstance(raw, _TrackedAttr):
                    own_attrs.add(name)
                    tracked_scalar_attrs.add(name)
                elif not name.startswith("_") and isinstance(raw, SCALAR_TYPES):
                    own_attrs.add(name)

            class_hierarchy[ckey] = {
                "bases": bases,
                "subclasses": subs,
                "own_attrs": sorted(own_attrs),
                "own_dict_keys": {k: sorted(v) for k, v in own_dict_keys.items()},
                "dict_spreads": dict_spreads,
                "tracked_dicts": sorted(tracked_dict_attrs),
                "tracked_scalars": sorted(tracked_scalar_attrs),
            }

        # Typing module spreads (e.g. sqlglot/typing/snowflake.py spreading **PARENT_EXPRESSION_METADATA)
        typing_spreads = {}
        for mod in list(sys.modules.values()):
            mname = getattr(mod, "__name__", "")
            if not mname.startswith("sqlglot.typing."):
                continue
            rel = self.relative_path(getattr(mod, "__file__", ""))
            if not rel:
                continue
            spread_info = ast_dict_spreads.get((rel, "", "EXPRESSION_METADATA"))
            parents = []
            explicit = []
            if spread_info:
                spread_exprs, explicit_set = spread_info
                explicit = sorted(explicit_set)
                # Find imported typing modules in mod
                try:
                    tree = ast.parse((self.root / rel).read_text(encoding="utf-8"))
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("sqlglot.typing."):
                            for al in node.names:
                                if al.name == "EXPRESSION_METADATA":
                                    prel = typing_mod_by_var.get(node.module)
                                    if prel:
                                        parents.append((prel, "EXPRESSION_METADATA"))
                except (OSError, SyntaxError):
                    pass
            typing_spreads[rel] = {"parents": parents, "explicit": explicit}

        # 4. Compute enum_typed_class_attrs (e.g. Dialect.NORMALIZATION_STRATEGY -> NormalizationStrategy)
        enum_typed_class_attrs = {}
        for root_cls in (Dialect, _TokenizerBase, Parser, Generator):
            # Collect all subclasses including root
            group = []

            def collect_grp(c):
                if c in key_of:
                    group.append(c)
                for s in c.__subclasses__():
                    collect_grp(s)

            collect_grp(root_cls)
            attr_enums = defaultdict(set)
            for c in group:
                for name, raw in c.__dict__.items():
                    if name.startswith("_"):
                        continue
                    val = raw._val if isinstance(raw, _TrackedAttr) else raw
                    if isinstance(val, enum.Enum):
                        ekey = self.class_key(type(val))
                        if ekey:
                            attr_enums[name].add(ekey)
                    elif isinstance(val, SCALAR_TYPES) and not isinstance(val, enum.Enum):
                        attr_enums[name].add("__NON_ENUM__")
            for name, ekeys in attr_enums.items():
                if len(ekeys) == 1:
                    only = next(iter(ekeys))
                    if only != "__NON_ENUM__":
                        for c in group:
                            enum_typed_class_attrs[(key_of[c], name)] = only

        return {
            "class_hierarchy": class_hierarchy,
            "typing_spreads": typing_spreads,
            "enum_classes": enum_classes,
            "enum_typed_class_attrs": enum_typed_class_attrs,
            "unsafe_none_eq_classes": sorted(unsafe_none_eq_classes),
            "safe_none_eq_classes": sorted(safe_none_eq_classes),
        }

    def export(self):
        tests = sorted(self.collected)
        indexes = {nodeid: i for i, nodeid in enumerate(tests)}
        runtime_contexts = {f"test:{test}": (1 << i) for test, i in indexes.items()}
        for context, users in self.fixture_users.items():
            mask = 0
            for test in users:
                idx = indexes.get(test)
                if idx is not None:
                    mask |= (1 << idx)
            runtime_contexts[context] = mask

        def to_masks(contexts):
            mask = 0
            collecting = set()
            for ctx in contexts:
                if ctx in runtime_contexts:
                    mask |= runtime_contexts[ctx]
                elif not ctx or ctx.startswith("collection:"):
                    collecting.add(ctx.removeprefix("collection:") or "*")
            return mask, sorted(collecting)

        def convert_line_map(src_map):
            out = {}
            for path, lines_dict in sorted(src_map.items()):
                lm = {}
                for ln, ctxs in sorted(lines_dict.items()):
                    m, _ = to_masks(ctxs)
                    if m:
                        lm[ln] = m
                if lm:
                    out[path] = lm
            return out

        def convert_func_map(src_map):
            out = {}
            for path, fdict in sorted(src_map.items()):
                fm = {}
                for fk, ctxs in sorted(fdict.items()):
                    m, _ = to_masks(ctxs)
                    if m:
                        fm[fk] = m
                if fm:
                    out[path] = fm
            return out

        def convert_collection_line_map(src_map):
            out = {}
            for path, lines_dict in sorted(src_map.items()):
                lm = {}
                for ln, ctxs in sorted(lines_dict.items()):
                    _, col = to_masks(ctxs)
                    if col:
                        lm[ln] = col
                if lm:
                    out[path] = lm
            return out

        def convert_collection_func_map(src_map):
            out = {}
            for path, fdict in sorted(src_map.items()):
                fm = {}
                for fk, ctxs in sorted(fdict.items()):
                    _, col = to_masks(ctxs)
                    if col:
                        fm[fk] = col
                if fm:
                    out[path] = fm
            return out

        resources = {}
        for path, contexts in sorted(self.resources.items()):
            m, col = to_masks(contexts)
            if m or col:
                resources[path] = {"mask": m, "collection": col}

        module_usage = {}
        for path, contexts in sorted(self.module_usage.items()):
            m, _ = to_masks(contexts)
            if m:
                module_usage[path] = m

        dict_key_usage = {}
        for k, contexts in self.dict_key_usage.items():
            m, _ = to_masks(contexts)
            if m:
                dict_key_usage[k] = m

        dict_all_usage = {}
        for k, contexts in self.dict_all_usage.items():
            m, _ = to_masks(contexts)
            if m:
                dict_all_usage[k] = m

        attr_usage = {}
        for k, contexts in self.attr_usage.items():
            m, _ = to_masks(contexts)
            if m:
                attr_usage[k] = m

        expr_usage = {}
        for k, contexts in self.expr_usage.items():
            m, _ = to_masks(contexts)
            if m:
                expr_usage[k] = m

        unverified_mask = 0
        for test in (self.collected - self.passed) | (self.unverified & self.collected):
            unverified_mask |= (1 << indexes[test])

        static_meta = self._collect_static_metadata()
        self.restore_all_hooks()

        return {
            "tests": tests,
            "func_lines": convert_line_map(self.func_lines),
            "func_calls": convert_func_map(self.func_calls),
            "init_lines": convert_line_map(self.init_lines),
            "init_collection": convert_collection_line_map(self.init_collection),
            "func_calls_collection": convert_collection_func_map(self.func_calls_collection),
            "resources": resources,
            "module_usage": module_usage,
            "dict_key_usage": dict_key_usage,
            "dict_all_usage": dict_all_usage,
            "attr_usage": attr_usage,
            "expr_usage": expr_usage,
            "unverified_mask": unverified_mask,
            "collection_errors": sorted(self.collection_errors),
            "test_cpu_ms": [self.test_cpu_ms.get(t, 0) for t in tests],
            **static_meta,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("test_paths", nargs="*", default=["tests"])
    args = parser.parse_args()

    root = Path.cwd()
    if not (root / "sqlglot" / "__init__.py").is_file():
        parser.error("run from the SQLGlot checkout")
    os.environ.setdefault("SKIP_INTEGRATION", "0")

    trace = CoverageIndex(root)
    original_pool = concurrent.futures.ProcessPoolExecutor
    concurrent.futures.ProcessPoolExecutor = _InlinePoolExecutor
    sys.addaudithook(trace.audit_open)
    trace.start_monitoring()
    try:
        pytest_args = ["-q", "--continue-on-collection-errors", "--tb=short", "-p", "no:cacheprovider"]
        if sys.platform != "win32":
            pytest_args += ["-o", "timeout=240", "-o", "timeout_method=signal"]
        status = pytest.main([*pytest_args, *args.test_paths], plugins=[trace])
    finally:
        trace.active = False
        trace.stop_monitoring()
        concurrent.futures.ProcessPoolExecutor = original_pool

    if status not in (pytest.ExitCode.OK, pytest.ExitCode.TESTS_FAILED) or not trace.passed:
        trace.restore_all_hooks()
        raise SystemExit(f"Could not build coverage from pytest exit status {status}")

    index = trace.export()
    args.output.write_bytes(zlib.compress(pickle.dumps(index, protocol=pickle.HIGHEST_PROTOCOL), level=6))
    unverified_cnt = index["unverified_mask"].bit_count()
    print(
        f"Indexed {len(index['tests'])} tests across {len(index['func_lines'])} files "
        f"({unverified_cnt} unverified, {len(index['collection_errors'])} collection errors)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
