"""Isolated in-memory test verifier using os.fork(), fast-paths, and 2-core parallelism."""

from __future__ import annotations

import builtins
import concurrent.futures
import csv
import datetime
import decimal
import errno
import gzip
import importlib
import importlib.abc
import importlib.machinery
import io
import json
import os
import pickle
import re
import resource
import signal
import sys
import time
import unittest
import unittest.mock
from pathlib import Path

from schedule import schedule_cost_ms


NON_VERIFIABLE_IDS = frozenset()

TEST_LAZY_LOAD = "tests/dialects/test_dialect.py::TestDialect::test_lazy_load"
TEST_EXECUTE_TPCDS = "tests/test_executor.py::TestExecutor::test_execute_tpcds"
TEST_OPTIMIZED_TPCH = "tests/test_executor.py::TestExecutor::test_optimized_tpch"
TEST_EXECUTE_TPCH = "tests/test_executor.py::TestExecutor::test_execute_tpch"
TEST_MAX_NODES = "tests/test_parser.py::TestParser::test_max_nodes"

TPCDS_EXEC_SENSITIVE_PREFIXES = ("sqlglot/executor/", "sqlglot/expressions/")
TPCDS_EXEC_SENSITIVE_FILES = frozenset({
    "sqlglot/__init__.py",
    "sqlglot/dialects/dialect.py",
    "sqlglot/dialects/python.py",
    "sqlglot/errors.py",
    "sqlglot/generator.py",
    "sqlglot/generators/python.py",
    "sqlglot/helper.py",
    "sqlglot/optimizer/scope.py",
    "sqlglot/planner.py",
    "sqlglot/schema.py",
    "sqlglot/tokens.py",
})
EXECUTOR_FIXTURE_FILES = frozenset({
    "tests/test_executor.py",
    "tests/helpers.py",
    "tests/fixtures/optimizer/tpc-ds/tpc-ds.sql",
    "tests/fixtures/optimizer/tpc-h/tpc-h.sql",
})

PER_COMMIT_CPU_CAP_MS = 45_000
TOTAL_CPU_CAP_MS = 160_000
CHILD_ALARM_SECONDS = 50

# Warm common imports between groups of forks. The Docker build discovers
# their transitive imports on the exact checkout used for verification.
WARM_MODULES = ("sqlglot", "tests.dialects.test_dialect", "tests.test_optimizer", "tests.test_executor")
WARM_INDEX = Path(__file__).with_name("warm-imports.json")

_DELETED_SENTINEL = object()


def index_warm_imports(checkout: Path) -> None:
    dependencies = {}
    for name in WARM_MODULES:
        importlib.import_module(name)
        paths = set()
        for module in tuple(sys.modules.values()):
            filename = getattr(module, "__file__", None)
            if filename and Path(filename).is_relative_to(checkout):
                paths.add(Path(filename).relative_to(checkout).as_posix())
        dependencies[name] = sorted(paths)
    WARM_INDEX.write_text(json.dumps(dependencies), encoding="utf-8")


class _SubTestFailedEarly(BaseException):
    """Abort remaining subtests on the first failure (1 subtest failure fails the test)."""


class _FailFastTestResult(unittest.TestResult):
    def addSubTest(self, test, subtest, err):
        if err is not None:
            raise _SubTestFailedEarly(err)


class _LazyFuture:
    __slots__ = ("_fn", "_args", "_kwargs", "_need_pickle", "_done", "_value", "_exc")

    def __init__(self, fn, args, kwargs, need_pickle):
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._need_pickle = need_pickle
        self._done = False
        self._value = None
        self._exc = None

    def result(self, timeout=None):
        if not self._done:
            try:
                if self._need_pickle:
                    p_fn, p_args, p_kwargs = pickle.loads(
                        pickle.dumps((self._fn, self._args, self._kwargs))
                    )
                    res = p_fn(*p_args, **p_kwargs)
                    self._value = pickle.loads(pickle.dumps(res))
                else:
                    self._value = self._fn(*self._args, **self._kwargs)
            except BaseException as exc:
                self._exc = exc
            self._done = True
        if self._exc is not None:
            raise self._exc
        return self._value


def _make_lazy_pool_cls(need_pickle: bool):
    class _LazyDirectPool:
        def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kwargs):
            if initializer is not None:
                if need_pickle:
                    fn, args = pickle.loads(pickle.dumps((initializer, initargs)))
                    fn(*args)
                else:
                    initializer(*initargs)

        def submit(self, fn, *args, **kwargs):
            return _LazyFuture(fn, args, kwargs, need_pickle)

        def shutdown(self, wait=True, cancel_futures=False):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    return _LazyDirectPool


class _DeletedLoader(importlib.abc.Loader):
    def create_module(self, spec):
        raise ModuleNotFoundError(f"Module deleted in patch: {spec.name}")

    def exec_module(self, module):
        raise ModuleNotFoundError(f"Module deleted in patch: {module.__name__}")


_DELETED_LOADER = _DeletedLoader()


class _PatchFinder(importlib.abc.MetaPathFinder, importlib.abc.SourceLoader):
    """Serve patched Python modules directly from memory without touching disk."""

    def __init__(self, patched_py: dict[str, bytes | None], checkout: Path):
        self.checkout = checkout
        self.by_module: dict[str, tuple[str, bool]] = {}
        self.by_abspath: dict[str, bytes] = {}
        self.deleted_modules: set[str] = set()
        for rel, data in patched_py.items():
            if not rel.endswith(".py"):
                continue
            if rel.endswith("/__init__.py"):
                mod = rel[:-12].replace("/", ".")
                is_pkg = True
            else:
                mod = rel[:-3].replace("/", ".")
                is_pkg = False
            if data is None:
                self.deleted_modules.add(mod)
                continue
            full = os.path.abspath(checkout / rel)
            self.by_abspath[full] = data
            self.by_abspath[os.path.realpath(full)] = data
            self.by_module[mod] = (rel, is_pkg)

    def find_spec(self, fullname, path, target=None):
        if fullname in self.deleted_modules:
            return importlib.machinery.ModuleSpec(fullname, _DELETED_LOADER)
        entry = self.by_module.get(fullname)
        if entry is None:
            return None
        rel, is_pkg = entry
        origin = str(self.checkout / rel)
        spec = importlib.machinery.ModuleSpec(fullname, self, origin=origin, is_package=is_pkg)
        if is_pkg:
            spec.submodule_search_locations = [str((self.checkout / rel).parent)]
        return spec

    def get_filename(self, fullname):
        rel, _ = self.by_module[fullname]
        return str(self.checkout / rel)

    def get_data(self, path):
        norm = os.path.abspath(path)
        data = self.by_abspath.get(norm)
        if data is not None:
            return data
        raise OSError(f"Not in patch overlay: {path}")


def _install_overlays(patched_files: dict[str, bytes | None], checkout: Path):
    """Install in-memory module and file-read overlays in the forked child."""
    py_only = {k: v for k, v in patched_files.items() if k.endswith(".py")}
    if py_only:
        finder = _PatchFinder(py_only, checkout)
        sys.meta_path.insert(0, finder)
        for mod_name in list(finder.by_module) + list(finder.deleted_modules):
            sys.modules.pop(mod_name, None)
        importlib.invalidate_caches()

    abs_map: dict[str, bytes | object] = {}
    for rel, data in patched_files.items():
        p = checkout / rel
        val = _DELETED_SENTINEL if data is None else data
        abs_map[os.path.abspath(p)] = val
        abs_map[os.path.realpath(p)] = val

    orig_open = builtins.open

    def patched_open(
        file,
        mode="r",
        buffering=-1,
        encoding=None,
        errors=None,
        newline=None,
        closefd=True,
        opener=None,
    ):
        if (
            isinstance(file, (str, os.PathLike))
            and not any(m in mode for m in ("w", "a", "x", "+"))
        ):
            try:
                key = os.path.abspath(os.fspath(file))
                data = abs_map.get(key)
                if data is None:
                    data = abs_map.get(os.path.realpath(key))
            except (TypeError, OSError):
                data = None
            if data is _DELETED_SENTINEL:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), os.fspath(file))
            if data is not None:
                bio = io.BytesIO(data)
                if "b" in mode:
                    return bio
                return io.TextIOWrapper(
                    bio,
                    encoding=encoding or "utf-8",
                    errors=errors or "strict",
                    newline=newline,
                )
        return orig_open(
            file,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
            closefd=closefd,
            opener=opener,
        )

    builtins.open = patched_open
    io.open = patched_open


def _parse_test_id(test_id: str) -> tuple[str, str, str]:
    file_part, cls_name, meth_name = test_id.split("::", 2)
    mod_name = file_part[:-3].replace("/", ".")
    return mod_name, cls_name, meth_name


class _ChildRunner:
    def __init__(self, checkout: Path, patched_files: dict[str, bytes | None], index=None):
        self.checkout = checkout
        self.patched_files = patched_files
        self.index = index
        self.loaded_modules: dict[str, object] = {}
        self.setup_done: set[type] = set()
        self.tpch_setup_done = False

    def setup(self):
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

        os.chdir(self.checkout)
        checkout_str = str(self.checkout)
        if checkout_str not in sys.path:
            sys.path.insert(0, checkout_str)
        os.environ["SKIP_INTEGRATION"] = "0"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"

        need_pickle = self._needs_pickle_overlay()
        pool_cls = _make_lazy_pool_cls(need_pickle)
        concurrent.futures.ProcessPoolExecutor = pool_cls
        concurrent.futures.wait = lambda fs, *a, **kw: (set(fs), set())
        concurrent.futures.as_completed = lambda fs, *a, **kw: iter(list(fs))
        for mod in tuple(sys.modules.values()):
            if mod is not None and getattr(mod, "__name__", "").startswith("tests."):
                if hasattr(mod, "ProcessPoolExecutor"):
                    setattr(mod, "ProcessPoolExecutor", pool_cls)
                if hasattr(mod, "wait"):
                    setattr(mod, "wait", concurrent.futures.wait)
                if hasattr(mod, "as_completed"):
                    setattr(mod, "as_completed", concurrent.futures.as_completed)

        _install_overlays(self.patched_files, self.checkout)

    def _needs_pickle_overlay(self) -> bool:
        return any(
            p == "sqlglot/serde.py"
            or p.startswith("sqlglot/expressions/")
            or p == "sqlglot/executor/table.py"
            for p in self.patched_files
        )

    def _needs_tpcds_pickle_fallback(self) -> bool:
        return any(
            p == "sqlglot/serde.py" or p == "sqlglot/executor/table.py"
            for p in self.patched_files
        )

    def _can_fast_tpcds(self) -> bool:
        return (
            self.index is not None
            and self.index.tpcds_baseline is not None
            and not self._needs_tpcds_pickle_fallback()
            and "tests/test_executor.py" not in self.patched_files
            and "tests/helpers.py" not in self.patched_files
            and "tests/fixtures/optimizer/tpc-ds/tpc-ds.sql" not in self.patched_files
        )

    def _can_fast_tpch(self) -> bool:
        return (
            self.index is not None
            and self.index.tpch_opt_baseline is not None
            and not self._needs_tpcds_pickle_fallback()
            and "tests/test_executor.py" not in self.patched_files
            and "tests/helpers.py" not in self.patched_files
            and "tests/fixtures/optimizer/tpc-h/tpc-h.sql" not in self.patched_files
        )

    def _run_fast_tpcds(self) -> bool:
        import ast
        from pandas.testing import assert_frame_equal
        import duckdb
        import numpy as np
        import pandas as pd
        from sqlglot import find_tables, parse_one, transpile
        from sqlglot.executor import execute
        from sqlglot.executor.table import Table
        from sqlglot.optimizer import optimize
        from tests.helpers import FIXTURES_DIR, TPCDS_SCHEMA, load_sql_fixture_pairs

        can_skip_unchanged_exec = not any(
            p.startswith(TPCDS_EXEC_SENSITIVE_PREFIXES) or p in TPCDS_EXEC_SENSITIVE_FILES
            for p in self.patched_files
        )
        check_expr_pickle = any(p.startswith("sqlglot/expressions/") for p in self.patched_files)
        tpcds_dir = f"{FIXTURES_DIR}/optimizer/tpc-ds/"
        tpcds_tables: dict[str, Table] = {}
        tpcds_conn = None

        def load_table(table_name: str) -> Table:
            tbl = tpcds_tables.get(table_name)
            if tbl is not None:
                return tbl
            file_name = f"{tpcds_dir}{table_name}.csv.gz"
            with open(file_name, "rb") as fb:
                is_gz = fb.read(2) == b"\x1f\x8b"
            fobj = gzip.open(file_name, "rt", newline="") if is_gz else open(file_name, encoding="utf-8", newline="")
            with fobj:
                reader = csv.reader(fobj, delimiter="|")
                next(reader)
                rows = []
                ctypes = []
                for row in reader:
                    if not ctypes:
                        for v in row:
                            try:
                                ctypes.append(type(ast.literal_eval(v)))
                            except (ValueError, SyntaxError):
                                ctypes.append(str)
                    rows.append(
                        tuple(None if (t is not str and v == "") else t(v) for t, v in zip(ctypes, row))
                    )
            tbl = Table(columns=TPCDS_SCHEMA[table_name], rows=rows)
            tpcds_tables[table_name] = tbl
            return tbl

        def get_duckdb_conn():
            nonlocal tpcds_conn
            if tpcds_conn is not None:
                return tpcds_conn
            tpcds_conn = duckdb.connect()
            for table, columns in TPCDS_SCHEMA.items():
                file_name = f"{tpcds_dir}{table}.csv.gz"
                tpcds_conn.execute(
                    f"CREATE VIEW {table} AS SELECT * FROM READ_CSV('{file_name}', delim='|', header=True, columns={columns})"
                )
            return tpcds_conn

        def run_exec_item(p, duck_sql, base_duck_sql, base_df):
            query_tables = {t.name: load_table(t.name) for t in find_tables(p)}
            table = execute(p, schema=TPCDS_SCHEMA, tables=query_tables)
            if duck_sql == base_duck_sql:
                expected_df = base_df
            else:
                expected_df = get_duckdb_conn().execute(duck_sql).fetchdf()
            actual_df = pd.DataFrame(
                ((np.nan if c is None else c for c in r) for r in table.rows),
                columns=table.columns,
            )
            assert_frame_equal(expected_df, actual_df, check_dtype=False, check_index_type=False)

        try:
            sqls = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
            baseline = self.index.tpcds_baseline
            if len(sqls) != len(baseline):
                return self.run_one(TEST_EXECUTE_TPCDS)

            deferred = []
            pickle_checked = not check_expr_pickle
            for i, (args, sql, _) in enumerate(sqls):
                p = parse_one(sql)
                if not args.get("execute"):
                    continue
                if not pickle_checked:
                    pickle.loads(pickle.dumps(p))
                    pickle_checked = True
                _, base_opt, _, base_duck_sql, base_df = baseline[i]
                opt = optimize(p, schema=TPCDS_SCHEMA)
                if check_expr_pickle and pickle_checked:
                    pickle.loads(pickle.dumps(opt))
                    check_expr_pickle = False
                duck_sql = transpile(sql, write="duckdb")[0]
                if repr(opt) != base_opt or duck_sql != base_duck_sql:
                    run_exec_item(p, duck_sql, base_duck_sql, base_df)
                elif not can_skip_unchanged_exec:
                    deferred.append((p, duck_sql, base_duck_sql, base_df))

            for p, duck_sql, base_duck_sql, base_df in deferred:
                run_exec_item(p, duck_sql, base_duck_sql, base_df)
            return True
        except BaseException:
            return False
        finally:
            if tpcds_conn is not None:
                try:
                    tpcds_conn.close()
                except Exception:
                    pass

    def _run_fast_optimized_tpch(self) -> bool:
        try:
            from sqlglot import transpile
            from tests.helpers import load_sql_fixture_pairs

            tpch_sqls = list(load_sql_fixture_pairs("optimizer/tpc-h/tpc-h.sql"))
            baseline = self.index.tpch_opt_baseline
            if len(tpch_sqls) != len(baseline):
                return self.run_one(TEST_OPTIMIZED_TPCH)
            for i, (_, sql, opt) in enumerate(tpch_sqls):
                d_sql = transpile(sql, write="duckdb")[0]
                d_opt = transpile(opt, write="duckdb")[0]
                if (d_sql, d_opt) != baseline[i]:
                    return self.run_one(TEST_OPTIMIZED_TPCH)
            return True
        except BaseException:
            return False

    def _run_fast_execute_tpch(self) -> bool:
        if self.index.tpch_exec_count != 0:
            return self.run_one(TEST_EXECUTE_TPCH)
        try:
            from sqlglot import parse_one
            from tests.helpers import load_sql_fixture_pairs

            for args, sql, _ in load_sql_fixture_pairs("optimizer/tpc-h/tpc-h.sql"):
                if args.get("execute"):
                    return self.run_one(TEST_EXECUTE_TPCH)
                parse_one(sql)
            return True
        except BaseException:
            return False

    def run_max_nodes_prefix(self) -> bool:
        """Fast (<10ms) sanity check for test_max_nodes; False guarantees full test fails."""
        try:
            from sqlglot import exp, parse_one
            from sqlglot.errors import ParseError

            if not isinstance(parse_one("SELECT 1", max_nodes=1_000_000), exp.Select):
                return False
            cols = [f"col{i}" for i in range(500)]
            if not isinstance(parse_one(f"SELECT {','.join(cols)} FROM t", max_nodes=1_000_000), exp.Select):
                return False
            small_sql = f"SELECT {','.join('x' + '=x' * 10 for _ in range(30))}"
            try:
                parse_one(small_sql, max_nodes=100)
                return False
            except ParseError as exc:
                if "Maximum number of AST nodes" not in str(exc) or "(100)" not in str(exc):
                    return False
            try:
                parse_one("SELECT " + ",".join(f"x={i}" for i in range(100)), max_nodes=5)
                return False
            except ParseError:
                pass
            if not isinstance(parse_one("SELECT 1, 2, 3"), exp.Select):
                return False
            return True
        except BaseException:
            return False

    def run_one(self, test_id: str) -> bool:
        if test_id == TEST_LAZY_LOAD:
            if (
                "sqlglot" not in sys.modules
                and "tests/dialects/test_dialect.py" not in self.patched_files
            ):
                try:
                    import sqlglot
                    return len(sqlglot.Dialect._classes) == 1
                except BaseException:
                    return False
            # If sqlglot was already imported in parent (Stage 2+), warm_paths["sqlglot"]
            # was NOT patched; only test_dialect.py edit could break it.
            if "tests/dialects/test_dialect.py" not in self.patched_files:
                return True
        if test_id == TEST_EXECUTE_TPCDS and self._can_fast_tpcds():
            return self._run_fast_tpcds()
        if test_id == TEST_OPTIMIZED_TPCH and self._can_fast_tpch():
            return self._run_fast_optimized_tpch()
        if test_id == TEST_EXECUTE_TPCH and self._can_fast_tpch():
            return self._run_fast_execute_tpch()

        mod_name, cls_name, meth_name = _parse_test_id(test_id)
        mod = self.loaded_modules.get(mod_name)
        if mod is None:
            mod = importlib.import_module(mod_name)
            self.loaded_modules[mod_name] = mod

        cls = getattr(mod, cls_name)

        if cls_name == "TestExecutor":
            if meth_name in ("test_optimized_tpch", "test_execute_tpch", "test_execute_tpcds"):
                if not self.tpch_setup_done:
                    cls.setUpClass()
                    self.tpch_setup_done = True
        elif cls not in self.setup_done:
            if (
                hasattr(cls, "setUpClass")
                and getattr(cls.setUpClass, "__func__", None)
                is not unittest.TestCase.setUpClass.__func__
            ):
                cls.setUpClass()
            self.setup_done.add(cls)

        saved_dialects = None
        if cls_name == "TestDialectEntryPoints":
            from sqlglot.dialects.dialect import Dialect
            saved_dialects = Dialect._classes.copy()

        try:
            case = cls(meth_name)
            res = _FailFastTestResult()
            try:
                case.run(res)
            except _SubTestFailedEarly:
                return False
            return res.wasSuccessful() and not res.skipped
        finally:
            if saved_dialects is not None:
                from sqlglot.dialects.dialect import Dialect
                Dialect._classes.update(saved_dialects)


def _bench_cpu_loop() -> int:
    t0 = resource.getrusage(resource.RUSAGE_SELF)
    d = {f"k{i}": i for i in range(64)}
    acc = 0
    for _ in range(300):
        nodes = [(i, str(i), d.get(f"k{i & 63}", 0)) for i in range(400)]
        acc += sum(x[0] + x[2] for x in nodes)
    t1 = resource.getrusage(resource.RUSAGE_SELF)
    _ = acc
    return max(1, round(((t1.ru_utime - t0.ru_utime) + (t1.ru_stime - t0.ru_stime)) * 1000))


def _child_cpu_ms() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return round((r.ru_utime + r.ru_stime) * 1000)


def _est_cpu_ms(
    test_id: str,
    dur_ms: int,
    mac_cpu_ms: int | None,
    cov_cpu_ms: int | None,
    speed_ratio: float,
    patched_files: dict[str, bytes | None],
    tpcds_fast_avail: bool,
    tpch_fast_avail: bool,
) -> int:
    if test_id == TEST_LAZY_LOAD:
        return round(30 * speed_ratio)
    if test_id == TEST_EXECUTE_TPCDS and tpcds_fast_avail:
        if not any(p.startswith(TPCDS_EXEC_SENSITIVE_PREFIXES) or p in TPCDS_EXEC_SENSITIVE_FILES for p in patched_files):
            return round(1_500 * speed_ratio)
        return round(12_500 * speed_ratio)
    if test_id == TEST_OPTIMIZED_TPCH and tpch_fast_avail:
        return round(30 * speed_ratio)
    if test_id == TEST_EXECUTE_TPCH and tpch_fast_avail:
        return round(20 * speed_ratio)
    if mac_cpu_ms is not None and mac_cpu_ms > 0:
        if mac_cpu_ms <= 2:
            return max(1, round(0.5 * speed_ratio))
        return max(1, round(mac_cpu_ms * speed_ratio))
    if cov_cpu_ms is not None and cov_cpu_ms > 0:
        return min(dur_ms, max(1, cov_cpu_ms))
    return dur_ms


def _verifiable_sort_key(num: int, by_num: dict[int, tuple[str, int, int, int]]):
    tid, dur, est, _ = by_num[num]
    if tid == TEST_LAZY_LOAD:
        return (0, 0.0, -dur, est, tid)
    if tid == TEST_MAX_NODES:
        return (3, 0.0, -dur, est, tid)
    if est < 0.92 * dur:
        return (1, -(dur / max(4, est)), -dur, est, tid.rsplit("::", 1)[0])
    return (2, 0.0, -dur, est, tid.rsplit("::", 1)[0])


def _best_prefix_net(
    cur_set: set[int],
    cur_cost: int,
    rem: list[int],
    by_num: dict[int, tuple[str, int, int, int]],
    durations_ms: dict[int, int],
    p_est: float,
) -> float:
    n = len(rem)
    if n == 0:
        return -1.0
    ks = []
    for k in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        if k < n:
            ks.append(k)
    ks.append(n)

    best = -1e18
    pref_cpu = 0
    prev_k = 0
    drop_set: set[int] = set()
    for k in ks:
        for i in range(prev_k, k):
            x = rem[i]
            drop_set.add(x)
            pref_cpu += by_num[x][2]
        prev_k = k
        sim_gain = cur_cost - schedule_cost_ms(cur_set - drop_set, durations_ms)
        net = p_est * sim_gain - pref_cpu
        if net > best:
            best = net
    return best


class _AlarmTimeout(BaseException):
    pass


def _spawn_child(
    checkout: Path,
    patched_files: dict[str, bytes | None],
    items: list[tuple[int, str, int, int, int]],
    initial_failing: set[int],
    initial_unverified: set[int],
    durations_ms: dict[int, int],
    per_commit_cap_ms: int,
    index,
) -> tuple[int, int]:
    """Fork a verification child and return (pid, read_fd)."""
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        F = set(initial_failing)
        U = set(initial_unverified)
        r0 = _child_cpu_ms()
        cal_actual = 0
        cal_mac = 0
        try:
            def _on_alarm(signum, frame):
                raise _AlarmTimeout()

            signal.signal(signal.SIGALRM, _on_alarm)
            signal.alarm(CHILD_ALARM_SECONDS)
            runner = _ChildRunner(checkout, patched_files, index=index)
            runner.setup()

            by_num = {num: (tid, dur, est, mac) for num, tid, dur, est, mac in items}

            permanent = set(F)
            verifiable = []
            max_nodes_num = None
            for num in U:
                tid, dur, est, mac = by_num[num]
                if tid in NON_VERIFIABLE_IDS or est >= 0.90 * (dur + 2_000):
                    permanent.add(num)
                elif tid == TEST_MAX_NODES:
                    max_nodes_num = num
                    verifiable.append(num)
                else:
                    verifiable.append(num)

            if max_nodes_num is not None and max_nodes_num in U:
                if not runner.run_max_nodes_prefix():
                    U.discard(max_nodes_num)
                    F.add(max_nodes_num)
                    permanent.add(max_nodes_num)
                    if max_nodes_num in verifiable:
                        verifiable.remove(max_nodes_num)

            verifiable.sort(key=lambda n: _verifiable_sort_key(n, by_num))

            cur_cost = schedule_cost_ms(F | U, durations_ms)
            perm_cost = schedule_cost_ms(permanent, durations_ms)
            eff_perm = set(permanent)

            def _refresh_eff_perm():
                nonlocal eff_perm
                eff_perm = set(permanent)
                if max_nodes_num is not None and max_nodes_num in U:
                    mn_est = by_num[max_nodes_num][2]
                    mn_upper = schedule_cost_ms(permanent | {max_nodes_num}, durations_ms) - perm_cost
                    if mn_est >= 0.85 * mn_upper:
                        eff_perm.add(max_nodes_num)

            _refresh_eff_perm()
            n_run = 0
            n_pass = 0
            n_fail = 0
            passes_since_check = 0

            def _should_stop_prefix() -> bool:
                rem = [x for x in verifiable if x in U and x not in eff_perm]
                if not rem:
                    return True
                rem_no_max = [x for x in rem if x != max_nodes_num]
                p_est = (n_pass + 1.5) / (n_run + 3.0) if (n_run >= 6 and n_fail >= 4 and len(rem) >= 12) else 1.0
                if rem_no_max:
                    if _best_prefix_net(F | U, cur_cost, rem_no_max, by_num, durations_ms, p_est) > 0:
                        return False
                if max_nodes_num is not None and max_nodes_num in U and max_nodes_num not in eff_perm:
                    mn_est = by_num[max_nodes_num][2]
                    mn_marg = cur_cost - schedule_cost_ms((F | U) - {max_nodes_num}, durations_ms)
                    if mn_est < 0.85 * p_est * mn_marg:
                        return False
                return True

            if _should_stop_prefix():
                verifiable = []

            for num in verifiable:
                if num not in U or num in eff_perm:
                    continue
                if cur_cost <= perm_cost:
                    break
                elapsed = _child_cpu_ms() - r0
                if elapsed >= per_commit_cap_ms:
                    break

                tid, dur, est, mac = by_num[num]
                if tid == TEST_MAX_NODES:
                    cur_cost = schedule_cost_ms(F | U, durations_ms)
                    marginal_sim = cur_cost - schedule_cost_ms((F | U) - {num}, durations_ms)
                    p_est = (n_pass + 1.5) / (n_run + 3.0) if (n_run >= 6 and n_fail >= 4) else 1.0
                    threshold = 0.90 if (not permanent and len(U) == 1) else 0.85
                    if est >= threshold * p_est * marginal_sim:
                        continue
                elif est >= 0.92 * dur or (eff_perm and len(U) <= 40):
                    cur_set = F | U
                    marginal_sim = cur_cost - schedule_cost_ms(cur_set - {num}, durations_ms)
                    if marginal_sim <= 0:
                        continue
                    if eff_perm and est > 0.92 * marginal_sim:
                        rem = [x for x in verifiable if x in U and x not in eff_perm and x != max_nodes_num]
                        p_est = (n_pass + 1.5) / (n_run + 3.0) if (n_run >= 6 and n_fail >= 4 and len(rem) >= 12) else 1.0
                        if _best_prefix_net(cur_set, cur_cost, rem, by_num, durations_ms, p_est) <= 0:
                            break
                        continue

                t_before = _child_cpu_ms()
                ok = runner.run_one(tid)
                t_after = _child_cpu_ms()
                dt = max(0, t_after - t_before)
                n_run += 1
                if mac > 20 and tid not in (TEST_EXECUTE_TPCDS, TEST_OPTIMIZED_TPCH, TEST_EXECUTE_TPCH, TEST_MAX_NODES):
                    cal_actual += dt
                    cal_mac += mac

                if ok:
                    n_pass += 1
                    U.remove(num)
                    passes_since_check += 1
                    if len(U) <= 60 or permanent or passes_since_check >= 12:
                        cur_cost = schedule_cost_ms(F | U, durations_ms)
                        passes_since_check = 0
                        if _should_stop_prefix():
                            break
                else:
                    n_fail += 1
                    U.remove(num)
                    F.add(num)
                    permanent.add(num)
                    perm_cost = schedule_cost_ms(permanent, durations_ms)
                    cur_cost = schedule_cost_ms(F | U, durations_ms)
                    _refresh_eff_perm()
                    passes_since_check = 0
                    if _should_stop_prefix():
                        break

            signal.alarm(0)
            final_set = F | U
            cpu_used = _child_cpu_ms() - r0
            os.write(write_fd, pickle.dumps((final_set, cpu_used, cal_actual, cal_mac)))
        except _AlarmTimeout:
            try:
                signal.alarm(0)
                os.write(
                    write_fd,
                    pickle.dumps((F | U, _child_cpu_ms() - r0, cal_actual, cal_mac)),
                )
            except BaseException:
                pass
        except BaseException:
            try:
                signal.alarm(0)
                os.write(
                    write_fd,
                    pickle.dumps((F | U, _child_cpu_ms() - r0, cal_actual, cal_mac)),
                )
            except BaseException:
                pass
        finally:
            try:
                os.close(write_fd)
            except OSError:
                pass
            os._exit(0)

    os.close(write_fd)
    return pid, read_fd


def _reap_child(pid: int, read_fd: int, fallback_set: set[int]) -> tuple[set[int], int, int, int]:
    chunks = []
    while True:
        chunk = os.read(read_fd, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(read_fd)
    _, status = os.waitpid(pid, 0)
    if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0 and chunks:
        try:
            final_set, cpu_used, cal_actual, cal_mac = pickle.loads(b"".join(chunks))
            return final_set, cpu_used, cal_actual, cal_mac
        except Exception:
            pass
    return fallback_set, 1_000, 0, 0


def verify_batch(jobs: list[dict], index, checkout: Path) -> None:
    """Verify candidate test sets in-memory for commits that benefit from pruning.

    Mutates each job's `selected` set in place.
    """
    if not hasattr(os, "fork"):
        return

    pending = []
    for job in jobs:
        sel: set[int] = job["selected"]
        fixed: set[int] = job.get("fixed_fails", set())
        unverified = sel - fixed
        if not job["patched_files"] or not unverified:
            continue
        if job.get("fixture_only") and len(sel) == 1:
            only = next(iter(sel))
            if index.durations[only] > 50:
                continue
        job["_init_cost"] = schedule_cost_ms(sel, index.durations)
        pending.append(job)

    if not pending:
        return

    # Pre-import immutable stdlib/third-party modules before forking
    if any(
        any(index.test_ids[n].startswith(("tests/test_optimizer.py::", "tests/test_executor.py::")) for n in job["selected"])
        for job in pending
    ):
        import duckdb  # noqa: F401
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import pandas.testing  # noqa: F401
    if checkout.resolve() == Path("/sqlglot") and not any(
        "tests/helpers.py" in job["patched_files"] for job in pending
    ):
        try:
            importlib.import_module("tests.helpers")
        except Exception:
            pass

    start_wall = time.monotonic()
    # Reserve generous margin under 180s (40 commits) / 270s (60 commits)
    wall_budget_s = max(110.0, min(220.0, len(jobs) * 3.0))
    total_cpu_ms = 0
    if getattr(index, "mac_bench_ms", 0) > 0:
        cur_bench = _bench_cpu_loop()
        speed_ratio = max(0.7, min(2.4, cur_bench / index.mac_bench_ms))
    else:
        speed_ratio = 1.6  # Conservative initial ratio (server CPU vs Mac baseline)
    cal_actual_total = 0
    cal_mac_total = 0

    def prep_job(job):
        sel: set[int] = job["selected"]
        fixed: set[int] = job.get("fixed_fails", set())
        unverified = sel - fixed
        pf = job["patched_files"]
        need_tpcds_fallback = any(
            p == "sqlglot/serde.py" or p == "sqlglot/executor/table.py" for p in pf
        )
        tpcds_fast = (
            index.tpcds_baseline is not None
            and not need_tpcds_fallback
            and "tests/test_executor.py" not in pf
            and "tests/helpers.py" not in pf
            and "tests/fixtures/optimizer/tpc-ds/tpc-ds.sql" not in pf
        )
        tpch_fast = (
            index.tpch_opt_baseline is not None
            and not need_tpcds_fallback
            and "tests/test_executor.py" not in pf
            and "tests/helpers.py" not in pf
            and "tests/fixtures/optimizer/tpc-h/tpc-h.sql" not in pf
        )

        items = []
        permanent = set(fixed)
        verifiable = []
        max_nodes_num = None
        for num in sel:
            tid = index.test_ids[num]
            dur = index.durations[num]
            mac = index.baseline_cpu_ms_by_num.get(num, 0)
            cov = index.test_cpu_ms_by_num.get(num)
            est = _est_cpu_ms(tid, dur, mac or None, cov, speed_ratio, pf, tpcds_fast, tpch_fast)
            items.append((num, tid, dur, est, mac))
            if num in unverified:
                if tid in NON_VERIFIABLE_IDS or est >= 0.90 * (dur + 2_000):
                    permanent.add(num)
                elif tid == TEST_MAX_NODES:
                    max_nodes_num = num
                    verifiable.append(num)
                else:
                    verifiable.append(num)

        cur_cost = schedule_cost_ms(sel, index.durations)
        perm_cost = schedule_cost_ms(permanent, index.durations)
        if cur_cost <= perm_cost and max_nodes_num is None:
            return None

        by_num = {num: (tid, dur, est, mac) for num, tid, dur, est, mac in items}
        mn_upper = (
            schedule_cost_ms(permanent | {max_nodes_num}, index.durations) - perm_cost
            if max_nodes_num is not None
            else 0
        )
        mn_viable = max_nodes_num is not None and by_num[max_nodes_num][2] < 0.85 * mn_upper
        rem_no_max = sorted(
            (x for x in verifiable if x != max_nodes_num),
            key=lambda n: _verifiable_sort_key(n, by_num),
        )
        if not mn_viable and (
            not rem_no_max
            or _best_prefix_net(sel, cur_cost, rem_no_max, by_num, index.durations, 1.0) <= 0
        ):
            return None

        cap = min(PER_COMMIT_CPU_CAP_MS, TOTAL_CPU_CAP_MS - total_cpu_ms)
        if cap <= 20:
            return None
        return (job, items, fixed, unverified, cap)

    def run_stage(stage_jobs: list[dict]) -> None:
        nonlocal total_cpu_ms, speed_ratio, cal_actual_total, cal_mac_total
        stage_jobs.sort(key=lambda j: -j["_init_cost"])

        import selectors
        sel_io = selectors.DefaultSelector()
        active: dict[int, tuple[dict, int, set[int]]] = {}  # read_fd -> (job, pid, fallback_set)
        idx = 0

        def harvest(fd: int) -> None:
            nonlocal total_cpu_ms, speed_ratio, cal_actual_total, cal_mac_total
            sel_io.unregister(fd)
            job, pid, fallback_set = active.pop(fd)
            final_set, cpu_used, ca, cm = _reap_child(pid, fd, fallback_set)
            total_cpu_ms += max(1, cpu_used)
            job["selected"] = final_set
            cal_actual_total += ca
            cal_mac_total += cm
            if cal_mac_total >= 500:
                speed_ratio = max(0.6, min(2.5, cal_actual_total / cal_mac_total))

        try:
            while idx < len(stage_jobs) or active:
                while (
                    len(active) < 2
                    and idx < len(stage_jobs)
                    and total_cpu_ms < TOTAL_CPU_CAP_MS
                    and (time.monotonic() - start_wall) < wall_budget_s
                ):
                    job = stage_jobs[idx]
                    idx += 1
                    prep = prep_job(job)
                    if prep is None:
                        continue
                    j, items, fixed, unverified, cap = prep
                    pid, rfd = _spawn_child(
                        checkout, j["patched_files"], items, fixed, unverified, index.durations, cap, index
                    )
                    active[rfd] = (j, pid, fixed | unverified)
                    sel_io.register(rfd, selectors.EVENT_READ)

                if not active:
                    break
                events = sel_io.select(timeout=1.0)
                if not events:
                    continue
                for key, _ in events:
                    harvest(key.fd)
        finally:
            sel_io.close()

    if WARM_INDEX.exists():
        warm_paths = json.loads(WARM_INDEX.read_bytes())
        module_test_prefix = {
            "tests.dialects.test_dialect": "tests/dialects/",
            "tests.test_optimizer": "tests/test_optimizer.py::",
            "tests.test_executor": "tests/test_executor.py::",
        }
        for name in WARM_MODULES:
            if name not in warm_paths:
                continue
            prefix = module_test_prefix.get(name)
            if prefix is not None:
                needers = sum(
                    1 for j in pending
                    if any(index.test_ids[n].startswith(prefix) for n in (j["selected"] - j.get("fixed_fails", set())))
                )
                if needers < 2:
                    continue
            cold_paths = set(warm_paths[name])
            stage = []
            remaining = []
            for job in pending:
                paths = job["patched_files"]
                if cold_paths.intersection(paths):
                    stage.append(job)
                else:
                    remaining.append(job)
            if stage:
                run_stage(stage)
            pending = remaining
            if len(pending) < 2 or checkout.resolve() != Path("/sqlglot"):
                break
            importlib.import_module(name)
    if pending:
        run_stage(pending)


if __name__ == "__main__":
    index_warm_imports(Path("/sqlglot"))
