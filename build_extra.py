"""Build extra_index.pkl.zlib from the pinned unmodified SQLGlot checkout.

Run once against the pinned checkout:
    PYTHONPATH=/path/to/sqlglot python build_extra.py /path/to/sqlglot extra_index.pkl.zlib
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import importlib
import os
import pickle
import resource
import sys
import zlib
from collections import defaultdict, deque
from pathlib import Path


def _resolve_module_file(mod_name: str, existing_mods: dict[str, str]) -> str | None:
    if mod_name in existing_mods:
        return existing_mods[mod_name]
    return None


def _walk_non_tc_imports(node: ast.AST):
    """Yield Import/ImportFrom nodes excluding `if TYPE_CHECKING:` branches."""
    if isinstance(node, ast.If):
        t = node.test
        is_tc = (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING") or (
            isinstance(t, ast.Attribute)
            and t.attr == "TYPE_CHECKING"
            and isinstance(t.value, ast.Name)
            and t.value.id in ("t", "typing")
        )
        if is_tc:
            for s in node.orelse:
                yield from _walk_non_tc_imports(s)
            return
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        yield node
        return
    for child in ast.iter_child_nodes(node):
        yield from _walk_non_tc_imports(child)


def compute_static_importers(checkout: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return (test_importers, sqlglot_importers) via transitive AST import closure."""
    existing_mods: dict[str, str] = {}
    all_py: list[Path] = []
    for sub in ("sqlglot", "tests"):
        base = checkout / sub
        if not base.is_dir():
            continue
        for py in sorted(base.rglob("*.py")):
            rel = py.relative_to(checkout).as_posix()
            all_py.append(py)
            if rel.endswith("/__init__.py"):
                mod = rel[:-12].replace("/", ".")
            else:
                mod = rel[:-3].replace("/", ".")
            existing_mods[mod] = rel

    # Map lazy dialect exports in sqlglot.dialects.__init__
    dialect_by_attr = {
        p.stem: f"sqlglot/dialects/{p.stem}.py"
        for p in sorted((checkout / "sqlglot" / "dialects").glob("*.py"))
        if not p.name.startswith("__")
    }

    # Build direct import graph: importer_rel -> set[imported_rel]
    direct_imports: dict[str, set[str]] = defaultdict(set)
    for py in all_py:
        rel = py.relative_to(checkout).as_posix()
        cur_mod = rel[:-12].replace("/", ".") if rel.endswith("/__init__.py") else rel[:-3].replace("/", ".")
        pkg_parts = cur_mod.split(".") if rel.endswith("/__init__.py") else cur_mod.split(".")[:-1]
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=rel)
        except (OSError, SyntaxError):
            continue
        for node in _walk_non_tc_imports(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    for k in range(len(parts), 0, -1):
                        cand = ".".join(parts[:k])
                        target = _resolve_module_file(cand, existing_mods)
                        if target and target != rel:
                            direct_imports[rel].add(target)
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    up = len(pkg_parts) - (node.level - 1)
                    if up < 0:
                        continue
                    base_parts = pkg_parts[:up]
                    if node.module:
                        base_parts = base_parts + node.module.split(".")
                else:
                    base_parts = node.module.split(".") if node.module else []
                if not base_parts:
                    continue
                base_mod = ".".join(base_parts)
                target = _resolve_module_file(base_mod, existing_mods)
                if target and target != rel:
                    direct_imports[rel].add(target)
                for k in range(len(base_parts) - 1, 0, -1):
                    pt = _resolve_module_file(".".join(base_parts[:k]), existing_mods)
                    if pt and pt != rel:
                        direct_imports[rel].add(pt)
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    sub_mod = f"{base_mod}.{alias.name}"
                    st = _resolve_module_file(sub_mod, existing_mods)
                    if st and st != rel:
                        direct_imports[rel].add(st)
                    elif base_mod == "sqlglot.dialects":
                        dt = dialect_by_attr.get(alias.name.lower())
                        if dt and dt != rel:
                            direct_imports[rel].add(dt)

    # 1. Propagate from each tests/*.py file
    test_files = [
        p.relative_to(checkout).as_posix()
        for p in sorted((checkout / "tests").rglob("*.py"))
        if p.name.startswith("test_") and p.name != "test_docs.py"
    ]
    importers: dict[str, set[str]] = defaultdict(set)
    for tf in test_files:
        seen = {tf}
        q = deque([tf])
        while q:
            cur = q.popleft()
            for nxt in direct_imports.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    importers[nxt].add(tf)
                    q.append(nxt)
        importers[tf].add(tf)

    # 2. Propagate from each sqlglot/*.py file (so e.g. sqlglot/dialects/snowflake.py -> sqlglot/typing/snowflake.py)
    sqlglot_files = [
        p.relative_to(checkout).as_posix()
        for p in sorted((checkout / "sqlglot").rglob("*.py"))
    ]
    sqlglot_importers: dict[str, set[str]] = defaultdict(set)
    for sf in sqlglot_files:
        seen = {sf}
        q = deque([sf])
        while q:
            cur = q.popleft()
            for nxt in direct_imports.get(cur, ()):
                if nxt not in seen and nxt.startswith("sqlglot/"):
                    seen.add(nxt)
                    sqlglot_importers[nxt].add(sf)
                    q.append(nxt)

    return (
        {k: sorted(v) for k, v in importers.items()},
        {k: sorted(v) for k, v in sqlglot_importers.items()},
    )


def build_tpcds_and_tpch_baseline(checkout: Path) -> dict:
    from sqlglot import parse_one, transpile
    from sqlglot.optimizer import optimize
    from sqlglot.planner import Plan
    from tests.helpers import TPCDS_SCHEMA, load_sql_fixture_pairs
    from tests.test_executor import TestExecutor

    TestExecutor.setUpClass()
    tpcds_sqls = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
    tpch_sqls = list(load_sql_fixture_pairs("optimizer/tpc-h/tpc-h.sql"))

    tpcds = []
    for args, sql, _ in tpcds_sqls:
        p = parse_one(sql)
        p_repr = repr(p)
        if args.get("execute"):
            opt = optimize(p, schema=TPCDS_SCHEMA)
            opt_repr = repr(opt)
            plan_repr = repr(Plan(opt))
            duck_sql = transpile(sql, write="duckdb")[0]
            df = TestExecutor.tpcds_conn.execute(duck_sql).fetchdf()
            tpcds.append((p_repr, opt_repr, plan_repr, duck_sql, df))
        else:
            tpcds.append((p_repr, None, None, None, None))

    tpch_opt = [
        (transpile(sql, write="duckdb")[0], transpile(opt, write="duckdb")[0])
        for _, sql, opt in tpch_sqls
    ]
    tpch_exec_count = sum(1 for args, _, _ in tpch_sqls if args.get("execute"))
    return {
        "tpcds": tpcds,
        "tpch_opt": tpch_opt,
        "tpch_exec_count": tpch_exec_count,
    }


def _build_class_defs_cache(checkout: Path) -> dict[str, set[tuple[int, str]]]:
    cache: dict[str, set[tuple[int, str]]] = {}
    for py in sorted((checkout / "sqlglot").rglob("*.py")):
        rel = py.relative_to(checkout).as_posix()
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=rel)
            cache[rel] = {(n.lineno, n.name) for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        except (OSError, SyntaxError):
            cache[rel] = set()
    return cache


def _trace_import_in_fork(
    mod_name: str, checkout: Path, class_defs: dict[str, set[tuple[int, str]]]
) -> tuple[dict[str, set[int]], dict[str, set[tuple[int, str]]]]:
    rfd, wfd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(rfd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)
        root_prefix = checkout.resolve().as_posix().rstrip("/") + "/"
        lines: dict[str, set[int]] = defaultdict(set)
        funcs: dict[str, set[tuple[int, str]]] = defaultdict(set)
        code_cache: dict[object, tuple[str, tuple[int, str]] | None] = {}

        def _info(code):
            res = code_cache.get(code)
            if res is not None:
                return None if res is False else res
            fn = code.co_filename
            if not fn.startswith(root_prefix):
                code_cache[code] = False
                return None
            rel = fn[len(root_prefix):]
            if not rel.startswith("sqlglot/") or code.co_name == "<module>":
                code_cache[code] = False
                return None
            if (code.co_firstlineno, code.co_name) in class_defs.get(rel, ()):
                code_cache[code] = False
                return None
            val = (rel, (code.co_firstlineno, code.co_name))
            code_cache[code] = val
            return val

        def on_start(code, off):
            res = _info(code)
            if res is not None:
                funcs[res[0]].add(res[1])
            return sys.monitoring.DISABLE

        def on_line(code, ln):
            res = _info(code)
            if res is not None:
                lines[res[0]].add(ln)
            return sys.monitoring.DISABLE

        tid = sys.monitoring.COVERAGE_ID
        sys.monitoring.use_tool_id(tid, "extra_imp")
        sys.monitoring.register_callback(tid, sys.monitoring.events.PY_START, on_start)
        sys.monitoring.register_callback(tid, sys.monitoring.events.LINE, on_line)
        sys.monitoring.set_events(tid, sys.monitoring.events.PY_START | sys.monitoring.events.LINE)
        try:
            importlib.import_module(mod_name)
        except BaseException:
            pass
        sys.monitoring.set_events(tid, 0)
        os.write(wfd, pickle.dumps((dict(lines), dict(funcs)), protocol=pickle.HIGHEST_PROTOCOL))
        os.close(wfd)
        os._exit(0)

    os.close(wfd)
    chunks = []
    while True:
        chunk = os.read(rfd, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(rfd)
    os.waitpid(pid, 0)
    return pickle.loads(b"".join(chunks))


def compute_import_trace(checkout: Path) -> dict:
    """Trace which sqlglot lines/functions execute during isolated test-file and submodule imports."""
    class_defs = _build_class_defs_cache(checkout)

    # Run trace in a clean pre-sqlglot subprocess so sys.modules is unpolluted
    def _run_batch_targets(targets: list[tuple[str, str]], preimport_sqlglot: bool) -> dict[str, tuple]:
        out: dict[str, tuple] = {}
        # Fork a clean manager process before sqlglot is imported if preimport_sqlglot is False
        rfd, wfd = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(rfd)
            if preimport_sqlglot:
                importlib.import_module("sqlglot")
            res = {}
            for key, mod_name in targets:
                if preimport_sqlglot and mod_name in sys.modules:
                    continue
                res[key] = _trace_import_in_fork(mod_name, checkout, class_defs)
            os.write(wfd, pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL))
            os.close(wfd)
            os._exit(0)
        os.close(wfd)
        chunks = []
        while True:
            chunk = os.read(rfd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
        os.close(rfd)
        os.waitpid(pid, 0)
        return pickle.loads(b"".join(chunks))

    base_res = _run_batch_targets([("sqlglot", "sqlglot")], preimport_sqlglot=False)
    base_lines, base_funcs = base_res["sqlglot"]

    test_files = [
        p.relative_to(checkout).as_posix()
        for p in sorted((checkout / "tests").rglob("*.py"))
        if p.name.startswith("test_") and p.name != "test_docs.py"
    ]
    test_targets = [(tf, tf[:-3].replace("/", ".")) for tf in test_files]

    sqlglot_targets = []
    for py in sorted((checkout / "sqlglot").rglob("*.py")):
        rel = py.relative_to(checkout).as_posix()
        if rel == "sqlglot/__main__.py":
            continue
        mod = rel[:-12].replace("/", ".") if rel.endswith("/__init__.py") else rel[:-3].replace("/", ".")
        sqlglot_targets.append((rel, mod))

    test_traces = _run_batch_targets(test_targets, preimport_sqlglot=False)
    mod_traces = _run_batch_targets(sqlglot_targets, preimport_sqlglot=True)

    sources = sorted(test_traces.keys()) + sorted(mod_traces.keys())
    src_bit = {s: (1 << i) for i, s in enumerate(sources)}

    line_src_mask: dict[tuple[str, int], int] = defaultdict(int)
    func_src_mask: dict[tuple[str, tuple[int, str]], int] = defaultdict(int)

    for src, (lmap, fmap) in {**test_traces, **mod_traces}.items():
        bit = src_bit[src]
        for rel, lns in lmap.items():
            base_set = base_lines.get(rel, ())
            for ln in lns:
                if ln not in base_set:
                    line_src_mask[(rel, ln)] |= bit
        for rel, fks in fmap.items():
            base_set = base_funcs.get(rel, ())
            for fk in fks:
                if fk not in base_set:
                    func_src_mask[(rel, fk)] |= bit

    mask_to_idx: dict[int, int] = {}
    import_masks: list[int] = []

    def _intern(m: int) -> int:
        idx = mask_to_idx.get(m)
        if idx is None:
            idx = len(import_masks)
            import_masks.append(m)
            mask_to_idx[m] = idx
        return idx

    import_line_idx: dict[str, dict[int, int]] = defaultdict(dict)
    for (rel, ln), m in line_src_mask.items():
        import_line_idx[rel][ln] = _intern(m)

    import_func_idx: dict[str, dict[tuple[int, str], int]] = defaultdict(dict)
    for (rel, fk), m in func_src_mask.items():
        import_func_idx[rel][fk] = _intern(m)

    return {
        "base_lines": {k: set(v) for k, v in base_lines.items()},
        "base_funcs": {k: set(v) for k, v in base_funcs.items()},
        "import_sources": sources,
        "import_masks": import_masks,
        "import_line_idx": dict(import_line_idx),
        "import_func_idx": dict(import_func_idx),
    }


def bench_cpu_loop() -> int:
    t0 = resource.getrusage(resource.RUSAGE_SELF)
    d = {f"k{i}": i for i in range(64)}
    acc = 0
    for _ in range(300):
        nodes = [(i, str(i), d.get(f"k{i & 63}", 0)) for i in range(400)]
        acc += sum(x[0] + x[2] for x in nodes)
    t1 = resource.getrusage(resource.RUSAGE_SELF)
    _ = acc
    return max(1, round(((t1.ru_utime - t0.ru_utime) + (t1.ru_stime - t0.ru_stime)) * 1000))


def measure_baseline_cpu(checkout: Path, test_ids: list[str]) -> dict[str, int]:
    from runner import NON_VERIFIABLE_IDS, _ChildRunner, _make_lazy_pool_cls

    concurrent.futures.ProcessPoolExecutor = _make_lazy_pool_cls(False)
    concurrent.futures.wait = lambda fs, *a, **kw: (set(fs), set())
    concurrent.futures.as_completed = lambda fs, *a, **kw: iter(list(fs))

    r = _ChildRunner(checkout, {})
    cpu_by_id: dict[str, int] = {}
    for tid in test_ids:
        if tid in NON_VERIFIABLE_IDS:
            continue
        ru0 = resource.getrusage(resource.RUSAGE_SELF)
        try:
            ok = r.run_one(tid)
        except Exception:
            ok = False
        ru1 = resource.getrusage(resource.RUSAGE_SELF)
        cpu_ms = round(((ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)) * 1000)
        if ok:
            cpu_by_id[tid] = max(1, cpu_ms)
    return cpu_by_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--coverage", type=Path, default=Path("coverage.pkl.zlib"))
    parser.add_argument("--reuse-from", type=Path, default=None)
    args = parser.parse_args()

    app_dir = str(Path(__file__).resolve().parent)
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    cov_path = args.coverage.resolve()
    out_path = args.output.resolve()
    checkout = args.checkout.resolve()
    os.chdir(checkout)
    os.environ["SKIP_INTEGRATION"] = "0"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    if str(checkout) not in sys.path:
        sys.path.insert(0, str(checkout))

    cov = pickle.loads(zlib.decompress(cov_path.read_bytes()))
    # Exclude skipped test from baseline measurement if present
    test_ids = [
        t for t in cov["tests"]
        if t != "tests/dialects/test_sqlite.py::TestSQLite::test_variadic_iif_execution"
    ]

    print("Computing static_importers...", file=sys.stderr)
    static_importers, sqlglot_importers = compute_static_importers(checkout)
    print("Tracing isolated import executions...", file=sys.stderr)
    import_trace = compute_import_trace(checkout)

    if args.reuse_from is not None and args.reuse_from.resolve().exists():
        prev = pickle.loads(zlib.decompress(args.reuse_from.resolve().read_bytes()))
        baseline_cpu_ms = prev["baseline_cpu_ms"]
        mac_bench_ms = bench_cpu_loop()
        exec_base = {
            "tpcds": prev["tpcds"],
            "tpch_opt": prev["tpch_opt"],
            "tpch_exec_count": prev["tpch_exec_count"],
        }
    else:
        print("Building TPC-DS/TPC-H baselines...", file=sys.stderr)
        exec_base = build_tpcds_and_tpch_baseline(checkout)
        print(f"Measuring baseline CPU across {len(test_ids)} tests...", file=sys.stderr)
        baseline_cpu_ms = measure_baseline_cpu(checkout, test_ids)
        mac_bench_ms = bench_cpu_loop()

    payload = {
        "static_importers": static_importers,
        "sqlglot_importers": sqlglot_importers,
        "import_trace": import_trace,
        "baseline_cpu_ms": baseline_cpu_ms,
        "mac_bench_ms": mac_bench_ms,
        **exec_base,
    }
    blob = zlib.compress(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL), level=9)
    out_path.write_bytes(blob)
    print(f"Wrote {out_path} ({len(blob)/1024:.1f} KB, {len(baseline_cpu_ms)} cpu timings)", file=sys.stderr)


if __name__ == "__main__":
    main()
