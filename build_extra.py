"""Build extra_index.pkl.zlib from the pinned unmodified SQLGlot checkout.

Run once against the pinned checkout:
    PYTHONPATH=/path/to/sqlglot python build_extra.py /path/to/sqlglot extra_index.pkl.zlib
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
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
    print("Building TPC-DS/TPC-H baselines...", file=sys.stderr)
    exec_base = build_tpcds_and_tpch_baseline(checkout)
    print(f"Measuring baseline CPU across {len(test_ids)} tests...", file=sys.stderr)
    baseline_cpu_ms = measure_baseline_cpu(checkout, test_ids)

    payload = {
        "static_importers": static_importers,
        "sqlglot_importers": sqlglot_importers,
        "baseline_cpu_ms": baseline_cpu_ms,
        **exec_base,
    }
    blob = zlib.compress(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL), level=9)
    out_path.write_bytes(blob)
    print(f"Wrote {out_path} ({len(blob)/1024:.1f} KB, {len(baseline_cpu_ms)} cpu timings)", file=sys.stderr)


if __name__ == "__main__":
    main()
