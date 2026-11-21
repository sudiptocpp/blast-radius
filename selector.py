"""Select SQLGlot tests from clean-baseline coverage, patch structure, and fast verification."""

from __future__ import annotations

import argparse
import ast
import enum
import json
import pickle
import zlib
from collections import Counter, defaultdict, deque
from pathlib import Path

from equiv import ast_equal, is_fixture_comment_only, is_semantically_equivalent
from patches import apply_python, apply_text, parse_patch
from runner import verify_batch
from schedule import schedule


APP = Path(__file__).resolve().parent
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
DIALECT_DIRS = ("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/")
DIALECT_BASE_FILES = frozenset({"sqlglot/dialects/dialect.py", "sqlglot/parsers/base.py"})
SHARED_TEST_FILES = frozenset({
    "tests/__init__.py",
    "tests/helpers.py",
    "tests/dialects/test_dialect.py",
    "tests/leakcheck.py",
    "tests/gen_fixtures.py",
})
CORE_BASE_FILES = frozenset({
    "sqlglot/__init__.py",
    "sqlglot/dialects/__init__.py",
    "sqlglot/dialects/dialect.py",
    "sqlglot/parsers/base.py",
    "sqlglot/parser.py",
    "sqlglot/generator.py",
    "sqlglot/tokens.py",
    "sqlglot/tokenizer_core.py",
    "sqlglot/expressions/core.py",
    "sqlglot/helper.py",
    "sqlglot/errors.py",
    "tests/__init__.py",
    "tests/helpers.py",
    "tests/dialects/test_dialect.py",
})

# Class dicts whose values are transformed at class-creation time into tries/inverses
VALUE_DERIVED_DICTS = frozenset({
    "TIME_MAPPING",
    "FORMAT_MAPPING",
    "CREATABLE_KIND_MAPPING",
    "UNESCAPED_SEQUENCES",
    "ESCAPED_SEQUENCES",
    "IDENTIFIER_ESCAPED_SEQUENCES",
    "INVERSE_TIME_MAPPING",
    "INVERSE_FORMAT_MAPPING",
    "INVERSE_CREATABLE_KIND_MAPPING",
    "DATE_PART_MAPPING",
    "JODA_TIME_MAPPING",
    "INVERSE_JODA_TIME_MAPPING",
    "MYSQL_INVERSE_TIME_MAPPING",
    "COERCES_TO",
})

# Class dicts whose key set feeds trie construction or _Dialect.__new__ key-presence checks
KEY_DERIVED_DICTS = frozenset({
    "SHOW_PARSERS",
    "SET_PARSERS",
    "NUMERIC_ESCAPES",
    "KEYWORDS",
    "SINGLE_TOKENS",
})

# Class attributes read by _Dialect.__new__ / _TokenizerBase.__init_subclass__
DERIVED_CLASS_ATTRS = frozenset({
    "QUOTES",
    "IDENTIFIERS",
    "BIT_STRINGS",
    "BYTE_STRINGS",
    "HEX_STRINGS",
    "NATIONAL_RAW_STRINGS",
    "RAW_STRINGS",
    "HEREDOC_STRINGS",
    "UNICODE_STRINGS",
    "STRING_ESCAPES",
    "BYTE_STRING_ESCAPES",
    "IDENTIFIER_ESCAPES",
    "COMMENTS",
    "HINT_START",
    "SUPPORTED_JSON_PATH_PARTS",
    "VALID_INTERVAL_UNITS",
    "NORMALIZATION_STRATEGY",
})


def _canonical_literal_key(node: ast.AST) -> str | None:
    try:
        k = ast.literal_eval(node)
    except Exception:
        return None
    if isinstance(k, str):
        return f"s:{k}"
    if isinstance(k, bool):
        return f"b:{'1' if k else '0'}"
    if isinstance(k, int):
        return f"i:{k}"
    if k is None:
        return "n:"
    if isinstance(k, tuple):
        parts = []
        for item in k:
            if isinstance(item, str):
                parts.append(f"s:{item}")
            elif isinstance(item, bool):
                parts.append(f"b:{'1' if item else '0'}")
            elif isinstance(item, int):
                parts.append(f"i:{item}")
            elif item is None:
                parts.append("n:")
            else:
                return None
        return f"t:({','.join(parts)})"
    return None


def _resolve_ast_dict_key(node: ast.AST, enum_classes_by_name: dict[str, dict]) -> str | None:
    ck = _canonical_literal_key(node)
    if ck is not None:
        return ck
    if isinstance(node, ast.Attribute):
        # e.g. exp.DType.INT or TokenType.SELECT or exp.Select
        parts = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            parts.reverse()
            # Match enum key: e:<module>.<qualname>.<member>
            if len(parts) >= 2:
                enum_short = parts[-2]
                member = parts[-1]
                einfo = enum_classes_by_name.get(enum_short)
                if einfo and member in einfo.get("members", ()):
                    cls_key = einfo["key"]  # "rel_path:QualName"
                    rel, qual = cls_key.split(":", 1)
                    mod = rel[:-3].replace("/", ".")
                    return f"e:{mod}.{qual}.{member}"
    if isinstance(node, ast.Tuple):
        parts = []
        for elt in node.elts:
            sub = _resolve_ast_dict_key(elt, enum_classes_by_name)
            if sub is None:
                return None
            parts.append(sub)
        return f"t:({','.join(parts)})"
    return None


def _resolve_expr_type_key(node: ast.AST, expr_keys_by_short: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        cls_key = expr_keys_by_short.get(node.id)
    elif isinstance(node, ast.Attribute):
        cls_key = expr_keys_by_short.get(node.attr)
    else:
        return None
    if not cls_key:
        return None
    rel, qual = cls_key.split(":", 1)
    mod = rel[:-3].replace("/", ".")
    return f"c:{mod}.{qual}"


def start_line(statement: ast.stmt) -> int:
    return min([statement.lineno, *(node.lineno for node in getattr(statement, "decorator_list", ()))])


class Source:
    """Look up pre-change statements, classes, and function/lambda bodies by line number."""

    def __init__(self, tree: ast.AST):
        self.tree = tree
        self.statements = []
        self.functions = []
        self.class_of_node = {}
        self.func_of_node = {}
        pending = deque([(tree, None, None)])
        while pending:
            node, cls, fn = pending.popleft()
            if isinstance(node, ast.stmt):
                self.statements.append(node)
                self.class_of_node[id(node)] = cls
            if isinstance(node, FUNCTIONS):
                self.functions.append(node)
                self.func_of_node[id(node)] = fn
                next_fn = node
            else:
                next_fn = fn
            if isinstance(node, ast.ClassDef):
                cls = node
            pending.extend((child, cls, next_fn) for child in ast.iter_child_nodes(node))

    def function_at(self, line: int):
        return min(
            (node for node in self.functions if node.lineno <= line <= node.end_lineno),
            key=lambda node: node.end_lineno - node.lineno,
            default=None,
        )

    def statement_at(self, line: int):
        return min(
            (node for node in self.statements if node.lineno <= line <= node.end_lineno),
            key=lambda node: (node.end_lineno - node.lineno, -node.col_offset),
            default=None,
        )

    def body_at(self, line: int):
        function = self.function_at(line)
        if function is None:
            return None
        if isinstance(function, ast.Lambda):
            if function.body.lineno <= line <= function.body.end_lineno:
                return function
            return None
        if function.lineno < line and function.body and start_line(function.body[0]) <= line:
            return function
        return None

    def expr_at_inside(self, line: int, root_node: ast.AST):
        return min(
            (
                node
                for node in ast.walk(root_node)
                if isinstance(node, ast.expr) and node.lineno <= line <= node.end_lineno
            ),
            key=lambda node: (node.end_lineno - node.lineno, -node.col_offset),
            default=None,
        )

    def class_qualname(self, cls_node: ast.ClassDef | None) -> str:
        if cls_node is None:
            return ""
        parts = []
        cur = cls_node
        while cur is not None:
            if isinstance(cur, ast.ClassDef):
                parts.append(cur.name)
            cur = self.class_of_node.get(id(cur))
        parts.reverse()
        return ".".join(parts)


def statement_header_lines(stmt: ast.stmt) -> range:
    """Return all lines evaluated as part of `stmt`'s header/condition before branching."""
    if isinstance(stmt, (ast.If, ast.While)):
        end = stmt.test.end_lineno or stmt.lineno
        return range(stmt.lineno, end + 1)
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        end = stmt.iter.end_lineno or stmt.lineno
        return range(stmt.lineno, end + 1)
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        end = (stmt.items[-1].end_lineno if stmt.items else stmt.lineno) or stmt.lineno
        return range(stmt.lineno, end + 1)
    if isinstance(stmt, ast.Match):
        end = stmt.subject.end_lineno or stmt.lineno
        return range(stmt.lineno, end + 1)
    if isinstance(stmt, (ast.Try, ast.TryStar, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return range(stmt.lineno, stmt.lineno + 1)
    end = stmt.end_lineno or stmt.lineno
    return range(stmt.lineno, end + 1)


def literal_binding(node: ast.stmt | None):
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target, annotation = node.targets[0], None
    elif isinstance(node, ast.AnnAssign) and node.value is not None:
        target, annotation = node.target, ast.dump(node.annotation)
    else:
        return None
    if not isinstance(target, ast.Name) or target.id.startswith("__"):
        return None
    try:
        value = ast.literal_eval(node.value)
    except (ValueError, TypeError):
        return None
    return target.id, annotation, type(value), node.col_offset


def _dict_assign_target(stmt: ast.stmt | None) -> tuple[str, ast.Dict, str | None] | None:
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target, annotation, val = stmt.targets[0], None, stmt.value
    elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
        target, annotation, val = stmt.target, ast.dump(stmt.annotation), stmt.value
    else:
        return None
    if not isinstance(target, ast.Name) or target.id.startswith("__"):
        return None
    if not isinstance(val, ast.Dict):
        return None
    return target.id, val, annotation


def _func_key(node: ast.AST) -> tuple[int, str]:
    if isinstance(node, ast.Lambda):
        return (node.lineno, "<lambda>")
    return (node.lineno, node.name)


def _is_pure_const_expr(node: ast.AST | None) -> bool:
    if node is None:
        return True
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Not, ast.Invert)):
        return _is_pure_const_expr(node.operand)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_pure_const_expr(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(_is_pure_const_expr(k) and _is_pure_const_expr(v) for k, v in zip(node.keys, node.values) if k is not None)
    return False


def _same_arg_names(a: ast.arguments, b: ast.arguments) -> bool:
    if [x.arg for x in a.posonlyargs] != [x.arg for x in b.posonlyargs]:
        return False
    if [x.arg for x in a.args] != [x.arg for x in b.args]:
        return False
    if [x.arg for x in a.kwonlyargs] != [x.arg for x in b.kwonlyargs]:
        return False
    if (a.vararg.arg if a.vararg else None) != (b.vararg.arg if b.vararg else None):
        return False
    if (a.kwarg.arg if a.kwarg else None) != (b.kwarg.arg if b.kwarg else None):
        return False
    return True


def _is_safe_default_only_change(old_fn: ast.AST, new_fn: ast.AST) -> bool:
    if type(old_fn) is not type(new_fn) or not isinstance(old_fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    if old_fn.name != new_fn.name:
        return False
    if not ast_equal(old_fn.decorator_list, new_fn.decorator_list):
        return False
    if not _same_arg_names(old_fn.args, new_fn.args):
        return False
    if len(old_fn.args.defaults) != len(new_fn.args.defaults):
        return False
    if len(old_fn.args.kw_defaults) != len(new_fn.args.kw_defaults):
        return False
    for d in new_fn.args.defaults:
        if not _is_pure_const_expr(d):
            return False
    for d in new_fn.args.kw_defaults:
        if not _is_pure_const_expr(d):
            return False
    return True


class Index:
    def __init__(self, tests: list[dict], coverage: dict, extra: dict | None = None):
        self.all_tests = {test["number"] for test in tests}
        self.test_files = defaultdict(set)
        self.test_files_mask = defaultdict(int)
        self.durations = {}
        self.test_ids = {}
        by_id = {}
        for test in tests:
            num = test["number"]
            by_id[test["id"]] = num
            self.test_ids[num] = test["id"]
            self.test_files[test["file"]].add(num)
            self.durations[num] = round(test["seconds"] * 1000)

        self.by_id = by_id
        self.numbers = [by_id.get(nodeid) for nodeid in coverage["tests"]]
        self.all_cov_mask = (1 << len(self.numbers)) - 1

        # Remap cov bit position -> test number & build test_file masks in cov bit space
        self.cov_idx_by_num = {}
        for i, num in enumerate(self.numbers):
            if num is not None:
                self.cov_idx_by_num[num] = i
        for fpath, nums in self.test_files.items():
            m = 0
            for n in nums:
                ci = self.cov_idx_by_num.get(n)
                if ci is not None:
                    m |= (1 << ci)
            self.test_files_mask[fpath] = m

        missing_from_cov = self.all_tests - {n for n in self.numbers if n is not None}
        self.unverified_numbers = missing_from_cov | self._mask_to_numbers(coverage["unverified_mask"])

        self.func_lines = coverage["func_lines"]
        self.func_calls = coverage["func_calls"]
        self.init_lines = coverage["init_lines"]
        self.init_collection = coverage["init_collection"]
        self.func_calls_collection = coverage["func_calls_collection"]
        self.resources = coverage["resources"]
        self.module_usage = coverage["module_usage"]
        self.dict_key_usage = coverage["dict_key_usage"]
        self.dict_all_usage = coverage["dict_all_usage"]
        self.attr_usage = coverage["attr_usage"]
        self.expr_usage = coverage["expr_usage"]
        self.class_hierarchy = coverage["class_hierarchy"]
        self.typing_spreads = coverage.get("typing_spreads", {})

        self.enum_classes = coverage["enum_classes"]
        self.enum_classes_by_name = {info["name"]: info for info in self.enum_classes.values()}
        self.enum_typed_class_attrs = coverage["enum_typed_class_attrs"]
        self.unsafe_none_eq_classes = set(coverage["unsafe_none_eq_classes"])
        self.safe_none_eq_classes = set(coverage["safe_none_eq_classes"])

        self.expr_keys_by_short = {
            ck.split(":", 1)[1].rsplit(".", 1)[-1]: ck
            for ck in self.class_hierarchy
            if ck.startswith("sqlglot/expressions/")
        }

        # Precompute reverse spread lookup: (parent_owner, attr) -> list[(child_owner, explicit_set)]
        self.spread_children = defaultdict(list)
        for ckey, info in self.class_hierarchy.items():
            for attr, sdata in info.get("dict_spreads", {}).items():
                explicit = set(sdata.get("explicit", ()))
                for p_owner, p_attr in sdata.get("spreads", ()):
                    self.spread_children[(p_owner, p_attr)].append((ckey, attr, explicit))
        for trel, sdata in self.typing_spreads.items():
            explicit = set(sdata.get("explicit", ()))
            for p_rel, p_attr in sdata.get("parents", ()):
                self.spread_children[(p_rel, p_attr)].append((trel, "EXPRESSION_METADATA", explicit))

        # Test CPU ms mapped to tests.json number
        cov_cpu = coverage.get("test_cpu_ms", [])
        self.test_cpu_ms_by_num = {}
        for i, ms in enumerate(cov_cpu):
            if i < len(self.numbers) and self.numbers[i] is not None:
                self.test_cpu_ms_by_num[self.numbers[i]] = ms

        if extra is None:
            extra_path = APP / "extra_index.pkl.zlib"
            if extra_path.exists():
                extra = pickle.loads(zlib.decompress(extra_path.read_bytes()))
            else:
                extra = {}

        self.static_importers_mask: dict[str, int] = {}
        for rel, test_files in extra.get("static_importers", {}).items():
            m = 0
            for tf in test_files:
                m |= self.test_files_mask.get(tf, 0)
            if m:
                self.static_importers_mask[rel] = m
        self.sqlglot_importers: dict[str, list[str]] = extra.get("sqlglot_importers", {})

        self.baseline_cpu_ms_by_num: dict[int, int] = {}
        for tid, ms in extra.get("baseline_cpu_ms", {}).items():
            num = by_id.get(tid)
            if num is not None:
                self.baseline_cpu_ms_by_num[num] = ms

        self.tpcds_baseline = extra.get("tpcds")
        self.tpch_opt_baseline = extra.get("tpch_opt")
        self.tpch_exec_count = extra.get("tpch_exec_count", 0)

        # Per-checkout AST / source cache across commits in a batch
        self._source_cache: dict[str, tuple[str, ast.AST, Source]] = {}

    def _get_old_parsed(self, path: str, before: str) -> tuple[ast.AST, Source]:
        cached = self._source_cache.get(path)
        if cached is not None and cached[0] == before:
            return cached[1], cached[2]
        old_tree = ast.parse(before.encode("utf-8"), filename=path, type_comments=True)
        old_src = Source(old_tree)
        self._source_cache[path] = (before, old_tree, old_src)
        return old_tree, old_src

    def _mask_to_numbers(self, mask: int) -> set[int]:
        if mask < 0 or mask == self.all_cov_mask:
            return self.all_tests.copy()
        out = set()
        nums = self.numbers
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            num = nums[idx]
            if num is not None:
                out.add(num)
            mask ^= lsb
        return out

    def collecting_mask(self, modules) -> int:
        if "*" in modules:
            return -1
        mask = 0
        for module in modules:
            mask |= self.test_files_mask.get(module, 0)
        return mask

    def resource_users_mask(self, path: str) -> int:
        resource = self.resources.get(path)
        if not resource:
            return 0
        col = self.collecting_mask(resource.get("collection", ()))
        if col < 0:
            return -1
        return resource.get("mask", 0) | col

    def dialect_users_mask(self, path: str) -> int:
        if path in DIALECT_BASE_FILES or not path.startswith(DIALECT_DIRS):
            return -1
        return self.module_users_mask(path)

    def _single_module_mask(self, path: str) -> int:
        mask = self.module_usage.get(path, 0) | self.static_importers_mask.get(path, 0)
        for covered in self.func_lines.get(path, {}).values():
            mask |= covered
        for covered in self.init_lines.get(path, {}).values():
            mask |= covered
        for collected in self.init_collection.get(path, {}).values():
            cm = self.collecting_mask(collected)
            if cm < 0:
                return -1
            mask |= cm
        for collected in self.func_calls_collection.get(path, {}).values():
            cm = self.collecting_mask(collected)
            if cm < 0:
                return -1
            mask |= cm
        return mask

    def module_users_mask(self, path: str) -> int:
        if path in CORE_BASE_FILES:
            return -1
        if path.startswith("tests/") and path not in SHARED_TEST_FILES:
            tfm = self.test_files_mask.get(path, 0)
            if tfm:
                res_m = self.resource_users_mask(path)
                if res_m < 0:
                    return -1
                return tfm | self.static_importers_mask.get(path, 0) | res_m
        mask = self._single_module_mask(path)
        if mask < 0:
            return -1
        res_m = self.resource_users_mask(path)
        if res_m < 0:
            return -1
        mask |= res_m
        for imp in self.sqlglot_importers.get(path, ()):
            if imp in CORE_BASE_FILES:
                return -1
            im = self._single_module_mask(imp)
            if im < 0:
                return -1
            mask |= im
        return mask if mask else -1

    def _inheriting_classes_for_attr(self, cls_key: str, attr: str) -> set[str]:
        result = {cls_key}
        queue = deque([cls_key])
        while queue:
            cur = queue.popleft()
            info = self.class_hierarchy.get(cur)
            if not info:
                continue
            for sub in info.get("subclasses", ()):
                if sub in result:
                    continue
                sub_info = self.class_hierarchy.get(sub)
                if sub_info and attr in sub_info.get("own_attrs", ()):
                    continue
                result.add(sub)
                queue.append(sub)
        return result

    def _inheriting_owners_for_dict_key(self, owner_key: str, attr: str, ck: str | None) -> tuple[set[tuple[str, str]], bool]:
        """Return (all (owner, attr) pairs inheriting key `ck`, has_unresolved_spread)."""
        result = {(owner_key, attr)}
        queue = deque([(owner_key, attr)])
        unresolved = False
        while queue:
            cur_owner, cur_attr = queue.popleft()
            # 1. Direct class inheritance (subclasses that don't redefine cur_attr)
            info = self.class_hierarchy.get(cur_owner)
            if info:
                for sub in info.get("subclasses", ()):
                    sub_info = self.class_hierarchy.get(sub)
                    if sub_info and cur_attr not in sub_info.get("own_dict_keys", {}):
                        pair = (sub, cur_attr)
                        if pair not in result:
                            result.add(pair)
                            queue.append(pair)
            # 2. Explicit spread dependents (**Parent.DICT)
            for child_owner, child_attr, explicit in self.spread_children.get((cur_owner, cur_attr), ()):
                cinfo = self.class_hierarchy.get(child_owner)
                if cinfo:
                    sdata = cinfo.get("dict_spreads", {}).get(child_attr, {})
                    if sdata.get("unresolved"):
                        unresolved = True
                if ck is None or ck not in explicit:
                    pair = (child_owner, child_attr)
                    if pair not in result:
                        result.add(pair)
                        queue.append(pair)
        return result, unresolved

    def _resolve_ast_key_any(self, node: ast.AST) -> str | None:
        ck = _resolve_ast_dict_key(node, self.enum_classes_by_name)
        if ck is not None:
            return ck
        return _resolve_expr_type_key(node, self.expr_keys_by_short)

    def _handle_class_dict_edit(
        self, path: str, owner_key: str, old_stmt: ast.stmt, new_stmt: ast.stmt
    ) -> int | None:
        """Analyze class/typing dict edit. Returns bitmask or None if fallback needed."""
        old_d = _dict_assign_target(old_stmt)
        new_d = _dict_assign_target(new_stmt)
        if old_d is None or new_d is None:
            return None
        old_attr, old_dict, old_ann = old_d
        new_attr, new_dict, new_ann = new_d
        if old_attr != new_attr or old_ann != new_ann:
            return None

        attr = old_attr
        if attr in VALUE_DERIVED_DICTS:
            return None

        # Verify owner was instrumented (class dict or typing EXPRESSION_METADATA)
        if owner_key in self.class_hierarchy:
            if attr not in self.class_hierarchy[owner_key].get("tracked_dicts", ()):
                return None
        elif owner_key not in self.typing_spreads or attr != "EXPRESSION_METADATA":
            return None

        # Separate unpacks (**expr) from explicit entries
        old_unpacks = [v for k, v in zip(old_dict.keys, old_dict.values) if k is None]
        new_unpacks = [v for k, v in zip(new_dict.keys, new_dict.values) if k is None]
        if len(old_unpacks) != len(new_unpacks) or not all(
            ast_equal(a, b) for a, b in zip(old_unpacks, new_unpacks)
        ):
            return None

        old_entries = {}
        for k, v in zip(old_dict.keys, old_dict.values):
            if k is None:
                continue
            ck = self._resolve_ast_key_any(k)
            if ck is None:
                return None
            old_entries[ck] = v

        new_entries = {}
        for k, v in zip(new_dict.keys, new_dict.values):
            if k is None:
                continue
            ck = self._resolve_ast_key_any(k)
            if ck is None:
                return None
            new_entries[ck] = v

        added_or_removed = set(old_entries) != set(new_entries)
        if added_or_removed and attr in KEY_DERIVED_DICTS:
            return None

        changed_keys = (set(old_entries) ^ set(new_entries)) | {
            ck for ck in (set(old_entries) & set(new_entries)) if not ast_equal(old_entries[ck], new_entries[ck])
        }
        if not changed_keys:
            return 0

        mask = 0
        # Special case: Expr.arg_types -> union expr_usage across inheriting subclasses
        if attr == "arg_types" and owner_key.startswith("sqlglot/expressions/"):
            owners, unresolved = self._inheriting_owners_for_dict_key(owner_key, attr, None)
            if unresolved:
                return None
            for sub_owner, sub_attr in owners:
                mask |= self.expr_usage.get(sub_owner, 0)
                mask |= self.dict_all_usage.get((sub_owner, sub_attr), 0)
                for ck in changed_keys:
                    mask |= self.dict_key_usage.get((sub_owner, sub_attr, ck), 0)
            return mask

        for ck in changed_keys:
            owners, unresolved = self._inheriting_owners_for_dict_key(owner_key, attr, ck)
            if unresolved:
                return None
            for sub_owner, sub_attr in owners:
                mask |= self.dict_key_usage.get((sub_owner, sub_attr, ck), 0)
                mask |= self.dict_all_usage.get((sub_owner, sub_attr), 0)
        return mask

    def _handle_class_scalar_edit(
        self, cls_key: str, old_stmt: ast.stmt, new_stmt: ast.stmt
    ) -> int | None:
        old_b = literal_binding(old_stmt)
        new_b = literal_binding(new_stmt)
        if not old_b or old_b != new_b:
            return None
        attr = old_b[0]
        if attr in DERIVED_CLASS_ATTRS:
            return None
        cinfo = self.class_hierarchy.get(cls_key)
        if not cinfo or attr not in cinfo.get("tracked_scalars", ()):
            return None
        mask = 0
        for sub in self._inheriting_classes_for_attr(cls_key, attr):
            mask |= self.attr_usage.get((sub, attr), 0)
        return mask

    def _narrow_test_file_edit(self, path: str, old_tree: ast.AST, old_src: Source, removed: set[int], added: dict[int, int], new_src: Source) -> int | None:
        """If edit in a test file is strictly inside one non-called test method, return its single-test mask."""
        methods = set()
        for line in removed:
            body = old_src.body_at(line)
            if not body or not isinstance(body, (ast.FunctionDef, ast.AsyncFunctionDef)) or not body.name.startswith("test_"):
                return None
            cls_node = old_src.class_of_node.get(id(body))
            if not cls_node:
                return None
            methods.add((cls_node.name, body.name))
        for line, anchor in added.items():
            body = new_src.body_at(line)
            old_body = old_src.body_at(anchor)
            if (
                not body
                or not old_body
                or not isinstance(body, (ast.FunctionDef, ast.AsyncFunctionDef))
                or not isinstance(old_body, (ast.FunctionDef, ast.AsyncFunctionDef))
                or body.name != old_body.name
                or not body.name.startswith("test_")
            ):
                return None
            cls_node = old_src.class_of_node.get(id(old_body))
            if not cls_node:
                return None
            methods.add((cls_node.name, body.name))
        if not methods:
            return None

        # Verify none of those test method names are called directly elsewhere in the file
        meth_names = {m for _, m in methods}
        for node in ast.walk(old_tree):
            if isinstance(node, ast.Call):
                fn = node.func
                if (isinstance(fn, ast.Attribute) and fn.attr in meth_names) or (
                    isinstance(fn, ast.Name) and fn.id in meth_names
                ):
                    return None

        mask = 0
        for cls_name, meth_name in methods:
            nid = f"{path}::{cls_name}::{meth_name}"
            num = self.by_id.get(nid)
            if num is None:
                return None
            ci = self.cov_idx_by_num.get(num)
            if ci is None:
                return None
            mask |= (1 << ci)
        return mask

    def python_users_mask(self, path: str, before: str, after: str, removed: set[int], added: dict[int, int]) -> int:
        old_tree, old = self._get_old_parsed(path, before)
        new_tree = ast.parse(after.encode("utf-8"), filename=path, type_comments=True)
        compile(new_tree, path, "exec")

        selected_mask = self.resource_users_mask(path)
        if selected_mask < 0:
            return -1

        if is_semantically_equivalent(path, old_tree, new_tree, self):
            return selected_mask

        new = Source(new_tree)
        if path.startswith("tests/"):
            narrow = self._narrow_test_file_edit(path, old_tree, old, removed, added, new)
            if narrow is not None:
                return selected_mask | narrow
            if path not in SHARED_TEST_FILES and self.test_files_mask.get(path, 0):
                return selected_mask | self.test_files_mask[path]
            selected_mask |= self.test_files_mask.get(path, 0)

        func_line_anchors: set[int] = set()
        func_call_anchors: set[tuple[int, str]] = set()
        # Group non-body edits by old statement id -> (owner_key, old_stmt, new_stmt)
        class_level_edits: dict[int, tuple[str, ast.stmt, ast.stmt]] = {}
        func_header_edits: dict[int, tuple[str, ast.stmt, ast.stmt | None]] = {}

        def _owner_key_for_stmt(src_obj: Source, stmt: ast.stmt) -> str:
            cls_node = src_obj.class_of_node.get(id(stmt))
            if cls_node is not None:
                return f"{path}:{src_obj.class_qualname(cls_node)}"
            return path

        def _stmt_assign_name(stmt: ast.stmt) -> str | None:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                return stmt.targets[0].id
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                return stmt.target.id
            return None

        def _build_assign_index(src_obj: Source) -> dict[tuple[str, str], ast.stmt | None]:
            idx: dict[tuple[str, str], ast.stmt | None] = {}
            for s in src_obj.statements:
                name = _stmt_assign_name(s)
                if name is None or src_obj.body_at(s.lineno) is not None:
                    continue
                key = (_owner_key_for_stmt(src_obj, s), name)
                idx[key] = None if key in idx else s
            return idx

        def _build_func_index(src_obj: Source) -> dict[tuple[str, str], ast.stmt | None]:
            idx: dict[tuple[str, str], ast.stmt | None] = {}
            for f in src_obj.functions:
                if isinstance(f, ast.Lambda) or src_obj.func_of_node.get(id(f)) is not None:
                    continue
                key = (_owner_key_for_stmt(src_obj, f), f.name)
                idx[key] = None if key in idx else f
            return idx

        old_assign_idx = None
        new_assign_idx = None
        old_func_idx = None
        new_func_idx = None

        def _module_fallback() -> int:
            dm = self.module_users_mask(path)
            return -1 if dm < 0 else (selected_mask | dm)

        for line in removed:
            body = old.body_at(line)
            if body is not None:
                if isinstance(body, ast.Lambda):
                    if body.lineno == body.end_lineno:
                        func_call_anchors.add(_func_key(body))
                    else:
                        func_line_anchors.add(line)
                        inner = old.expr_at_inside(line, body.body)
                        if isinstance(inner, (ast.BoolOp, ast.IfExp)):
                            func_line_anchors.update(range(inner.lineno, (inner.end_lineno or line) + 1))
                else:
                    stmt = old.statement_at(line)
                    if stmt is not None:
                        func_line_anchors.add(line)
                        func_line_anchors.update(statement_header_lines(stmt))
                    else:
                        func_call_anchors.add(_func_key(body))
            else:
                stmt = old.statement_at(line)
                if stmt is None:
                    return _module_fallback()
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    outer_fn = old.func_of_node.get(id(stmt))
                    if outer_fn is not None:
                        func_call_anchors.add(_func_key(outer_fn))
                        continue
                    if new_func_idx is None:
                        new_func_idx = _build_func_index(new)
                    okey = _owner_key_for_stmt(old, stmt)
                    new_fn = new_func_idx.get((okey, stmt.name))
                    func_header_edits[id(stmt)] = (okey, stmt, new_fn)
                    continue
                name = _stmt_assign_name(stmt)
                if name is None:
                    return _module_fallback()
                if new_assign_idx is None:
                    new_assign_idx = _build_assign_index(new)
                okey = _owner_key_for_stmt(old, stmt)
                new_stmt = new_assign_idx.get((okey, name))
                if new_stmt is None:
                    return _module_fallback()
                class_level_edits[id(stmt)] = (okey, stmt, new_stmt)

        insertions = Counter(added.values())
        for line, anchor in added.items():
            body = new.body_at(line)
            old_body = old.body_at(anchor)
            if body is not None and old_body is not None:
                is_lam = isinstance(body, ast.Lambda)
                if is_lam != isinstance(old_body, ast.Lambda):
                    return _module_fallback()
                if is_lam:
                    if (
                        old_body.lineno == old_body.end_lineno
                        or anchor not in removed
                        or insertions[anchor] != 1
                        or not ast_equal(old_body.args, body.args)
                    ):
                        func_call_anchors.add(_func_key(old_body))
                    else:
                        func_line_anchors.add(anchor)
                else:
                    if body.name != old_body.name:
                        return _module_fallback()
                    old_stmt = old.statement_at(anchor)
                    new_stmt = new.statement_at(line)
                    if (
                        anchor not in removed
                        or insertions[anchor] != 1
                        or old_stmt is None
                        or new_stmt is None
                        or type(old_stmt) is not type(new_stmt)
                        or old_stmt.col_offset != new_stmt.col_offset
                    ):
                        func_call_anchors.add(_func_key(old_body))
                    else:
                        func_line_anchors.update(statement_header_lines(old_stmt))
            elif body is not None or old_body is not None:
                # Check if a nested def inside an outer function was added/edited
                outer_old = old.func_of_node.get(id(old_body)) if old_body is not None else old.body_at(anchor)
                outer_new = new.func_of_node.get(id(body)) if body is not None else None
                if outer_old is not None and isinstance(body or old_body, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_call_anchors.add(_func_key(outer_old))
                    continue
                return _module_fallback()
            else:
                new_stmt = new.statement_at(line)
                if new_stmt is None:
                    return _module_fallback()
                if isinstance(new_stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    outer_fn = new.func_of_node.get(id(new_stmt))
                    if outer_fn is not None:
                        old_outer = old.body_at(anchor)
                        if old_outer is not None:
                            func_call_anchors.add(_func_key(old_outer))
                            continue
                        return _module_fallback()
                    if old_func_idx is None:
                        old_func_idx = _build_func_index(old)
                    okey = _owner_key_for_stmt(new, new_stmt)
                    old_fn = old_func_idx.get((okey, new_stmt.name))
                    if old_fn is None:
                        return _module_fallback()
                    func_header_edits[id(old_fn)] = (okey, old_fn, new_stmt)
                    continue
                name = _stmt_assign_name(new_stmt)
                if name is None:
                    return _module_fallback()
                if old_assign_idx is None:
                    old_assign_idx = _build_assign_index(old)
                okey = _owner_key_for_stmt(new, new_stmt)
                old_stmt = old_assign_idx.get((okey, name))
                if old_stmt is None:
                    return _module_fallback()
                class_level_edits[id(old_stmt)] = (okey, old_stmt, new_stmt)

        # Evaluate function header/default-argument edits
        file_func_col = self.func_calls_collection.get(path, {})
        for _, old_fn, new_fn in func_header_edits.values():
            if new_fn is not None and _is_safe_default_only_change(old_fn, new_fn):
                fk = _func_key(old_fn)
                if fk not in file_func_col:
                    func_call_anchors.add(fk)
                    continue
            return _module_fallback()

        # Evaluate class/module-level non-body edits
        for owner_key, old_stmt, new_stmt in class_level_edits.values():
            if _dict_assign_target(old_stmt) is not None and _dict_assign_target(new_stmt) is not None:
                res = self._handle_class_dict_edit(path, owner_key, old_stmt, new_stmt)
                if res is not None:
                    selected_mask |= res
                    continue
                dm = self.dialect_users_mask(path)
                if dm < 0:
                    return -1
                selected_mask |= dm
                continue

            if ":" in owner_key:
                res = self._handle_class_scalar_edit(owner_key, old_stmt, new_stmt)
                if res is not None:
                    selected_mask |= res
                    continue
                dm = self.dialect_users_mask(path)
                if dm < 0:
                    return -1
                selected_mask |= dm
                continue

            return _module_fallback()

        # Evaluate function/lambda body edits
        file_func_lines = self.func_lines.get(path, {})
        file_func_calls = self.func_calls.get(path, {})
        file_init_col = self.init_collection.get(path, {})

        for fk in func_call_anchors:
            if fk in file_func_col:
                return _module_fallback()
            selected_mask |= file_func_calls.get(fk, 0)

        for ln in func_line_anchors:
            if ln in file_init_col:
                # Function line executed during collection (import time)
                return _module_fallback()
            selected_mask |= file_func_lines.get(ln, 0)

        return selected_mask

    def select_static(self, patch: str, checkout: Path) -> tuple[set[int], dict[str, bytes | None], set[int], bool]:
        """Return (selected_numbers, patched_files, fixed_fails, fixture_only)."""
        selected_mask = 0
        fixture_readers = 0
        patched_files: dict[str, bytes | None] = {}
        fixture_only = True
        fallback_all = False
        try:
            for change in parse_patch(patch):
                path = change.path
                if change.after_path is None:
                    # Deleted file
                    fixture_only = False
                    patched_files[path] = None
                    if path.endswith(".py") and path.startswith(("sqlglot/", "tests/")):
                        dm = self.module_users_mask(path)
                    elif path in self.resources:
                        dm = self.resource_users_mask(path)
                    else:
                        dm = -1
                    if dm < 0:
                        fallback_all = True
                    else:
                        selected_mask |= dm
                    continue

                if change.before_path is None:
                    # Newly added file
                    fixture_only = False
                    _, after, _, _ = apply_text(change, checkout)
                    patched_files[path] = after.encode("utf-8")
                    fallback_all = True
                    continue

                if path.endswith(".py") and path.startswith(("sqlglot/", "tests/")):
                    fixture_only = False
                    before, after, removed, added = apply_python(change, checkout)
                    patched_files[path] = after.encode("utf-8")
                    try:
                        res = self.python_users_mask(path, before, after, removed, added)
                    except SyntaxError:
                        res = self.module_users_mask(path)
                    if res < 0:
                        fallback_all = True
                    else:
                        selected_mask |= res
                elif path in self.resources:
                    before, after, _, _ = apply_text(change, checkout)
                    patched_files[path] = after.encode("utf-8")
                    if is_fixture_comment_only(path, before, after):
                        continue
                    res = self.resource_users_mask(path)
                    if res < 0:
                        fallback_all = True
                    else:
                        selected_mask |= res
                        if res.bit_count() == 1:
                            fixture_readers |= res
                else:
                    before, after, _, _ = apply_text(change, checkout)
                    patched_files[path] = after.encode("utf-8")
                    fallback_all = True
        except (OSError, UnicodeError, ValueError, StopIteration, RecursionError):
            return self.all_tests.copy(), patched_files, set(), False

        if fallback_all:
            selected = self.all_tests.copy()
        else:
            selected = self._mask_to_numbers(selected_mask)
        if self.unverified_numbers:
            selected |= self.unverified_numbers
        fixed_fails = self._mask_to_numbers(fixture_readers) | self.unverified_numbers
        return selected, patched_files, fixed_fails, fixture_only


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/data"))
    parser.add_argument("--checkout", type=Path, default=Path("/sqlglot"))
    parser.add_argument("--coverage", type=Path, default=APP / "coverage.pkl.zlib")
    args = parser.parse_args()

    tests = json.loads((args.data / "tests.json").read_text(encoding="utf-8"))["tests"]
    coverage = pickle.loads(zlib.decompress(args.coverage.read_bytes()))
    index = Index(tests, coverage)

    jobs = []
    for patch_path in sorted((args.data / "commits").glob("*.patch")):
        patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
        selected, patched_files, fixed_fails, fixture_only = index.select_static(
            patch_text, args.checkout
        )
        jobs.append({
            "commit": patch_path.stem,
            "selected": selected,
            "patched_files": patched_files,
            "fixed_fails": fixed_fails,
            "fixture_only": fixture_only,
        })

    verify_batch(jobs, index, args.checkout)

    for job in jobs:
        workers = schedule(job["selected"], index.durations)
        print(json.dumps({"commit": job["commit"], "workers": workers}, separators=(",", ":")))


if __name__ == "__main__":
    main()
