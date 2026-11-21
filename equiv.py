"""Sound semantic equivalence checks for Python and SQL fixture patches."""

from __future__ import annotations

import ast
import copy
import typing as t


_BUILTIN_SAFE_NONE_EQ_TYPES = frozenset({
    "None",
    "NoneType",
    "bool",
    "int",
    "float",
    "str",
    "bytes",
    "bytearray",
    "list",
    "dict",
    "set",
    "frozenset",
    "tuple",
    "range",
    "slice",
    "type",
    "object",
    "complex",
    "Path",
    "PurePosixPath",
    "PureWindowsPath",
    "PurePath",
    "PosixPath",
    "WindowsPath",
    "Enum",
    "IntEnum",
    "StrEnum",
    "AutoName",
})

_UNSAFE_TYPE_NAMES = frozenset({
    "Any",
    "t.Any",
    "typing.Any",
    "reverse_key",
})

_IDENTITY_EQ_OP_PAIRS = frozenset({
    (ast.Is, ast.Eq),
    (ast.Eq, ast.Is),
    (ast.IsNot, ast.NotEq),
    (ast.NotEq, ast.IsNot),
})


def ast_equal(a: t.Any, b: t.Any) -> bool:
    """Fast structural AST equality ignoring source positions."""
    if type(a) is not type(b):
        return False
    if isinstance(a, ast.TypeIgnore):
        return a.tag == b.tag
    if isinstance(a, ast.AST):
        if getattr(a, "type_comment", None) != getattr(b, "type_comment", None):
            return False
        for field in a._fields:
            if not ast_equal(getattr(a, field), getattr(b, field)):
                return False
        return True
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(ast_equal(x, y) for x, y in zip(a, b))
    return a == b and type(a) is type(b)


def is_fixture_comment_only(path: str, before: str, after: str) -> bool:
    """Return True if a SQL fixture differs only in `--` comments or blank lines."""
    if not (path.startswith("tests/fixtures/") and path.endswith(".sql")):
        return False
    def _filter(s: str) -> list[str]:
        return [line for line in s.splitlines() if line and not line.startswith("--")]
    return _filter(before) == _filter(after)


def _diff_ast_nodes(a: t.Any, b: t.Any, diffs: list[tuple[ast.Compare, ast.Compare]]) -> bool:
    """Collect differing ast.Compare nodes if all other AST structure is identical."""
    if type(a) is not type(b):
        return False
    if isinstance(a, ast.Compare):
        if (
            len(a.ops) == 1
            and len(b.ops) == 1
            and len(a.comparators) == 1
            and len(b.comparators) == 1
            and ast_equal(a.left, b.left)
            and ast_equal(a.comparators[0], b.comparators[0])
        ):
            if type(a.ops[0]) is not type(b.ops[0]):
                diffs.append((a, b))
            return True
        # Chained comparisons cannot be rewritten by this rule. Unchanged
        # chains elsewhere in the module still permit inspecting other edits.
        return ast_equal(a, b)
    if isinstance(a, ast.TypeIgnore):
        return a.tag == b.tag
    if isinstance(a, ast.AST):
        if getattr(a, "type_comment", None) != getattr(b, "type_comment", None):
            return False
        for field in a._fields:
            if not _diff_ast_nodes(getattr(a, field), getattr(b, field), diffs):
                return False
        return True
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(_diff_ast_nodes(x, y, diffs) for x, y in zip(a, b))
    return a == b and type(a) is type(b)


def _unparse_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _unparse_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Constant) and node.value is None:
        return "None"
    return None


def _extract_annotation_leaf_names(ann: ast.AST | None) -> set[str] | None:
    """Extract leaf type names from a PEP 604 / typing union annotation."""
    if ann is None:
        return None
    if isinstance(ann, ast.Constant):
        if ann.value is None:
            return {"None"}
        if isinstance(ann.value, str):
            try:
                return _extract_annotation_leaf_names(ast.parse(ann.value, mode="eval").body)
            except SyntaxError:
                return None
        return None
    if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
        left = _extract_annotation_leaf_names(ann.left)
        right = _extract_annotation_leaf_names(ann.right)
        if left is None or right is None:
            return None
        return left | right
    if isinstance(ann, ast.Subscript):
        base = _unparse_name(ann.value)
        if base in ("Optional", "t.Optional", "typing.Optional"):
            inner = _extract_annotation_leaf_names(ann.slice)
            return (inner | {"None"}) if inner is not None else None
        if base in ("Union", "t.Union", "typing.Union"):
            items = ann.slice.elts if isinstance(ann.slice, ast.Tuple) else [ann.slice]
            out: set[str] = set()
            for item in items:
                sub = _extract_annotation_leaf_names(item)
                if sub is None:
                    return None
                out.update(sub)
            return out
        # Generic container like dict[str, int] or list[Expr] -> base container type
        return {base} if base else None
    if isinstance(ann, (ast.Name, ast.Attribute)):
        name = _unparse_name(ann)
        return {name} if name else None
    return None


def _type_names_safe_for_none_eq(names: set[str] | None, index: t.Any) -> bool:
    if not names:
        return False
    safe_classes = index.safe_none_eq_classes
    unsafe_classes = index.unsafe_none_eq_classes
    for raw in names:
        short = raw.rsplit(".", 1)[-1]
        if raw in _UNSAFE_TYPE_NAMES or short in _UNSAFE_TYPE_NAMES or short in unsafe_classes:
            return False
        if raw in _BUILTIN_SAFE_NONE_EQ_TYPES or short in _BUILTIN_SAFE_NONE_EQ_TYPES:
            continue
        if short in safe_classes or raw in safe_classes:
            continue
        return False
    return True


def _build_scopes(tree: ast.AST):
    """Map AST nodes to enclosing FunctionDef/AsyncFunctionDef and ClassDef."""
    enclosing_func: dict[int, ast.FunctionDef | ast.AsyncFunctionDef | None] = {}
    enclosing_class: dict[int, ast.ClassDef | None] = {}

    def visit(node: ast.AST, cur_func, cur_cls):
        enclosing_func[id(node)] = cur_func
        enclosing_class[id(node)] = cur_cls
        next_cls = node if isinstance(node, ast.ClassDef) else cur_cls
        next_func = node if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else cur_func
        for child in ast.iter_child_nodes(node):
            visit(child, next_func, next_cls)

    visit(tree, None, None)
    return enclosing_func, enclosing_class


def _find_var_annotations(func_node: ast.FunctionDef | ast.AsyncFunctionDef | None, var_name: str) -> set[str] | None:
    if func_node is None:
        return None
    anns: list[ast.AST] = []
    for arg in (*func_node.args.posonlyargs, *func_node.args.args, *func_node.args.kwonlyargs):
        if arg.arg == var_name and arg.annotation is not None:
            anns.append(arg.annotation)
    if func_node.args.vararg and func_node.args.vararg.arg == var_name and func_node.args.vararg.annotation:
        anns.append(func_node.args.vararg.annotation)
    if func_node.args.kwarg and func_node.args.kwarg.arg == var_name and func_node.args.kwarg.annotation:
        anns.append(func_node.args.kwarg.annotation)
    for node in ast.walk(func_node):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == var_name:
            anns.append(node.annotation)
    if not anns:
        return None
    out: set[str] = set()
    for a in anns:
        leafs = _extract_annotation_leaf_names(a)
        if leafs is None:
            return None
        out.update(leafs)
    return out


def _self_attr_assignments_and_annotations(
    cls_node: ast.ClassDef | None, attr_name: str
) -> tuple[list[ast.AST], set[str] | None]:
    """Return all RHS expressions assigned to `self.<attr_name>` and union of annotations in `cls_node`."""
    if cls_node is None:
        return [], None
    rhs_list: list[ast.AST] = []
    ann_names: set[str] = set()
    has_ann = False
    for node in ast.walk(cls_node):
        if isinstance(node, ast.AnnAssign):
            target = node.target
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == attr_name
            ):
                has_ann = True
                leafs = _extract_annotation_leaf_names(node.annotation)
                if leafs is None:
                    return rhs_list, None
                ann_names.update(leafs)
                if node.value is not None:
                    rhs_list.append(node.value)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr == attr_name
                ):
                    rhs_list.append(node.value)
        elif isinstance(node, ast.AugAssign):
            target = node.target
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == attr_name
            ):
                return rhs_list, None
    return rhs_list, (ann_names if has_ann else None)


def _is_none_expr(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _is_none_safe_expr(
    expr: ast.AST,
    func_node: ast.FunctionDef | ast.AsyncFunctionDef | None,
    cls_node: ast.ClassDef | None,
    index: t.Any,
) -> bool:
    if isinstance(expr, ast.Constant):
        return True
    if isinstance(expr, ast.Name):
        names = _find_var_annotations(func_node, expr.id)
        return _type_names_safe_for_none_eq(names, index)
    if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name) and expr.value.id == "self":
        rhs_list, ann_names = _self_attr_assignments_and_annotations(cls_node, expr.attr)
        if not _type_names_safe_for_none_eq(ann_names, index):
            return False
        # Verify no assignment directly assigns a known unsafe value
        for rhs in rhs_list:
            if isinstance(rhs, ast.Call) and _unparse_name(rhs.func) == "reverse_key":
                return False
        return True
    return False


def _extract_enum_member(node: ast.AST, index: t.Any) -> tuple[str, dict] | None:
    """If node looks like `<EnumName>.<MEMBER>`, return (short_enum_name, enum_info)."""
    if not isinstance(node, ast.Attribute):
        return None
    base = _unparse_name(node.value)
    if not base:
        return None
    short = base.rsplit(".", 1)[-1]
    info = index.enum_classes_by_name.get(short)
    if not info or not info.get("distinct_values"):
        return None
    if node.attr not in info.get("members", ()):
        return None
    return short, info


def _is_pure_enum_expr(
    expr: ast.AST,
    enum_short: str,
    enum_info: dict,
    path: str,
    func_node: ast.FunctionDef | ast.AsyncFunctionDef | None,
    cls_node: ast.ClassDef | None,
    index: t.Any,
) -> bool:
    """Prove `expr` can only evaluate to an instance of `enum_short` (or None) if not a pure Enum."""
    if enum_info.get("is_pure"):
        # Pure Enum does not inherit int/str/float, so == only matches identical Enum members
        # unless the other side defines a custom __eq__ like reverse_key or Dialect.
        if _is_none_safe_expr(expr, func_node, cls_node, index):
            return True
        return False

    # Value-inheriting Enum (e.g. (str, Enum) like NormalizationStrategy):
    # MUST prove `expr` is strictly an instance of `enum_short` or `None`!
    if isinstance(expr, ast.Name):
        names = _find_var_annotations(func_node, expr.id)
        if not names:
            return False
        return all(n.rsplit(".", 1)[-1] in (enum_short, "None", "NoneType") for n in names)

    if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name) and expr.value.id == "self":
        if cls_node is None:
            return False
        rhs_list, ann_names = _self_attr_assignments_and_annotations(cls_node, expr.attr)
        if not rhs_list:
            return False
        if ann_names is not None:
            if not all(n.rsplit(".", 1)[-1] in (enum_short, "None", "NoneType") for n in ann_names):
                return False
        cls_key = f"{path}:{cls_node.name}"
        for rhs in rhs_list:
            if isinstance(rhs, ast.Constant) and rhs.value is None:
                continue
            if isinstance(rhs, ast.Call):
                fn = _unparse_name(rhs.func)
                if fn and fn.rsplit(".", 1)[-1] == enum_short:
                    continue
                return False
            if isinstance(rhs, ast.Attribute):
                if isinstance(rhs.value, ast.Name) and rhs.value.id == "self":
                    enum_key = index.enum_typed_class_attrs.get((cls_key, rhs.attr))
                    if enum_key == enum_info["key"]:
                        continue
                    return False
                base = _unparse_name(rhs.value)
                if base and base.rsplit(".", 1)[-1] == enum_short:
                    continue
                return False
            return False
        return True

    return False


def _has_future_annotations(tree: ast.AST) -> bool:
    if not isinstance(tree, ast.Module):
        return False
    for stmt in tree.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            continue
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
            if any(al.name == "annotations" for al in stmt.names):
                return True
            continue
        break
    return False


def _uses_dataclass_or_assigns_tc(tree: ast.AST) -> tuple[bool, bool, bool]:
    """Return (has_dataclass, assigns_type_checking, imports_typing_tc)."""
    has_dc = False
    assigns_tc = False
    has_typing_import = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for dec in node.decorator_list:
                name = _unparse_name(dec.func if isinstance(dec, ast.Call) else dec)
                if name and name.rsplit(".", 1)[-1] == "dataclass":
                    has_dc = True
        elif isinstance(node, ast.Global):
            if any(n in ("TYPE_CHECKING", "t", "typing") for n in node.names):
                assigns_tc = True
    if isinstance(tree, ast.Module):
        for stmt in tree.body:
            if isinstance(stmt, ast.Import):
                for al in stmt.names:
                    if al.name == "typing" and (al.asname or "typing") in ("t", "typing"):
                        has_typing_import = True
            elif isinstance(stmt, ast.ImportFrom):
                if stmt.module == "typing" and any(al.name == "TYPE_CHECKING" and not al.asname for al in stmt.names):
                    has_typing_import = True
            elif isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name) and tgt.id in ("TYPE_CHECKING", "t", "typing"):
                        assigns_tc = True
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id in ("TYPE_CHECKING", "t", "typing"):
                assigns_tc = True
    return has_dc, assigns_tc, has_typing_import


def _is_type_checking_guard(test: ast.AST) -> bool:
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        if isinstance(test.value, ast.Name) and test.value.id in ("t", "typing"):
            return True
    return False


def _strip_type_checking_in_body(stmts: list[ast.stmt]) -> list[ast.stmt]:
    out: list[ast.stmt] = []
    for stmt in stmts:
        if isinstance(stmt, ast.If) and _is_type_checking_guard(stmt.test):
            out.extend(_strip_type_checking_in_body(stmt.orelse))
        else:
            out.append(stmt)
    return out


def _strip_docstring_in_body(stmts: list[ast.stmt]) -> list[ast.stmt]:
    if (
        stmts
        and isinstance(stmts[0], ast.Expr)
        and isinstance(stmts[0].value, ast.Constant)
        and isinstance(stmts[0].value.value, str)
    ):
        if len(stmts) == 1:
            return [ast.Pass()]
        return stmts[1:]
    return stmts


def _normalize_annotations_and_tc(
    tree: ast.AST, strip_annotations: bool, strip_tc: bool, strip_docstrings: bool = False
) -> ast.AST:
    cloned = copy.deepcopy(tree)
    for node in ast.walk(cloned):
        if strip_docstrings and isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            node.body = _strip_docstring_in_body(node.body)
        if strip_tc:
            for attr in ("body", "orelse", "finalbody"):
                stmts = getattr(node, attr, None)
                if isinstance(stmts, list) and stmts and all(isinstance(s, ast.stmt) for s in stmts):
                    setattr(node, attr, _strip_type_checking_in_body(stmts))
        if strip_annotations:
            if isinstance(node, ast.arg):
                node.annotation = None
                node.type_comment = None
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                node.returns = None
                node.type_comment = None
            elif isinstance(node, ast.AnnAssign):
                node.annotation = ast.Constant(value=None)
            elif hasattr(node, "type_comment"):
                node.type_comment = None
    return cloned


def is_semantically_equivalent(path: str, old_tree: ast.AST, new_tree: ast.AST, index: t.Any) -> bool:
    """Return True iff every AST difference is provably behavior-preserving."""
    if ast_equal(old_tree, new_tree):
        return True

    old_dc, old_atc, old_ti = _uses_dataclass_or_assigns_tc(old_tree)
    new_dc, new_atc, new_ti = _uses_dataclass_or_assigns_tc(new_tree)
    strip_tc = (not old_atc) and (not new_atc) and old_ti and new_ti
    strip_ann = (
        not path.startswith("sqlglot/typing/")
        and not old_dc
        and not new_dc
        and _has_future_annotations(old_tree)
        and _has_future_annotations(new_tree)
    )
    strip_docs = path.startswith("sqlglot/")
    if strip_tc or strip_ann or strip_docs:
        norm_old = _normalize_annotations_and_tc(old_tree, strip_ann, strip_tc, strip_docs)
        norm_new = _normalize_annotations_and_tc(new_tree, strip_ann, strip_tc, strip_docs)
        if ast_equal(norm_old, norm_new):
            return True
    else:
        norm_old = old_tree
        norm_new = new_tree

    diffs: list[tuple[ast.Compare, ast.Compare]] = []
    if not _diff_ast_nodes(norm_old, norm_new, diffs) or not diffs:
        return False

    enclosing_func, enclosing_class = _build_scopes(norm_old)

    for old_cmp, new_cmp in diffs:
        pair = (type(old_cmp.ops[0]), type(new_cmp.ops[0]))
        if pair not in _IDENTITY_EQ_OP_PAIRS:
            return False

        left = old_cmp.left
        right = old_cmp.comparators[0]
        func_node = enclosing_func.get(id(old_cmp))
        cls_node = enclosing_class.get(id(old_cmp))

        # Case A: comparison against None
        if _is_none_expr(right):
            if _is_none_safe_expr(left, func_node, cls_node, index):
                continue
            return False
        if _is_none_expr(left):
            if _is_none_safe_expr(right, func_node, cls_node, index):
                continue
            return False

        # Case B: comparison against a distinct-valued Enum member
        enum_right = _extract_enum_member(right, index)
        if enum_right is not None:
            enum_short, enum_info = enum_right
            if _is_pure_enum_expr(left, enum_short, enum_info, path, func_node, cls_node, index):
                continue
            return False

        enum_left = _extract_enum_member(left, index)
        if enum_left is not None:
            enum_short, enum_info = enum_left
            if _is_pure_enum_expr(right, enum_short, enum_info, path, func_node, cls_node, index):
                continue
            return False

        return False

    return True
