"""A tiny, safe expression language for recipe JSON ("L // 2", "1 if thin and w >= 4 else 2", "max(2, L - 3)").

Only arithmetic, comparisons, and/or/not, conditional expressions, a few functions and the names the recipe defines are allowed:
no attribute access, no calls to anything else, no strings. Everything evaluates to an int or a bool; anything else is an error.
"""

import ast
import operator

FUNCS = {"min": min, "max": max, "abs": abs, "round": round}
BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
CMPS = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
MAX_LEN = 200


class ExprError(ValueError):
    pass


def evaluate(text, names):
    """Value of `text` (a string, or an int/bool which is returned as is) with `names` {name: int|bool}."""
    if isinstance(text, bool) or isinstance(text, int):
        return text
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_LEN:
        raise ExprError(f"bad expression {text!r}")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as e:
        raise ExprError(f"cannot read expression {text!r}: {e.msg}") from None
    return _ev(tree.body, names, text)


def _ev(n, names, src):
    if isinstance(n, ast.Constant):
        if isinstance(n.value, (bool, int)):
            return n.value
        raise ExprError(f"only whole numbers are allowed in {src!r}")
    if isinstance(n, ast.Name):
        if n.id in ("true", "True"):
            return True
        if n.id in ("false", "False"):
            return False
        if n.id not in names:
            raise ExprError(f"unknown name {n.id!r} in {src!r}")
        return names[n.id]
    if isinstance(n, ast.BinOp) and type(n.op) in BINOPS:
        a, b = _ev(n.left, names, src), _ev(n.right, names, src)
        if isinstance(n.op, (ast.FloorDiv, ast.Mod)) and b == 0:
            raise ExprError(f"division by zero in {src!r}")
        return BINOPS[type(n.op)](a, b)
    if isinstance(n, ast.UnaryOp):
        v = _ev(n.operand, names, src)
        if isinstance(n.op, ast.USub):
            return -v
        if isinstance(n.op, ast.Not):
            return not v
    if isinstance(n, ast.BoolOp):
        vals = (_ev(v, names, src) for v in n.values)
        return all(vals) if isinstance(n.op, ast.And) else any(vals)
    if isinstance(n, ast.Compare) and all(type(o) in CMPS for o in n.ops):
        left = _ev(n.left, names, src)
        for op, right in zip(n.ops, n.comparators):
            r = _ev(right, names, src)
            if not CMPS[type(op)](left, r):
                return False
            left = r
        return True
    if isinstance(n, ast.IfExp):
        return _ev(n.body, names, src) if _ev(n.test, names, src) else _ev(n.orelse, names, src)
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCS and not n.keywords:
        return FUNCS[n.func.id](*[_ev(a, names, src) for a in n.args])
    raise ExprError(f"not allowed in a recipe expression: {src!r}")
