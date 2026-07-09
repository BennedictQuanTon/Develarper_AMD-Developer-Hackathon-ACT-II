import ast
import operator
import re
from collections.abc import Callable
from typing import Any


class SafeEvalVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.operators: dict[type[ast.AST], Callable[..., Any]] = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.BitXor: operator.pow,  # Map ^ to exponentiation
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

    def visit(self, node: ast.AST) -> int | float:
        val = super().visit(node)
        if isinstance(val, int | float):
            return val
        raise TypeError(f"Unexpected AST node value type: {type(val)}")

    def visit_BinOp(self, node: ast.BinOp) -> int | float:
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        if op_type in self.operators:
            # Prevent division by zero
            if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                raise ZeroDivisionError("Division by zero in AST evaluation")
            # Prevent massive exponentiation to avoid resource exhaustion
            if op_type in (ast.Pow, ast.BitXor) and right > 1000:
                raise ValueError("Exponent too large in AST evaluation")
            res = self.operators[op_type](left, right)
            if isinstance(res, int | float):
                return res
            raise TypeError("Binary operator result is not numeric")
        raise ValueError(f"Unsupported binary operator: {op_type}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> int | float:
        operand = self.visit(node.operand)
        op_type = type(node.op)
        if op_type in self.operators:
            res = self.operators[op_type](operand)
            if isinstance(res, int | float):
                return res
            raise TypeError("Unary operator result is not numeric")
        raise ValueError(f"Unsupported unary operator: {op_type}")

    def visit_Constant(self, node: ast.Constant) -> int | float:
        if isinstance(node.value, int | float):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    def visit_Expr(self, node: ast.Expr) -> int | float:
        return self.visit(node.value)

    def visit_Module(self, node: ast.Module) -> int | float:
        if len(node.body) != 1:
            raise ValueError("Module must contain exactly one expression")
        return self.visit(node.body[0])

    def visit_Expression(self, node: ast.Expression) -> int | float:
        return self.visit(node.body)

    def generic_visit(self, node: ast.AST) -> None:
        raise ValueError(f"Forbidden syntax node: {type(node)}")


def evaluate_math_expression(prompt: str) -> str | None:
    # Clean the prompt to extract potential expression
    p = prompt.strip().lower()

    # Strip common prefixes
    prefixes = ["calculate", "solve", "evaluate", "what is", "compute", "can you calculate", "can you solve", "please calculate", ":"]

    for prefix in prefixes:
        if p.startswith(prefix):
            p = p[len(prefix) :].strip()

    # Remove leading/trailing punctuation except parenthesis
    p = p.strip("? \t\r\n.")

    # Regex to check if the string contains only valid math expression characters
    # Allowed: digits, whitespace, operators (+ - * / % ^ ** //), parenthesis, decimals
    if not re.match(r"^[\d\s\+\-\*\/\%\^\(\)\.\/]+$", p):
        return None

    # Extra safety check: must contain at least one digit and one operator to be a valid expression
    if not any(c.isdigit() for c in p):
        return None
    if not any(c in p for c in "+-*/%^"):
        return None

    try:
        # Parse expression into AST
        tree = ast.parse(p, mode="eval")
        # Evaluate safely
        visitor = SafeEvalVisitor()
        result = visitor.visit(tree)

        # Convert result to string, format floats cleanly
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
    except Exception:
        return None
