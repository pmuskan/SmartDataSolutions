import ast
import operator as op
from typing import Any

from src.schemas import ToolCall

# Supported safe operators
ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _safe_eval_ast(node: ast.AST) -> int | float:
    """Recursively evaluate an AST expression with only safe math operations."""
    if isinstance(node, ast.Constant):  # Numbers
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    elif isinstance(node, ast.BinOp):  # e.g., a + b, a * b
        left = _safe_eval_ast(node.left)
        right = _safe_eval_ast(node.right)
        op_type = type(node.op)
        if op_type in ALLOWED_OPERATORS:
            if op_type == ast.Div and right == 0:
                raise ZeroDivisionError("Division by zero in calculation")
            return ALLOWED_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported binary operator: {op_type}")

    elif isinstance(node, ast.UnaryOp):  # e.g., -a, +a
        operand = _safe_eval_ast(node.operand)
        op_type = type(node.op)
        if op_type in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type}")

    raise ValueError(f"Unsupported AST node expression: {type(node)}")


def evaluate_calculator_expression(expression: str, inputs: dict[str, Any] | None = None) -> ToolCall:
    """
    Safely evaluate a mathematical expression for financial arithmetic.
    Supports YoY %, differences, ratios, margins.
    """
    inputs = inputs or {}
    clean_expr = expression.replace("$", "").replace(",", "").replace("%", "").strip()

    try:
        parsed_ast = ast.parse(clean_expr, mode="eval")
        result_val = _safe_eval_ast(parsed_ast.body)
        
        if isinstance(result_val, float):
            result_val = round(result_val, 4)

        return ToolCall(
            tool_name="calculator",
            expression=expression,
            inputs=inputs,
            result=result_val
        )
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        return ToolCall(
            tool_name="calculator",
            expression=expression,
            inputs=inputs,
            result=f"Error: {e!s}"
        )


def calculate_yoy_growth(current: float, prior: float) -> ToolCall:
    """Helper for YoY growth percentage calculation: ((current - prior) / prior) * 100"""
    expr = f"(({current} - {prior}) / {prior}) * 100"
    return evaluate_calculator_expression(expr, inputs={"current": current, "prior": prior})


def calculate_share_percentage(part: float, total: float) -> ToolCall:
    """Helper for percentage share calculation: (part / total) * 100"""
    expr = f"({part} / {total}) * 100"
    return evaluate_calculator_expression(expr, inputs={"part": part, "total": total})
