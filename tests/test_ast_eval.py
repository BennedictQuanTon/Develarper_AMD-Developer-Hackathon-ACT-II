import pytest
from agent.ast_eval import evaluate_math_expression

def test_valid_expressions() -> None:
    assert evaluate_math_expression("342 * 12") == "4104"
    assert evaluate_math_expression("12 + 34") == "46"
    assert evaluate_math_expression("100 / 4") == "25"
    assert evaluate_math_expression("100 / 3") == str(100 / 3)
    assert evaluate_math_expression("2^8") == "256"
    assert evaluate_math_expression("2**8") == "256"
    assert evaluate_math_expression("10 % 3") == "1"
    assert evaluate_math_expression("-5 + 10") == "5"
    assert evaluate_math_expression("(5 + 5) * 2") == "20"

def test_prefix_cleaning() -> None:
    assert evaluate_math_expression("Calculate: 10 + 20") == "30"
    assert evaluate_math_expression("solve 5 * 5") == "25"
    assert evaluate_math_expression("what is 8 / 2?") == "4"
    assert evaluate_math_expression("evaluate 3 + 4.") == "7"

def test_invalid_expressions() -> None:
    # Not a pure math expression
    assert evaluate_math_expression("Calculate the area of a circle with radius 5") is None
    # Contains letters other than keywords
    assert evaluate_math_expression("10 + x") is None
    # Empty or no digits/operators
    assert evaluate_math_expression("abc") is None
    assert evaluate_math_expression("123") is None
    assert evaluate_math_expression("+ - *") is None

def test_safety_checks() -> None:
    # Division by zero
    assert evaluate_math_expression("10 / 0") is None
    # Power exponent too large to prevent CPU exhaustion
    assert evaluate_math_expression("2 ^ 1001") is None
