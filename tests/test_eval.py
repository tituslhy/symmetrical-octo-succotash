from src.eval import numerical_match, parse_answer
from src.prompt import format_table


# --- parse_answer ---
def test_parse_answer_standard():
    response = "STEPS:\n1. found it\nANSWER: 0.14136"
    answer, error = parse_answer(response)
    assert answer == "0.14136"
    assert error is False

def test_parse_answer_missing():
    response = "I think the answer is around 14%"
    answer, error = parse_answer(response)
    assert error is True

def test_parse_answer_case_insensitive():
    response = "answer: 206588.0"
    answer, error = parse_answer(response)
    assert answer == "206588.0"
    assert error is False


# --- numerical_match ---
def test_numerical_match_exact():
    assert numerical_match("206588.0", 206588.0) is True

def test_numerical_match_tolerance():
    # rounding differences should pass
    assert numerical_match("0.14136", 0.14136) is True
    assert numerical_match("0.141", 0.14136) is True  # within 1%

def test_numerical_match_percentage_string():
    # "14.1%" should match gold of 0.141
    assert numerical_match("14.1%", 0.141) is True

def test_numerical_match_wrong_sign():
    # the Turn 3 failure we saw — 10.0 vs -10.0
    assert numerical_match("10.0", -10.0) is False

def test_numerical_match_zero():
    assert numerical_match("0.0", 0.0) is True

def test_numerical_match_string_fallback():
    # non-numeric gold falls back to string match
    assert numerical_match("n/a", "n/a") is True
    assert numerical_match("yes", "no") is False


# --- format_table ---
def test_format_table_basic():
    table = {
        "2009": {"net income": 103102.0, "net cash": 206588.0},
        "2008": {"net income": 104222.0, "net cash": 181001.0},
    }
    result = format_table(table)
    assert "net income" in result
    assert "103102.0" in result
    assert "2009" in result

def test_format_table_empty():
    assert format_table({}) == ""
