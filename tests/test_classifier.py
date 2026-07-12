from agent.classifier import (
    ROUTE_API_CODE,
    ROUTE_API_LOGIC,
    ROUTE_API_LONG,
    ROUTE_API_MATH,
    ROUTE_LOCAL_GENERAL,
    ROUTE_LOCAL_NER,
    ROUTE_LOCAL_SENTIMENT,
    classify,
)


def test_classifier_math() -> None:
    assert classify("Calculate 342 * 12") == ROUTE_API_MATH
    assert classify("Solve for x: 2x + 5 = 15") == ROUTE_API_MATH
    assert classify("What is the percentage increase from 50 to 75?") == ROUTE_API_MATH
    assert classify("Find the derivative of x^2") == ROUTE_API_MATH


def test_classifier_code() -> None:
    assert classify("Find and fix the bug in this Python function:\ndef add(a, b): return a - b") == ROUTE_API_CODE
    assert classify("Write a Python script to scrape a website") == ROUTE_API_CODE
    assert classify("How do I compile a C++ file?") == ROUTE_API_CODE
    assert classify("Refactor this code to use list comprehension") == ROUTE_API_CODE


def test_classifier_logic() -> None:
    assert classify("If John is taller than Mary, and Mary is taller than Sue, is John taller than Sue?") == ROUTE_API_LOGIC
    assert classify("Solve this riddle: I speak without a mouth and hear without ears.") == ROUTE_API_LOGIC
    assert classify("Given the constraints: A must be next to B, and B cannot be next to C.") == ROUTE_API_LOGIC


def test_classifier_sentiment() -> None:
    assert classify("Classify the sentiment of this review: 'This is the best movie I have ever seen!'") == ROUTE_LOCAL_SENTIMENT
    assert classify("Is the tone of this text positive, negative, or neutral?") == ROUTE_LOCAL_SENTIMENT
    assert classify("What emotion is expressed in this letter?") == ROUTE_LOCAL_SENTIMENT


def test_classifier_ner() -> None:
    assert classify("Extract all named entities from this sentence: 'Google was founded by Larry Page.'") == ROUTE_LOCAL_NER
    assert classify("Identify person and organization entities in this paragraph.") == ROUTE_LOCAL_NER
    assert classify("Extract names and locations from this context in JSON format.") == ROUTE_LOCAL_NER


def test_classifier_general() -> None:
    assert classify("What is the capital of France?") == ROUTE_LOCAL_GENERAL
    assert classify("Summarize this article: ...") == ROUTE_LOCAL_GENERAL
    assert classify("Define the term photosynthesis.") == ROUTE_LOCAL_GENERAL


def test_classifier_long_context() -> None:
    long_prompt = "a" * 6001
    assert classify(long_prompt) == ROUTE_API_LONG


def test_classifier_ambiguous_prompts() -> None:
    # Code + sentiment keywords — LLM should pick API_CODE (writing code).
    result = classify("Write a Python script to calculate sentiment of text")
    assert result == ROUTE_API_CODE


def test_classifier_negative_penalties() -> None:
    # Story request with numbers — should not be routed to math.
    result = classify("Write a story about a boy who has 3 apples and 2 oranges")
    assert result == ROUTE_LOCAL_GENERAL
