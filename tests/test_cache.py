from agent.cache import SemanticCache

def test_cache_basic_get_set() -> None:
    cache = SemanticCache()
    prompt = "What is the capital of France?"
    answer = "Paris"
    
    assert cache.get(prompt) is None
    cache.set(prompt, answer)
    assert cache.get(prompt) == answer

def test_cache_normalization() -> None:
    cache = SemanticCache()
    # Mixed case, punctuation, extra spacing
    prompt_1 = "What is the capital of France?"
    prompt_2 = "  what is the, capital of... France  "
    
    cache.set(prompt_1, "Paris")
    # Should resolve to the same hash
    assert cache.get(prompt_2) == "Paris"

def test_cache_clear() -> None:
    cache = SemanticCache()
    cache.set("test", "result")
    cache.clear()
    assert cache.get("test") is None
