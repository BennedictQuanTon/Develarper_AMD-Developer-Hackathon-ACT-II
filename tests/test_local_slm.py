import os
import pytest
from engines.local_slm import LocalSLMEngine

@pytest.mark.skipif(not os.path.exists("models/qwen2.5-1.5b-instruct-q4_k_m.gguf"), reason="Model weights not found")
def test_local_slm_basic() -> None:
    # Use Metal on Mac for faster test execution
    os.environ["LOCAL_N_GPU_LAYERS"] = "-1"
    
    engine = LocalSLMEngine.get_instance()
    assert engine.model is not None
    
    response = engine.generate(
        prompt="Hi",
        system_prompt="You are a helpful assistant. Output exactly 'Hello' and nothing else.",
        max_tokens=10
    )
    assert response.strip() == "Hello"

@pytest.mark.skipif(not os.path.exists("models/qwen2.5-1.5b-instruct-q4_k_m.gguf"), reason="Model weights not found")
def test_local_slm_escalation() -> None:
    engine = LocalSLMEngine.get_instance()
    
    response = engine.generate(
        prompt="This is a complex question requiring escalation.",
        system_prompt="Always output __ESCALATE__",
        max_tokens=10
    )
    assert response.strip() == "__ESCALATE__"

if __name__ == "__main__":
    test_local_slm_basic()
    test_local_slm_escalation()
    print("All local SLM tests passed successfully!")
