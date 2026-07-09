import logging
import os
from typing import Optional
from llama_cpp import Llama

logger = logging.getLogger(__name__)

class LocalSLMEngine:
    _instance: Optional['LocalSLMEngine'] = None

    @classmethod
    def get_instance(cls) -> 'LocalSLMEngine':
        if cls._instance is None:
            model_path = os.environ.get("LOCAL_MODEL_PATH", "models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
            cls._instance = cls(model_path=model_path)
        return cls._instance

    def __init__(self, model_path: str) -> None:
        n_ctx = int(os.environ.get("LOCAL_N_CTX", "2048"))
        n_threads = int(os.environ.get("LOCAL_N_THREADS", "2"))
        # Default n_gpu_layers to -1 (automatic metal) on Mac/Jupyter, but 0 in production
        # We can dynamically set this via environment variables.
        n_gpu_layers = int(os.environ.get("LOCAL_N_GPU_LAYERS", "0"))
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        logger.info(f"Initializing local SLM Llama engine with model: {model_path}")
        logger.info(f"Parameters: n_ctx={n_ctx}, n_threads={n_threads}, n_gpu_layers={n_gpu_layers}")
        
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )

    def generate(
        self, 
        prompt: str, 
        system_prompt: str = "", 
        max_tokens: int = 250,
        temperature: float = 0.1,
        grammar=None
    ) -> str:
        # Format using Qwen2.5 Chat Template
        formatted_prompt = ""
        if system_prompt:
            formatted_prompt += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        formatted_prompt += f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        response = self.model(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>", "<|im_start|>", "assistant\n"],
            echo=False,
            grammar=grammar
        )
        
        choice = response["choices"][0]
        text = choice["text"].strip()
        
        # Check for context ceiling or max tokens truncation
        if choice["finish_reason"] == "length":
            logger.warning("Local SLM response was truncated due to length limits. Escalating.")
            return "__ESCALATE__"
            
        # Check if model self-escalated
        if "__ESCALATE__" in text:
            logger.info("Local SLM requested escalation.")
            return "__ESCALATE__"
            
        return text
