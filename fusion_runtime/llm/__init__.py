"""LLM engines."""
from fusion_runtime.llm.base import ChatMessage, LLMBase, LLMResult
from fusion_runtime.llm.llama_cpp import LlamaCppLLM
from fusion_runtime.llm.openai_compat import OpenAILLM


def create_llm(config) -> LLMBase:
    """Factory function to create LLM instance from config."""
    from fusion_runtime.config import Provider
    
    if config.provider == Provider.LLAMA_CPP:
        return LlamaCppLLM(config)
    elif config.provider == Provider.OPENAI:
        return OpenAILLM(config)
    raise ValueError(f"Unknown LLM provider: {config.provider}")
