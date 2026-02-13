from typing import Optional, List, Mapping, Any
from tools.ollama_client import generate as ollama_generate

try:
    from langchain.llms.base import LLM

    class OllamaLLM(LLM):
        """LangChain LLM wrapper for Ollama HTTP client (simple)._call returns generated text."""

        model: Optional[str]

        def __init__(self, model: Optional[str] = None):
            self.model = model

        @property
        def _llm_type(self) -> str:
            return "ollama"

        def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
            # call our ollama_client.generate
            return ollama_generate(prompt, model=self.model)

        def _identifying_params(self) -> Mapping[str, Any]:
            return {"model": self.model}
except Exception:
    # langchain not installed; provide a lightweight fallback class
    class OllamaLLM:
        def __init__(self, model: Optional[str] = None):
            self.model = model

        def __call__(self, prompt: str) -> str:
            return ollama_generate(prompt, model=self.model)

