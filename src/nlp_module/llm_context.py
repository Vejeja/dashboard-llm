from .llm_strategy import LLMStrategy

class LLMClient:
    def __init__(self, strategy: LLMStrategy):
        self.strategy = strategy

    def generate(self, prompt: str, **options):
        return self.strategy.generate(prompt, **options)
