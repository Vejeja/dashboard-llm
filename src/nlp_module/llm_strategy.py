from abc import ABC, abstractmethod
from typing import Dict, Any

class LLMStrategy(ABC):
    @abstractmethod
    def generate(self, prompt: str, **options: Any) -> str:
        """
        Универсальный метод генерации текста.
        Опции зависят от реализации (model, system_prompt_name, temperature и т.д.).
        """
        ...

class OpenRouterStrategy(LLMStrategy):
    def __init__(self, api_key: str, model: str, endpoint: str, system_prompts: Dict[str, str]):
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.system_prompts = system_prompts

    def generate(self, prompt: str, system_prompt_name: str = "default", **options) -> str:
        import requests, re
        sp = self.system_prompts.get(system_prompt_name, "")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        messages = ([{"role":"system","content":sp}] if sp else []) + [{"role":"user","content":prompt}]
        payload = {"model": self.model, "messages": messages, **options}
        resp = requests.post(self.endpoint, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return re.sub(r"</?think>", "", text, flags=re.DOTALL).strip()
