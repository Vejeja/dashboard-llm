# src/nlp_module/llm.py

import requests
import re
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict
from json import JSONDecodeError


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt_name: str = "default") -> str:
        ...


class OpenRouterClient(BaseLLM):
    def __init__(
        self,
        api_key: str,
        model: str,
        system_prompts: Optional[Dict[str, str]] = None,
        prompts_dir: Optional[str] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.system_prompts = system_prompts or {}
        self.prompts_dir = prompts_dir or os.getenv("SYSTEM_PROMPTS_DIR", "prompts")

        # ← вот тут исправили на api.openrouter.ai
        self.endpoint = os.getenv(
            "OPENROUTER_ENDPOINT"
        )

    def _load_prompt_from_file(self, name: str) -> str:
        fn = os.path.join(self.prompts_dir, f"{name}.txt")
        try:
            return open(fn, encoding="utf-8").read()
        except FileNotFoundError:
            return ""

    def generate(self, prompt: str, system_prompt_name: str = "default") -> str:
        # сначала из переданного словаря
        sp = self.system_prompts.get(system_prompt_name, "")
        # если нет — из файла
        if not sp:
            sp = self._load_prompt_from_file(system_prompt_name)

        headers = {"Authorization": f"Bearer {self.api_key}"}
        messages = []
        if sp:
            messages.append({"role": "system", "content": sp})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "messages": messages}

        resp = requests.post(self.endpoint, json=payload, headers=headers)
        resp.raise_for_status()

        try:
            data = resp.json()
        except JSONDecodeError:
            ct = resp.headers.get("Content-Type", "")
            snippet = resp.text[:500].replace("\n", "\\n")
            raise RuntimeError(
                f"Ожидался JSON, но получили {resp.status_code} {ct}\n"
                f"Snippet: {snippet}"
            )

        content = data["choices"][0]["message"]["content"]
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        return content.replace("<think>", "").replace("</think>", "").strip()


def LLMClient(
    api_key: str,
    model: str,
    provider: str = "openrouter",
    system_prompts: Optional[Dict[str, str]] = None,
    prompts_dir: Optional[str] = None,
) -> BaseLLM:
    if provider == "openrouter":
        return OpenRouterClient(api_key, model, system_prompts, prompts_dir)
    raise ValueError(f"Unknown LLM provider: {provider}")
