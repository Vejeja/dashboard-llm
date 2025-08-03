from abc import ABC, abstractmethod
from typing import List, Any
import requests

class EmbedStrategy(ABC):
    @abstractmethod
    def embed(self, text: str, **options: Any) -> List[float]:
        """
        Универсальный метод для эмбеддинга текста.
        """
        ...

# Реализация через Yandex.Cloud
class YandexEmbedStrategy(EmbedStrategy):
    def __init__(self, iam_token: str, model_uri: str):
        if not iam_token or not model_uri:
            raise ValueError("YandexEmbedStrategy requires iam_token and model_uri")
        self.endpoint = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
        self.headers = {"Authorization": f"Bearer {iam_token}"}
        self.model_uri = model_uri

    def embed(self, text: str, **options) -> List[float]:
        body = {"modelUri": self.model_uri, "text": text}
        resp = requests.post(self.endpoint, json=body, headers=self.headers)
        resp.raise_for_status()
        return resp.json()["embedding"]

# Стратегия через OpenRouter
class OpenRouterEmbedStrategy(EmbedStrategy):
    def __init__(self, api_key: str, model: str, endpoint: str):
        if not api_key or not model or not endpoint:
            raise ValueError("OpenRouterEmbedStrategy requires api_key, model and endpoint")
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint

    def embed(self, text: str, **options) -> List[float]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model, "input": text, **options}
        resp = requests.post(self.endpoint, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]
