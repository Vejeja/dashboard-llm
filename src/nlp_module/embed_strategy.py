from abc import ABC, abstractmethod
from typing import List, Any
import requests

class EmbedStrategy(ABC):
    @abstractmethod
    def embed_short(self, text: str, **options: Any) -> List[float]:
        """
        Эмбеддинг коротких текстов (query).
        """
        ...

    @abstractmethod
    def embed_long(self, text: str, **options: Any) -> List[float]:
        """
        Эмбеддинг длинных документов (doc).
        """
        ...

class YandexEmbedStrategy(EmbedStrategy):
    """
    Эмбеддинг через Yandex.Cloud:
      - короткий:  emb://<folder_id>/text-search-query/latest
      - длинный:   emb://<folder_id>/text-search-doc/latest
    Если в конструктор передан полный URI (emb://… или gpt://…), он
    будет использован и для query, и для doc.
    """
    def __init__(self, iam_token: str, model: str):
        if not iam_token or not model:
            raise ValueError("YandexEmbedStrategy requires iam_token and model (folder_id or full URI)")
        self.endpoint = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
        self.headers = {"Authorization": f"Bearer {iam_token}"}
        if model.startswith("emb://") or model.startswith("gpt://"):
            self.short_model_uri = model
            self.long_model_uri  = model
        else:
            folder_id = model
            self.short_model_uri = f"emb://{folder_id}/text-search-query/latest"
            self.long_model_uri  = f"emb://{folder_id}/text-search-doc/latest"

    def _embed(self, model_uri: str, text: str) -> List[float]:
        body = {"modelUri": model_uri, "text": text}
        resp = requests.post(self.endpoint, json=body, headers=self.headers)
        resp.raise_for_status()
        return [float(x) for x in resp.json().get("embedding", [])]

    def embed_short(self, text: str, **options: Any) -> List[float]:
        return self._embed(self.short_model_uri, text)

    def embed_long(self, text: str, **options: Any) -> List[float]:
        return self._embed(self.long_model_uri, text)

class OpenRouterEmbedStrategy(EmbedStrategy):
    """
    Эмбеддинг через OpenRouter API для любого текста (query и doc).
    """
    def __init__(self, api_key: str, model: str, endpoint: str):
        if not api_key or not model or not endpoint:
            raise ValueError("OpenRouterEmbedStrategy requires api_key, model and endpoint")
        self.api_key = api_key
        self.model   = model
        self.endpoint= endpoint

    def _embed(self, text: str, **options: Any) -> List[float]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model, "input": text, **options}
        resp = requests.post(self.endpoint, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    def embed_short(self, text: str, **options: Any) -> List[float]:
        return self._embed(text, **options)

    def embed_long(self, text: str, **options: Any) -> List[float]:
        return self._embed(text, **options)
