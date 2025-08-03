from abc import ABC, abstractmethod
from typing import List
import openai
from sentence_transformers import SentenceTransformer
import requests


class Embedder(ABC):
    @abstractmethod
    def embed_short(self, text: str) -> List[float]:
        """Для коротких текстов."""
        ...

    @abstractmethod
    def embed_long(self, text: str) -> List[float]:
        """Для больших документов."""
        ...


class LocalHFEmbedder(Embedder):
    """
    Локальный эмбеддер на основе Hugging Face SentenceTransformers.
    """
    def __init__(self, api_key: str, model: str):
        # api_key не используется при локальном запуске
        self.model = SentenceTransformer(model)

    def embed_short(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

    def embed_long(self, text: str) -> List[float]:
        return self.embed_short(text)


class OpenAIEmbedder(Embedder):
    """
    Эмбеддер на основе OpenAI Embeddings API.
    """
    def __init__(self, api_key: str, model: str):
        openai.api_key = api_key
        self.model = model

    def embed_short(self, text: str) -> List[float]:
        resp = openai.Embedding.create(input=text, model=self.model)
        return resp["data"][0]["embedding"]

    def embed_long(self, text: str) -> List[float]:
        return self.embed_short(text)


class YandexEmbedder(Embedder):
    """
    Эмбеддер через Yandex Cloud Foundation Models Embeddings API.

    Требует IAM-токен и URI модели.
    """
    def __init__(self, iam_token: str, model_uri: str):
        if not iam_token or not model_uri:
            raise ValueError("Yandex Embedder requires iam_token and model_uri")
        self.endpoint = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
        self.headers = {"Authorization": f"Bearer {iam_token}"}
        self.model_uri = model_uri

    def _embed(self, text: str) -> List[float]:
        body = {"modelUri": self.model_uri, "text": text}
        resp = requests.post(self.endpoint, json=body, headers=self.headers)
        resp.raise_for_status()
        data = resp.json()
        return [float(x) for x in data.get("embedding", [])]

    def embed_short(self, text: str) -> List[float]:
        return self._embed(text)

    def embed_long(self, text: str) -> List[float]:
        return self._embed(text)


def create_embedder(api_key: str, provider: str, model: str) -> Embedder:
    """
    Фабрика эмбеддеров.

    :param api_key: токен для выбранного сервиса (для Yandex — IAM-токен)
    :param provider: 'hf', 'openai' или 'yandex'
    :param model: для HF — полный repo_id; для OpenAI — имя модели; для Yandex — URI модели
    """
    if provider == "hf":
        return LocalHFEmbedder(api_key, model)
    if provider == "openai":
        return OpenAIEmbedder(api_key, model)
    if provider == "yandex":
        return YandexEmbedder(api_key, model)
    raise ValueError(f"Unknown embedder provider: {provider}")