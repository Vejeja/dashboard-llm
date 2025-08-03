from abc import ABC, abstractmethod
from typing import List
from sentence_transformers import SentenceTransformer
import requests


class Embedder(ABC):
    @abstractmethod
    def embed_short(self, text: str) -> List[float]:
        """Для коротких текстов (запросов)."""
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


class YandexEmbedder(Embedder):
    """
    Эмбеддер через Yandex Cloud Foundation Models Embeddings API.

    При инициализации можно передать:
      - folder_id: будут использованы стандартные модели:
        - короткие: emb://<folder_id>/text-search-query/latest
        - длинные: emb://<folder_id>/text-search-doc/latest
      - или полный URI модели ("emb://..." или "gpt://...").

    Требует IAM-токен и folder_id или модельного URI.
    """
    def __init__(self, iam_token: str, model: str):
        if not iam_token or not model:
            raise ValueError("Yandex Embedder requires iam_token and folder_id or model URI")
        self.endpoint = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
        self.headers = {"Authorization": f"Bearer {iam_token}"}
        if model.startswith("emb://") or model.startswith("gpt://"):
            self.short_model_uri = model
            self.long_model_uri = model
        else:
            folder_id = model
            self.short_model_uri = f"emb://{folder_id}/text-search-query/latest"
            self.long_model_uri = f"emb://{folder_id}/text-search-doc/latest"

    def _embed(self, model_uri: str, text: str) -> List[float]:
        body = {"modelUri": model_uri, "text": text}
        resp = requests.post(self.endpoint, json=body, headers=self.headers)
        resp.raise_for_status()
        data = resp.json()
        return [float(x) for x in data.get("embedding", [])]

    def embed_short(self, text: str) -> List[float]:
        """Векторизация короткого текста (запросов)."""
        return self._embed(self.short_model_uri, text)

    def embed_long(self, text: str) -> List[float]:
        """Векторизация длинного текста (документов)."""
        return self._embed(self.long_model_uri, text)


def create_embedder(api_key: str, provider: str, model: str) -> Embedder:
    """
    Фабрика эмбеддеров.

    :param api_key: токен для выбранного сервиса:
        - для HF: не используется;
        - для Yandex: IAM-токен.
    :param provider: 'hf' или 'yandex'
    :param model: для HF — полный repo_id;
                  для Yandex — folder_id или полный URI модели.
    """
    if provider == "hf":
        return LocalHFEmbedder(api_key, model)
    if provider == "yandex":
        return YandexEmbedder(api_key, model)
    raise ValueError(f"Unknown embedder provider: {provider}")
