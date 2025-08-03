from abc import ABC, abstractmethod
import requests


class Translator(ABC):
    @abstractmethod
    def translate(self, text: str, src: str, dst: str) -> str:
        """Переводит текст между двумя языками."""
        ...


class YandexTranslator(Translator):
    """
    Перевод через Yandex.Cloud Translate API v2.

    Требует:
      - OAuth-токен (получается через `yc iam create-token` или SDK).
      - folderId вашего облака.
    """
    def __init__(self, oauth_token: str, folder_id: str):
        if not oauth_token or not folder_id:
            raise ValueError("YANDEX_OAUTH_TOKEN и YANDEX_FOLDER_ID должны быть заданы")
        self.url = "https://translate.api.cloud.yandex.net/translate/v2/translate"
        self.headers = {"Authorization": f"Bearer {oauth_token}"}
        self.folder_id = folder_id

    def translate(self, text: str, src: str, dst: str) -> str:
        body = {
            "folderId": self.folder_id,
            "texts": [text],
            "sourceLanguageCode": src,
            "targetLanguageCode": dst,
        }
        resp = requests.post(self.url, json=body, headers=self.headers)
        resp.raise_for_status()
        data = resp.json()
        # {"translations":[{"text":"Привет, мир!"}]}
        return data["translations"][0]["text"]


def create_translator(oauth_token: str, folder_id: str) -> Translator:
    """
    Фабрика переводчика (только Yandex.Cloud).
    :param oauth_token: результат `yc iam create-token`
    :param folder_id: ваш folderId из Yandex.Cloud
    """
    return YandexTranslator(oauth_token, folder_id)
