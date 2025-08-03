import os
from dotenv import load_dotenv
import requests

load_dotenv()

from nlp_module.embed_strategy import YandexEmbedStrategy, OpenRouterEmbedStrategy
from nlp_module.embedder_context import Embedder
from nlp_module.llm_strategy import OpenRouterStrategy
from nlp_module.llm_context import LLMClient


def test_llm_with_all_prompts():
    system_prompts = {
        "default": "You are a helpful assistant.",
        "ru": "Отвечай на русском",
    }
    api_key = os.getenv("OPENROUTER_API_KEY")
    endpoint = os.getenv("OPENROUTER_ENDPOINT")
    model = os.getenv("OPENROUTER_MODEL", "gpt-4o-mini")
    if not api_key or not endpoint:
        print("OPENROUTER_API_KEY или OPENROUTER_ENDPOINT не заданы.")
        return

    strat = OpenRouterStrategy(api_key, model, endpoint, system_prompts)
    client = LLMClient(strat)

    print("=== LLM через OpenRouterStrategy ===")
    for name in system_prompts:
        try:
            resp = client.generate("Hello!", system_prompt_name=name)
            print(f"[{name}] {resp}\n")
        except Exception as e:
            print(f"Ошибка в LLM генерации для промпта '{name}': {e}")


def test_yandex_embedder():
    iam = os.getenv("YANDEX_OAUTH_TOKEN")
    fid = os.getenv("YANDEX_FOLDER_ID")
    if not iam or not fid:
        print("YANDEX_OAUTH_TOKEN или YANDEX_FOLDER_ID не заданы.")
        return

    short_uri = f"emb://{fid}/text-search-query/latest"
    long_uri  = f"emb://{fid}/text-search-doc/latest"

    print("=== YandexEmbedStrategy ===")
    strat_y = YandexEmbedStrategy(iam, short_uri)
    emb_y = Embedder(strat_y)
    try:
        v1 = emb_y.embed_short("Привет")
        v2 = emb_y.embed_long("Привет " * 50)
        print(f"Yandex short len={len(v1)}, long len={len(v2)}")
    except requests.HTTPError as e:
        print(f"Ошибка YandexEmbedStrategy: {e}")


def test_openrouter_embedder():
    key = os.getenv("OPENROUTER_API_KEY")
    endpoint = os.getenv("OPENROUTER_ENDPOINT")
    model = os.getenv("OPENROUTER_MODEL", "embed-model")
    if not key or not endpoint:
        print("OPENROUTER_API_KEY или OPENROUTER_ENDPOINT не заданы.")
        return

    print("=== OpenRouterEmbedStrategy ===")
    strat_or = OpenRouterEmbedStrategy(key, model, endpoint)
    emb_or = Embedder(strat_or)
    try:
        vs = emb_or.embed_short("Test embedding via OpenRouter")
        print(f"OpenRouter short len={len(vs)}")
    except requests.HTTPError as e:
        print(f"Ошибка OpenRouterEmbedStrategy: {e}")

if __name__ == "__main__":
    test_llm_with_all_prompts()
    test_yandex_embedder()
    test_openrouter_embedder()