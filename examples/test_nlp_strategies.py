import os
from dotenv import load_dotenv
import requests

load_dotenv()

from nlp_module.embed_strategy import YandexEmbedStrategy, OpenRouterEmbedStrategy
from nlp_module.embedder_context import Embedder
from nlp_module.llm_strategy import OpenRouterStrategy
from nlp_module.llm_context import LLMClient


def test_llm_with_all_prompts():
    prompts = {"default": "You are a helpful assistant.", "ru": "Отвечай на русском"}
    key = os.getenv("OPENROUTER_API_KEY")
    chat_ep = os.getenv("OPENROUTER_CHAT_ENDPOINT")
    chat_model = os.getenv("OPENROUTER_MODEL", "gpt-4o-mini")
    if not key or not chat_ep:
        print("OPENROUTER_API_KEY или OPENROUTER_CHAT_ENDPOINT не заданы")
        return

    strat = OpenRouterStrategy(key, chat_model, chat_ep, prompts)
    client = LLMClient(strat)
    print("=== LLM через OpenRouterStrategy ===")
    for name in prompts:
        try:
            out = client.generate("Hello!", system_prompt_name=name)
            print(f"[{name}] {out}")
        except Exception as e:
            print(f"[{name}] Ошибка: {e}")


def test_yandex_embedder():
    iam = os.getenv("YANDEX_OAUTH_TOKEN")
    fid = os.getenv("YANDEX_FOLDER_ID")
    if not iam or not fid:
        print("YANDEX_OAUTH_TOKEN или YANDEX_FOLDER_ID не заданы")
        return
    print("=== YandexEmbedStrategy ===")
    strat = YandexEmbedStrategy(iam, fid)  # передаём folder_id
    emb = Embedder(strat)
    try:
        s = emb.embed_short("Привет")
        l = emb.embed_long("Привет " * 50)
        print(f"Yandex short len={len(s)}, long len={len(l)}")
    except requests.HTTPError as e:
        print(f"Yandex error: {e}")


def test_openrouter_embedder():
    key = os.getenv("OPENROUTER_API_KEY")
    embed_ep = os.getenv("OPENROUTER_EMBED_ENDPOINT")
    embed_model = os.getenv("OPENROUTER_MODEL", "embedding-model")
    if not key or not embed_ep:
        print("OPENROUTER_API_KEY или OPENROUTER_EMBED_ENDPOINT не заданы")
        return
    print("=== OpenRouterEmbedStrategy ===")
    strat = OpenRouterEmbedStrategy(key, embed_model, embed_ep)
    emb = Embedder(strat)
    try:
        qs = emb.embed_short("Test embed short via OpenRouter")
        ql = emb.embed_long("Doc text " * 30)
        print(f"OpenRouter short len={len(qs)}, long len={len(ql)}")
    except requests.HTTPError as e:
        print(f"OpenRouter error: {e}")

if __name__ == "__main__":
    test_llm_with_all_prompts()
    test_yandex_embedder()
    test_openrouter_embedder()