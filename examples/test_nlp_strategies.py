import os
from dotenv import load_dotenv

load_dotenv()

from nlp_module.embed_strategy import YandexEmbedStrategy
from nlp_module.embedder_context import Embedder
from nlp_module.llm_strategy import OpenRouterStrategy
from nlp_module.llm_context import LLMClient
from nlp_module.embedder import create_embedder


def test_llm_with_all_prompts():
    system_prompts = {
        "default": "You are a helpful assistant.",
        "ru": "Отвечай на русском",
        # другие prompts, если нужно
    }

    api_key = os.getenv("OPENROUTER_API_KEY")
    endpoint = os.getenv("OPENROUTER_ENDPOINT")
    model = os.getenv("OPENROUTER_MODEL")

    if not api_key or not endpoint:
        print("Переменные OPENROUTER_API_KEY или OPENROUTER_ENDPOINT не заданы.")
        return

    strategy = OpenRouterStrategy(
        api_key=api_key,
        model=model,
        endpoint=endpoint,
        system_prompts=system_prompts
    )
    client = LLMClient(strategy)

    print("=== Тест LLM с разными system prompts ===")
    for name in system_prompts:
        print(f"\n[Prompt: {name}]")
        output = client.generate("Hello, how are you?", system_prompt_name=name)
        print(output)


def test_yandex_embedder():
    iam_token = os.getenv("YANDEX_OAUTH_TOKEN")
    folder_id = os.getenv("YANDEX_FOLDER_ID")

    if not iam_token or not folder_id:
        print("Переменные YANDEX_OAUTH_TOKEN или YANDEX_FOLDER_ID не заданы. Пропускаем тест эмбеддинга.")
        return

    # Динамически сформированные URI моделей
    short_uri = f"emb://{folder_id}/text-search-query/latest"
    long_uri  = f"emb://{folder_id}/text-search-doc/latest"

    print("\n=== Yandex Embed через create_embedder ===")
    print("- только короткий URI")
    emb_short_only = create_embedder(iam_token, "yandex", short_uri)
    vec_short = emb_short_only.embed_short("Привет, мир!")
    print(f"Короткий URI: размер вектора = {len(vec_short)}")

    print("\n=== Yandex Embed через стратегию напрямую ===")
    strategy = YandexEmbedStrategy(iam_token=iam_token, model_uri=short_uri)
    embedder = Embedder(strategy)
    vec_long = embedder.embed_long("Привет, мир! " * 100)
    print(f"Длинный текст через стратегию: размер вектора = {len(vec_long)}")

if __name__ == "__main__":
    # test_llm_with_all_prompts()
    test_yandex_embedder()
