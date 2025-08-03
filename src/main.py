import os
from dotenv import load_dotenv
import requests

# 1) LLM через стратегию и контекст
from nlp_module.llm_strategy import OpenRouterStrategy
from nlp_module.llm_context import LLMClient

# 2) Перевод
from nlp_module.translator import YandexTranslator

# 3) Эмбеддинги
from nlp_module.embed_strategy import YandexEmbedStrategy
from nlp_module.embedder_context import Embedder

load_dotenv()
API_KEY       = os.getenv("OPENROUTER_API_KEY")
CHAT_MODEL    = os.getenv("OPENROUTER_MODEL", "gpt-4o-mini")
CHAT_EP       = os.getenv("OPENROUTER_ENDPOINT")
YANDEX_TOKEN  = os.getenv("YANDEX_OAUTH_TOKEN")
YANDEX_FOLDER = os.getenv("YANDEX_FOLDER_ID")

if not all([API_KEY, CHAT_EP, YANDEX_TOKEN, YANDEX_FOLDER]):
    raise RuntimeError("Не заданы необходимые переменные окружения")

# ---- 1) Сборка LLM клиента через стратегию ----
# Загружаем системные промты из каталога prompts
system_prompts = {"default": None}
prompts_dir = "prompts"

strategy = OpenRouterStrategy(
    api_key=API_KEY,
    model=CHAT_MODEL,
    endpoint=CHAT_EP,
    system_prompts=system_prompts
)
llm = LLMClient(strategy)

# Подгружаем кастомный промт из файла custom.txt и сохраняем в стратегию
custom = llm.strategy.system_prompts.get("custom")
if custom is None:
    # В OpenRouterStrategy у нас нет _load_prompt_from_file,
    # поэтому прочитаем файл вручную:
    path = os.path.join(prompts_dir, "custom.txt")
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        llm.strategy.system_prompts["custom"] = text
    except FileNotFoundError:
        pass

query = "Расскажи о погоде в Праге."
try:
    response = llm.generate(query, system_prompt_name="custom")
except requests.exceptions.ChunkedEncodingError as e:
    print(f"Ошибка от LLM: {e}")
    response = ""
print("LLM ответ (RU):", response)

# ---- 2) Перевод через YandexTranslator ----
translator = YandexTranslator(oauth_token=YANDEX_TOKEN, folder_id=YANDEX_FOLDER)
translated = translator.translate(response, src="ru", dst="en")
print("Перевод (EN):", translated)

# ---- 3) Эмбеддинги через YandexEmbedStrategy и Embedder ----
embed_strategy = YandexEmbedStrategy(iam_token=YANDEX_TOKEN, model=YANDEX_FOLDER)
embedder = Embedder(embed_strategy)
vec_short = embedder.embed_short(translated)
vec_long  = embedder.embed_long(translated)
print(f"Embed short len = {len(vec_short)}")
print(f"Embed long  len = {len(vec_long)}")
