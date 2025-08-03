# src/main.py
"""
Главный скрипт:
1) Генерирует ответ от LLM (OpenRouterStrategy) с использованием системного промта из файла
2) Переводит ответ на английский через YandexTranslator
3) Создает эмбеддинги (короткий и длинный) для перевода через YandexEmbedStrategy
"""
import os
from dotenv import load_dotenv
import requests

from nlp_module.llm import OpenRouterClient
from nlp_module.translator import YandexTranslator
from nlp_module.embed_strategy import YandexEmbedStrategy
from nlp_module.embedder_context import Embedder

# Загрузка конфигурации
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
CHAT_MODEL = os.getenv("OPENROUTER_MODEL", "gpt-4o-mini")
CHAT_EP = os.getenv("OPENROUTER_ENDPOINT")
YANDEX_TOKEN = os.getenv("YANDEX_OAUTH_TOKEN")
YANDEX_FOLDER = os.getenv("YANDEX_FOLDER_ID")

if not all([API_KEY, CHAT_EP, YANDEX_TOKEN, YANDEX_FOLDER]):
    raise RuntimeError("Не заданы необходимые переменные окружения")

# 1) Генерация ответа
system_prompts = {"default": None}
llm = OpenRouterClient(
    api_key=API_KEY,
    model=CHAT_MODEL,
    system_prompts={},
    prompts_dir="prompts"
)
# Загружаем системный промт из файла "custom.txt"
sys_text = llm._load_prompt_from_file("custom")
if sys_text:
    llm.system_prompts["custom"] = sys_text

# Формируем запрос к LLM
title = "Расскажи о погоде в Праге."
query = title

# Генерация ответа с обработкой возможного ChunkedEncodingError
try:
    response = llm.generate(query, system_prompt_name="custom")
except requests.exceptions.ChunkedEncodingError as e:
    print(f"Ошибка при получении ответа от LLM (ChunkedEncodingError): {e}")
    response = ""
print("LLM ответ (RU):", response)

# 2) Перевод на английский
translator = YandexTranslator(oauth_token=YANDEX_TOKEN, folder_id=YANDEX_FOLDER)
translated = translator.translate(response, src="ru", dst="en")
print("Перевод (EN):", translated)

# 3) Эмбеддинги Yandex
embed_strategy = YandexEmbedStrategy(iam_token=YANDEX_TOKEN, model=YANDEX_FOLDER)
embedder = Embedder(embed_strategy)
vec_short = embedder.embed_short(translated)
vec_long  = embedder.embed_long(translated)

print(f"Embed short len={len(vec_short)}")
print(f"Embed long len={len(vec_long)}")
