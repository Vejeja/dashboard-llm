#!/usr/bin/env python3

from dotenv import load_dotenv
import os

from nlp_module.translator import create_translator

def main():
    load_dotenv()

    token     = os.getenv("YANDEX_OAUTH_TOKEN")
    folder_id = os.getenv("YANDEX_FOLDER_ID")
    if not token or not folder_id:
        raise RuntimeError("Установите YANDEX_OAUTH_TOKEN и YANDEX_FOLDER_ID в .env")

    translator = create_translator(oauth_token=token, folder_id=folder_id)
    text = "Hello, world!"
    print("Yandex.Cloud → ru:", translator.translate(text, src="en", dst="ru"))

if __name__ == "__main__":
    main()
