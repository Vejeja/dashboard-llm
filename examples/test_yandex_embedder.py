from dotenv import load_dotenv
import os

from nlp_module.embedder import create_embedder

load_dotenv()  

def manual_test():
    iam_token = os.getenv("YANDEX_OAUTH_TOKEN")
    folder_id = os.getenv("YANDEX_FOLDER_ID")
    if not iam_token or not folder_id:
        raise RuntimeError("Не задан YANDEX_OAUTH_TOKEN или YANDEX_FOLDER_ID")

    model_uri = f"emb://{folder_id}/text-search-query/latest"


    embedder = create_embedder(iam_token, "yandex", model_uri)

    text = "Привет, мир!"
    emb_short = embedder.embed_short(text)
    print("Short embedding length:", len(emb_short))
    assert isinstance(emb_short, list) and len(emb_short) > 0
    assert all(isinstance(x, float) for x in emb_short)

    emb_long = embedder.embed_long(text * 50)
    print("Long embedding length:", len(emb_long))
    assert isinstance(emb_long, list) and len(emb_long) > 0
    assert all(isinstance(x, float) for x in emb_long)

    print("Первый 5 элементов embedding:", emb_short[:5])

if __name__ == "__main__":
    manual_test()
