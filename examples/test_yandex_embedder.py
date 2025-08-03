from dotenv import load_dotenv
import os
from nlp_module.embedder import create_embedder

load_dotenv()

def manual_test():
    iam_token = os.getenv("YANDEX_OAUTH_TOKEN")
    folder_id = os.getenv("YANDEX_FOLDER_ID")
    if not iam_token or not folder_id:
        raise RuntimeError("Не задан YANDEX_OAUTH_TOKEN или YANDEX_FOLDER_ID")

    print("\n=== Тест через folder_id ===")
    emb1 = create_embedder(iam_token, "yandex", folder_id)
    text = "Привет, мир!"
    print("- короткий текст")
    short1 = emb1.embed_short(text)
    print("  length:", len(short1))
    assert all(isinstance(x, float) for x in short1)
    print("- длинный текст")
    long1 = emb1.embed_long(text * 50)
    print("  length:", len(long1))
    assert all(isinstance(x, float) for x in long1)

    print("\n=== Тест через явные URI ===")
    short_uri = f"emb://{folder_id}/text-search-query/latest"
    long_uri  = f"emb://{folder_id}/text-search-doc/latest"

    print("- только короткий URI")
    emb_short_only = create_embedder(iam_token, "yandex", short_uri)
    s2 = emb_short_only.embed_short(text)
    print("  short length:", len(s2))
    assert all(isinstance(x, float) for x in s2)
    l2 = emb_short_only.embed_long(text)
    print("  long-via-short length:", len(l2))
    assert all(isinstance(x, float) for x in l2)

    print("- только длинный URI")
    emb_long_only = create_embedder(iam_token, "yandex", long_uri)
    s3 = emb_long_only.embed_short(text)
    print("  short-via-long length:", len(s3))
    assert all(isinstance(x, float) for x in s3)
    l3 = emb_long_only.embed_long(text * 10)
    print("  long length:", len(l3))
    assert all(isinstance(x, float) for x in l3)

if __name__ == "__main__":
    manual_test()
