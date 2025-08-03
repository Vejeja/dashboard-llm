#!/usr/bin/env python
from dotenv import load_dotenv
import os

from nlp_module.embedder import create_embedder

def main():
    load_dotenv()
    provider = os.getenv("EMBEDDER_PROVIDER")
    model    = os.getenv("EMBEDDER_MODEL")
    api_key  = os.getenv("HF_API_TOKEN") if provider=="hf" else os.getenv("OPENAI_API_KEY")

    emb = create_embedder(api_key=api_key, provider=provider, model=model)
    sample = "This is a test sentence."
    vec_short = emb.embed_short(sample)
    print("embed_short:", vec_short)
    # vec_long = emb.embed_long(...)  # если нужно

if __name__ == "__main__":
    main()
