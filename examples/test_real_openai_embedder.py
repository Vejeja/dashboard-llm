#!/usr/bin/env python3

from dotenv import load_dotenv
import os

from nlp_module.embedder import create_embedder

def main():
    load_dotenv()  # подгрузит EMBEDDER_PROVIDER, EMBEDDER_MODEL, OPENAI_API_KEY

    provider = os.getenv("EMBEDDER_PROVIDER", "openai")
    model    = os.getenv("EMBEDDER_MODEL", "text-embedding-3-small")
    api_key  = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in your .env before running this script")

    emb = create_embedder(api_key=api_key, provider=provider, model=model)

    sample = "This is a test sentence for real OpenAI embeddings."
    vec = emb.embed_short(sample)

    print(f"Model: {model}")
    print(f"Embedding length: {len(vec)}")
    print("First 10 dimensions:", vec[:10])

if __name__ == "__main__":
    main()
