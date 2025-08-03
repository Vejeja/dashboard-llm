#!/usr/bin/env python3

from dotenv import load_dotenv
import os

from nlp_module.embedder import create_embedder

def main():
    load_dotenv()  # подгрузит EMBEDDER_PROVIDER и EMBEDDER_MODEL из .env

    provider = os.getenv("EMBEDDER_PROVIDER", "hf")
    model    = os.getenv("EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    api_key  = os.getenv("HF_API_TOKEN", "")  # не используется локально

    emb = create_embedder(api_key=api_key, provider=provider, model=model)

    sample = "This is a test sentence for local HF embeddings."
    vec = emb.embed_short(sample)

    print(f"Model: {model}")
    print(f"Embedding length: {len(vec)}")
    print("First 10 dimensions:", vec[:10])

if __name__ == "__main__":
    main()
