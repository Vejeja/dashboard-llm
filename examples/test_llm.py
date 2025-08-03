#!/usr/bin/env python3
from dotenv import load_dotenv
import os
import sys

from nlp_module.llm import LLMClient

def main():
    load_dotenv()

    api_key        = os.getenv("OPENROUTER_API_KEY")
    model          = os.getenv("OPENROUTER_MODEL")
    prompts_dir    = os.getenv("SYSTEM_PROMPTS_DIR", "prompts")
    default_prompt = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")

    if not api_key or not model:
        print("Error: Set OPENROUTER_API_KEY and OPENROUTER_MODEL in your .env", file=sys.stderr)
        sys.exit(1)

    # Передаём в клиент:
    client = LLMClient(
        api_key=api_key,
        model=model,
        provider="openrouter",
        system_prompts={"default": default_prompt},
        prompts_dir=prompts_dir,
    )

    # Попробуем два варианта: 
    # 1) из переменной среды ("default") 
    # 2) из файла prompts/custom.txt
    for name in ("default", "custom"):
        print(f"\n=== prompt name: {name!r} ===")
        resp = client.generate("Hello, world!", system_prompt_name=name)
        print(resp)

if __name__ == "__main__":
    main()
