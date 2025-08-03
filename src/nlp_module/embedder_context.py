from typing import List, Any
from .embed_strategy import EmbedStrategy

class Embedder:
    """
    Контекст для эмбеддинга коротких и длинных текстов через EmbedStrategy.
    """
    def __init__(self, strategy: EmbedStrategy, **default_options):
        self.strategy = strategy
        self.default_options = default_options

    def embed_short(self, text: str, **options) -> List[float]:
        opts = {**self.default_options, **options}
        return self.strategy.embed_short(text, **opts)

    def embed_long(self, text: str, **options) -> List[float]:
        opts = {**self.default_options, **options}
        return self.strategy.embed_long(text, **opts)
