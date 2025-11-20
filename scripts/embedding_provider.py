import os
from typing import List
from dotenv import load_dotenv
import openai
load_dotenv()

USE_OPENAI = os.getenv("USE_OPENAI", "1") == "1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if USE_OPENAI:
    openai.api_key = OPENAI_API_KEY

class EmbeddingProvider:
    def __init__(self, model: str = "text-embedding-3-large"):
        self.model = model

    def embed_text(self, text: str) -> List[float]:
        """
        Return a list[float] embedding. This function is synchronous for simplicity.
        Replace with async if needed.
        """
        if USE_OPENAI:
            resp = openai.Embedding.create(model=self.model, input=text)
            emb = resp["data"][0]["embedding"]
            return emb
        else:
            return [0.0] * 1536
