from functools import cached_property

from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

from app.config import settings


class LLMClient:
    @cached_property
    def _embed_model(self) -> SentenceTransformer:
        return SentenceTransformer(settings.hf_embed_model)

    def generate(self, prompt: str, system: str = "") -> str:
        client = InferenceClient(api_key=settings.hf_api_key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat_completion(
            model=settings.hf_model,
            messages=messages,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._embed_model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()
