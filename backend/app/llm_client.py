from functools import cached_property

import ollama

from app.config import settings


class LLMClient:
    @cached_property
    def _client(self) -> ollama.Client:
        return ollama.Client(host=settings.ollama_base_url)

    def generate(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat(model=settings.ollama_model, messages=messages)
        return response.message.content.strip()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            self._client.embeddings(model=settings.ollama_embed_model, prompt=text)["embedding"]
            for text in texts
        ]
