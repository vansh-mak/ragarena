from functools import cached_property

import ollama

from app.config import settings


class LLMClient:
    @cached_property
    def _ollama(self) -> ollama.Client:
        return ollama.Client(host=settings.ollama_base_url)

    def generate(self, prompt: str, system: str = "") -> str:
        if settings.use_groq:
            from groq import Groq
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            response = Groq(api_key=settings.groq_api_key).chat.completions.create(
                model=settings.groq_model,
                messages=messages,
                max_tokens=1024,
            )
            return response.choices[0].message.content

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self._ollama.chat(model=settings.ollama_model, messages=messages)
        return response.message.content.strip()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if settings.use_groq:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(settings.groq_embed_model)
            return model.encode(texts, normalize_embeddings=True).tolist()

        return [
            self._ollama.embeddings(model=settings.ollama_embed_model, prompt=text)["embedding"]
            for text in texts
        ]
