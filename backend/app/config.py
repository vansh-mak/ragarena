from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    hf_api_key: str
    hf_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    hf_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    tavily_api_key: str
    database_url: str
    redis_url: str
    chromadb_path: str = "./chromadb_data"
    frontend_url: str = "http://localhost:5173"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
