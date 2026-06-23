from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str
    tavily_api_key: str
    database_url: str
    redis_url: str
    chromadb_path: str = "./chromadb_data"
    frontend_url: str = "http://localhost:5173"

    class Config:
        env_file = ".env"

settings = Settings()