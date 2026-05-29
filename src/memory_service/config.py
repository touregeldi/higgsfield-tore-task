from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://memory:memory@db:5432/memory"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    max_body_bytes: int = 1_048_576
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openai_api_key.strip())
