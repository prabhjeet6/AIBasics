from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openrouter_api_key: str
    openrouter_model:str
    ollama_model:str
    langfuse_public_key:str
    langfuse_base_url:str
    langfuse_secret_key:str
    tavily_api_key:str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()