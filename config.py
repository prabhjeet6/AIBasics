from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
import os

# find the directory containing this file
BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    openrouter_api_key: str
    openrouter_model:str
    ollama_model:str
    langfuse_public_key:str
    langfuse_base_url:str
    langfuse_secret_key:str
    tavily_api_key:str
    pinecone_api_key:str
    embedding_model_name:str
    index_name:str
    embedding_model_name:str
    text_splitter_chunk_size:int
    text_splitter_chunk_overlap:int
    chunking_strategy:str
    max_characters:int
    file_path:str
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
    )
    langchain_documentation_index_name:str


settings = Settings()

os.environ.setdefault("TAVILY_API_KEY", settings.tavily_api_key)

os.environ["PINECONE_API_KEY"] = settings.pinecone_api_key

os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_base_url)