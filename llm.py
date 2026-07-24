from langchain_openrouter import ChatOpenRouter
from config import settings


def get_llm():
    return ChatOpenRouter(
        api_key=settings.openrouter_api_key,
        model=settings.model,
        temperature=0,
    )