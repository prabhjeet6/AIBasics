from langchain_openrouter import ChatOpenRouter
from config import settings
from langchain_ollama import ChatOllama


def get_llm():
    return ChatOpenRouter(
        api_key=settings.openrouter_api_key,
        model=settings.openrouter_model,
        temperature=0,
    )

# ollama is a tool that allows to run open weight models
# langchain-ollama allows to integrate such models with lanchain framework
# langchain is a framework to conviently use any model without changing design phylosiphy of your app
def get_ollama_llm():
    return ChatOllama(
        
        model=settings.ollama_model,
        temperature=0,
    )