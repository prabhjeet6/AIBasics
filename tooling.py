from tavily import TavilyClient
from langchain.tools import tool
from langchain_tavily import TavilySearch
import os
from config import settings

os.environ.setdefault("TAVILY_API_KEY", settings.tavily_api_key)

tavily_client=TavilyClient()


# an agent needs a model and list of tools to execute
# a tool can be user definded function with @tool annotation, and description, that helps the model to understand 
# the purpose of that tool/function. A tool can also be third party, for example here we use Tavily as a tool for web 
# search or the custom function below 

@tool
def search(query:str)->str:
    """
    Tool that searches over internet
    Args:
     query: The query to search for
    Returns: 
     The search result
    """
    print(f"Searching for {query}")
    return tavily_client.search(query=query)

