
from langchain_openrouter import ChatOpenRouter
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

from tooling import search
from config import settings
from models import AgentResponse

# ReAct
# What it is: A reasoning strategy for an LLM.
# Goal: Decide what to do next.
#Pattern: Reason → Tool Call → Observe → Reason → ... → Final Answer
#Focus: AI decision-making.
#Typical use: Tool-using agents.

#LangGraph
#What it is: A workflow/orchestration framework.
#Goal: Decide which step/node executes next.
#Pattern: Graph of nodes and edges (LLMs, tools, Python functions, humans, APIs, databases).
#Focus: Application execution and state management.
#Supports: Loops, branching, retries, parallel execution, human approval, persistence.

# original reAct Model involved reasoning and thought, it used to provide Action, mordern tool/function calling goes
# beyond thought and applies action through particular tools to get an observation

def get_llm():
    return ChatOpenRouter(
        api_key=settings.openrouter_api_key,
        model=settings.openrouter_model,
        temperature=0,
    )

# ollama is a tool that allows to run open weight models
# langchain-ollama allows to integrate such models with langchain framework
# langchain is a framework to conveniently use any model without changing design philosophy of your app
def get_ollama_llm():
    return ChatOllama(
        
        model=settings.ollama_model,
        temperature=0,
    )



def get_agent():

    llm = get_llm()
    
    #llm =get_ollama_llm() # gemma3:270b does not support Tools

    #tools=[TavilySearch()]

    #agent=create_agent(model=llm,tools=[search])

    # bind_tools(...) simply informs the model of tools existence. User is responsible of executing the tool,
    # sending the results back and calling the llm again.

    # create_agent(...) on the other hand, binds the tools, invokes llm and orchestrates the reAct loop out of the box.

    # directly passing TavilySearch tool from langchain_tavili instead of passing a custom tool that used TavilyClient() underneath
    agent=create_agent(model=llm,tools=[TavilySearch], response_format=AgentResponse)  
    return agent


