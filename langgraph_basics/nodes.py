from langchain_classic.evaluation.scoring.prompt import SYSTEM_MESSAGE
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from agentutils import get_llm

SYSTEM_MESSAGE=""" You are a helpful assistant that can use tools to answer questions"""

@tool
def triple(num:float)->float:
    """
    param num:a number to triple
    returns: the triple of input number

    """
    return float(num)*3

tools=[TavilySearch(max_results=1),triple]
llm = get_llm().bind_tools(tools)

def run_agent_reasoning(state:MessagesState)->MessagesState:
    """
    Run the agent reasoning node

    """
    response= get_llm().invoke([{"role":"system","content":SYSTEM_MESSAGE},*state["messages"]])
    return {"messages":[response]}

tool_node=ToolNode(tools)