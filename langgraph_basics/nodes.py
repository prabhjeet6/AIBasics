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
    # Here * means, unpack all the messages into this list being passed to llm, so everytime
    # run_agent_reasoning(...) runs, llm sees the entire conversation history
    response= llm.invoke([{"role":"system","content":SYSTEM_MESSAGE},*state["messages"]])
    return {"messages":[response]}

# ToolNode is the LangGraph component that takes the LLM's requested tool call and actually
# runs the corresponding Python tool.
# ToolNode receives AI message from llm which has reasoned to execute a particular tool,
# But, actual tool execution happens through ToolNode
tool_node=ToolNode(tools)