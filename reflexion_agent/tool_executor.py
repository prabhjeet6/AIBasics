from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch, tavily_search

from langgraph.prebuilt import ToolNode
from .schemas import AnswerQuestion,ReviseAnswer

tavily_tool=TavilySearch(max_results=5)

def run_queries(search_queries:list[str], **kwargs):
    """Run the generated queries"""
    return tavily_tool.batch([{"query":query} for query in search_queries])

# llm sends tool execution decision to LangGraph's ToolNode

execute_tools=ToolNode(
    [
        # Take my Python run_queries() function, expose it to LangChain as a structured tool, and call that tool AnswerQuestion
        StructuredTool.from_function(run_queries,name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries,name=ReviseAnswer.__name__),
    ]
)