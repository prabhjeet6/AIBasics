import asyncio

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.messages import  HumanMessage

# main.py acts as the MCP host application, and it uses an MCP client to communicate with each MCP server.
# Each MCP client connection communicates with one MCP server

# asyncio — runs asynchronous Python code.
# ClientSession — manages the MCP conversation between client and server.
# StdioServerParameters — specifies how to launch the server process.
# stdio_client — starts the server subprocess and establishes communication over stdin/stdout.

from agentutils import get_llm

llm=get_llm()

# configure the server
stdio_server_params=StdioServerParameters(
    command="python",
    args=["servers/math_server.py"],
)

# demonstrates only connecting with stdio here
async def main():
    async with stdio_client(stdio_server_params) as (read, write):
        async with ClientSession(read_stream=read, write_stream=write) as session:
            await session.initialize()
            print("session initialized")
            # converts mcp tools into langchain tools
            tools = await load_mcp_tools(session)

            agent = create_react_agent(llm, tools)

            result = await agent.ainvoke(
                {"messages": [HumanMessage(content="What is 54 + 2 * 3?")]}
            )
            print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())