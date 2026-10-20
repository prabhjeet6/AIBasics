from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a:int,b:int)->int:
    """ Add two numbers """
    return a+b

@mcp.tool()
def multiply(a:int,b:int)->int:
    """ Multiply two numbers """
    return a*b

# starts  MCP server and tells it to communicate using stdio (standard input and standard output).
if __name__ == "__main__":
    mcp.run(transport="stdio")