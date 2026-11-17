from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("Weather")


# Define an asynchronous MCP tool
@mcp.tool()
async def get_weather(location: str) -> str:
    """Get the weather for a given location."""
    return "It's always raining in California"


# Start MCP server
if __name__ == "__main__":
    mcp.run(transport="streamable-http")

