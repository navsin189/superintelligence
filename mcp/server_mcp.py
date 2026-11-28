from mcp.server import MCPServer
import requests

mcp = MCPServer(
    "Demo",
    version="1.0.0",
    description="A demo MCP server with greeting and time tools.",
    instructions="Use the greeting tool to greet someone, and the time tool to get current date/time for a timezone.",
)

@mcp.tool()
def greeting(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"

@mcp.tool()
async def time(timezone="Asia/Kolkata"):
    """It takes timezone as a input and get the date and time from time.now api. Then returns the Datetime"""
    response = requests.get(f"https://time.now/developer/api/time?timezone={timezone}")
    return response.json()["datetime"]