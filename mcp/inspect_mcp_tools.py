import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    # 1. Configure uvx to spawn duckduckgo-mcp-server
    server_params = StdioServerParameters(
        command="uvx",  # Use full path like "/opt/homebrew/bin/uvx" if uvx isn't in your PATH
        args=["duckduckgo-mcp-server"],
    )

    # 2. Connect via stdio and list all exposed tools
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_response = await session.list_tools()
            print(f"Found {len(tools_response.tools)} tools:\n")

            for tool in tools_response.tools:
                print(f"Tool: {tool.name}")
                print(f"Description: {tool.description}")
                print(f"Schema:\n{json.dumps(tool.input_schema, indent=2)}")
                print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())