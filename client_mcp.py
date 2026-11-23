import anyio

from mcp import Client


async def main() -> None:
    async with Client("http://localhost:8000/mcp") as client:
        print("#"*5," MCP Server Info ","#"*5)
        print(client.server_info)
        print(client.server_capabilities)
        print(client.protocol_version)
        print(client.instructions)
        result = await client.list_tools()
        print("#"*5," MCP Server Tools Info ","#"*5)
        for tool in result.tools:
            print(tool.name)
            print(tool.title)
            print(tool.description)
            print(tool.input_schema)


if __name__ == "__main__":
    anyio.run(main)