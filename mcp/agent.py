"""
Simple MCP + Ollama Agent
=========================
Connects to our local MCP server (server_mcp.py) over stdio,
exposes its tools to a local Ollama LLM, and runs an interactive
prompt loop with tool-calling support.

Flow:
  User prompt → Ollama LLM → (optional) tool call → MCP server → tool result → LLM → final answer
"""

import asyncio
import json
import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "granite4.2:3b"


# ---------------------------------------------------------------------------
# 2. CONVERT MCP TOOLS → OLLAMA FORMAT
# ---------------------------------------------------------------------------
# MCP tools have their own schema. Ollama expects OpenAI-style tool defs.
# This function bridges the two.

def mcp_tools_to_ollama(mcp_tools) -> list[dict]:
    """Convert MCP tool list into Ollama's tool-calling format."""
    ollama_tools = []
    for tool in mcp_tools:
        schema = dict(
            getattr(tool, "inputSchema", None)
            or getattr(tool, "input_schema", {})
        )
        ollama_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": schema,
            },
        })
    return ollama_tools


# ---------------------------------------------------------------------------
# 3. CALL OLLAMA
# ---------------------------------------------------------------------------

async def call_ollama(messages: list[dict], tools: list[dict]) -> dict:
    """Send messages + tools to Ollama and return the assistant message."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": tools,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
        return resp.json()["message"]


# ---------------------------------------------------------------------------
# 4. AGENT LOOP — the core orchestration
# ---------------------------------------------------------------------------
# One user prompt can trigger multiple rounds:
#   LLM decides to call a tool → we execute it via MCP → feed result back → repeat
# The loop ends when the LLM responds with plain text (no tool calls).

async def agent_loop(session: ClientSession, tools: list[dict], user_prompt: str):
    """Run one prompt through the LLM, handling any tool calls via MCP."""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to tools. "
                "Use the tools when the user's question requires them. "
                "Otherwise, answer directly from your knowledge."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    # Allow up to 5 tool-call rounds per prompt (safety cap)
    for step in range(5):
        # Ask the LLM
        assistant_msg = await call_ollama(messages, tools)
        messages.append(assistant_msg)

        # Check if the LLM wants to call any tools
        tool_calls = assistant_msg.get("tool_calls")
        if not tool_calls:
            # No tool calls — this is the final text answer
            print(f"\nAssistant: {assistant_msg.get('content', '')}")
            return

        # Execute each tool call through MCP
        for call in tool_calls:
            fn_name = call["function"]["name"]
            fn_args = call["function"]["arguments"]

            print(f"  [Tool call] {fn_name}({json.dumps(fn_args)})")

            # Call the tool on the MCP server
            result = await session.call_tool(fn_name, arguments=fn_args)

            # Extract text content from the MCP response
            tool_output = "\n".join(
                c.text for c in result.content if c.type == "text"
            )
            print(f"  [Tool result] {tool_output}")

            # Feed the tool result back to the LLM
            messages.append({
                "role": "tool",
                "name": fn_name,
                "content": tool_output,
            })

    # If we exhausted all rounds, ask LLM to wrap up
    messages.append({"role": "user", "content": "Summarize what you found."})
    final = await call_ollama(messages, tools=[])
    print(f"\nAssistant: {final.get('content', '')}")


# ---------------------------------------------------------------------------
# 5. MAIN — connect to MCP server and start the prompt loop
# ---------------------------------------------------------------------------

async def main():
    # Point to our server_mcp.py via stdio
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "mcp", "run", "server_mcp.py"],
    )

    # Connect: spawn the MCP server as a subprocess, communicate over stdio
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Discover tools from the MCP server
            tools_resp = await session.list_tools()
            ollama_tools = mcp_tools_to_ollama(tools_resp.tools)

            print(f"Connected to MCP server. Found {len(ollama_tools)} tools:")
            for t in ollama_tools:
                print(f"  - {t['function']['name']}: {t['function']['description']}")
            print()

            # Interactive prompt loop
            while True:
                user_input = input("You: ").strip()
                if not user_input or user_input.lower() in {"exit", "quit"}:
                    break
                await agent_loop(session, ollama_tools, user_input)
                print("-" * 40)


if __name__ == "__main__":
    asyncio.run(main())
