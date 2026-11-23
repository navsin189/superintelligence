## Prerequities
- [Install UV](https://docs.astral.sh/uv/getting-started/installation/)
- `uv sync` --> Download all the package
- `uv python install` --> if python is not installed
- [Install Ollama](https://ollama.com/download)
  - Select local based on your preferences. Example - **granite4.2:3b**
  - Alterntively, use remote LLM via API keys.
  - Update the model name in **modelcontextprotocol.py**
- `uv run modelcontextprotocol.py` and enter prompt
- `inspect_mcp_tools.py` is just meant to check the available tools.