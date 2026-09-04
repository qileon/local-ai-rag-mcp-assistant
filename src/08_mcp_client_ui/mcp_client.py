import asyncio  # mcp's client works asynchronously, so we need asyncio to run it
import os  # this lets us find mcp_server.py regardless of where this script is run from
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MCP_SERVER_PATH = os.path.join(SCRIPT_DIR, "..", "07_mcp_server", "mcp_server.py")


async def main():
    # this describes how to start our server: run python with mcp_server.py as the argument
    server_params = StdioServerParameters(
        command="python",
        args=[MCP_SERVER_PATH]
    )

    # this actually launches mcp_server.py as a background process and connects to it
    # over its stdin/stdout, the exact same "pipe" the Inspector was using
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # this handshake has to happen before anything else works
            await session.initialize()

            # ask the server what tools it has, same as Inspector's "List Tools" button
            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")

            question = input("\nQuestion: ")

            # actually call the search_notes tool with our question as its argument
            result = await session.call_tool("search_notes", {"question": question})

            print("\nAnswer:")
            for item in result.content:
                print(item.text)


asyncio.run(main())
