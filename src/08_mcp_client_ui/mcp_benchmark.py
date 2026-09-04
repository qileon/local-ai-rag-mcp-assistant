import asyncio  # mcp's client works asynchronously
import time  # this lets us measure how long each approach actually takes
import os  # this lets us find mcp_server.py regardless of where this script is run from
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MCP_SERVER_PATH = os.path.join(SCRIPT_DIR, "..", "07_mcp_server", "mcp_server.py")

# same 3 questions asked both ways, so the comparison is fair
QUESTIONS = [
    "How many orders are Approved?",
    "How many orders are Draft?",
    "How many orders are Cancelled?",
]


# this is exactly what mcp_ui.py does today: open a brand new connection,
# meaning a brand new mcp_server.py process, for every single question
async def ask_with_fresh_session(question):
    server_params = StdioServerParameters(command="python", args=[MCP_SERVER_PATH])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("query_orders", {"question": question})
            return "\n".join(item.text for item in result.content)


async def benchmark_fresh_sessions():
    print("--- Fresh session per question (like mcp_ui.py) ---")
    total_start = time.time()

    for question in QUESTIONS:
        start = time.time()
        await ask_with_fresh_session(question)
        print(f"'{question}' took {time.time() - start:.1f}s")

    print(f"Total: {time.time() - total_start:.1f}s\n")


# this opens the connection once, then reuses the same session for all 3 questions,
# the mcp_server.py process only starts up (and loads chromadb/pandas) a single time
async def benchmark_persistent_session():
    print("--- One persistent session for all questions ---")
    server_params = StdioServerParameters(command="python", args=[MCP_SERVER_PATH])
    total_start = time.time()

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            for question in QUESTIONS:
                start = time.time()
                await session.call_tool("query_orders", {"question": question})
                print(f"'{question}' took {time.time() - start:.1f}s")

    print(f"Total: {time.time() - total_start:.1f}s\n")


async def main():
    await benchmark_fresh_sessions()
    await benchmark_persistent_session()


asyncio.run(main())
