import asyncio  # mcp's client works asynchronously, gradio can handle async functions directly
from contextlib import AsyncExitStack  # this lets us keep multiple "with" blocks open across calls
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import ollama  # this connects to the local ai model, used only for the routing decision
import gradio as gr  # this builds the interface
import os  # this lets us find mcp_server.py regardless of where this script is run from

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MCP_SERVER_PATH = os.path.join(SCRIPT_DIR, "..", "07_mcp_server", "mcp_server.py")

# these start out empty, and get filled in on the very first question the user asks,
# then reused for every question after that instead of restarting mcp_server.py each time
exit_stack = AsyncExitStack()
session = None
session_lock = asyncio.Lock()  # stops two questions arriving at once from both trying to start a session


# this only actually does anything the first time it's called, after that it just
# hands back the same session object every time, skipping mcp_server.py's startup cost
async def get_session():
    global session

    async with session_lock:
        if session is None:
            server_params = StdioServerParameters(command="python", args=[MCP_SERVER_PATH])
            read, write = await exit_stack.enter_async_context(stdio_client(server_params))
            session = await exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

    return session


def classify_question(question):
    prompt = f"""Decide whether the question below should be answered using:
- DATA: if it requires counting, summing, filtering, or calculating values from a table of sales orders
- TEXT: if it requires information from written notes, explanations, or stories, not a calculation

Respond with only one word: DATA or TEXT.

Question: {question}
"""
    response = ollama.chat(model="llama3.1", messages=[{"role": "user", "content": prompt}])
    reply = response["message"]["content"].strip().upper()
    return "DATA" if "DATA" in reply else "TEXT"


# gradio can call an async function directly as an event handler, it awaits it for us,
# which matters here: get_session() has to run inside gradio's own event loop, not a
# separate one, otherwise the connection created here couldn't be reused safely later
async def chat_respond(message, chat_history):
    route = classify_question(message)
    tool_name = "query_orders" if route == "DATA" else "search_notes"

    active_session = await get_session()
    result = await active_session.call_tool(tool_name, {"question": message})
    answer = "\n".join(item.text for item in result.content)

    tag = f"\n\n---\n🔌 *Tool used (via MCP, persistent session): {tool_name}*"

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": answer + tag})
    return "", chat_history


with gr.Blocks(title="MCP Client UI") as demo:
    gr.Markdown("# MCP Client - Notes + Order Data")
    gr.Markdown("This talks to mcp_server.py over the real MCP protocol, using one persistent connection instead of restarting the server for every question.")

    chatbot = gr.Chatbot(height=500, type="messages")
    msg = gr.Textbox(label="Your question", placeholder="Type a question and press enter...")
    msg.submit(chat_respond, inputs=[msg, chatbot], outputs=[msg, chatbot])

demo.launch(server_port=7862)
