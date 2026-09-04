import asyncio  # mcp's client works asynchronously, gradio can handle async functions directly
from contextlib import AsyncExitStack  # this lets us keep the mcp connection open across questions
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import ollama  # this connects to the local ai model
import gradio as gr  # this builds the interface
import os  # this helps us build a display name from a full path, and find mcp_server.py

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
MCP_SERVER_PATH = os.path.join(SCRIPT_DIR, "..", "07_mcp_server", "mcp_server.py")

# this is where the model looks for files when you haven't manually picked one yourself
DEFAULT_FOLDER = os.path.join(PROJECT_ROOT, "data", "notes")

exit_stack = AsyncExitStack()
session = None
session_lock = asyncio.Lock()


async def get_session():
    global session
    async with session_lock:
        if session is None:
            server_params = StdioServerParameters(command="python", args=[MCP_SERVER_PATH])
            read, write = await exit_stack.enter_async_context(stdio_client(server_params))
            session = await exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
    return session


async def call_tool(tool_name, arguments):
    active_session = await get_session()
    result = await active_session.call_tool(tool_name, arguments)
    return "\n".join(item.text for item in result.content)


# turns the chat_history list of {"role": ..., "content": ...} dicts into plain text,
# so we can drop it into a prompt as "here's what was said before"
def format_history(chat_history):
    if not chat_history:
        return "(no previous messages)"

    lines = []
    for entry in chat_history:
        lines.append(f"{entry['role']}: {entry['content']}")
    return "\n".join(lines)


# when no file has been manually chosen, this is what "pulling data from files" actually
# means: first see what's available, then decide which one is worth opening, then open it
async def discover_file(question, history_text):
    file_list_text = await call_tool("list_files", {"folder_path": DEFAULT_FOLDER})

    pick_prompt = f"""Conversation so far:
{history_text}

Here is a list of filenames in a folder:
{file_list_text}

Based on the question below, which single filename is most likely to contain the answer?
Use the conversation above only to understand what the question is referring to (e.g. "that file", "it").
Respond with only the filename, nothing else.

Question: {question}
"""
    pick_response = ollama.chat(model="qwen2.5:14b", messages=[{"role": "user", "content": pick_prompt}])
    chosen_filename = pick_response["message"]["content"].strip()

    return os.path.join(DEFAULT_FOLDER, chosen_filename), chosen_filename


async def chat_respond(message, chat_history, uploaded_file):
    # build the history text from everything said BEFORE this new message
    history_text = format_history(chat_history)

    if uploaded_file:
        # a specific file was manually picked, so we lock onto exactly that one,
        # no exploring, no letting the model second-guess the choice
        filepath = uploaded_file
        display_name = os.path.basename(filepath)
        mode_tag = f"manually selected: {display_name}"
    else:
        # nothing picked, so the model has to figure out where to even look first
        filepath, display_name = await discover_file(message, history_text)
        mode_tag = f"auto-discovered: {display_name}"

    file_content = await call_tool("read_file", {"filepath": filepath})

    prompt = f"""Conversation so far:
{history_text}

Answer the question using only the information in the document below.
Do not copy the document word-for-word, write your own answer in your own words.
Use the conversation above only to understand what the question is referring to (e.g. "it", "that file"),
not as a source of facts, the document below is the only source of facts.

Document ({display_name}):
{file_content}

Question: {message}
"""

    response = ollama.chat(model="qwen2.5:14b", messages=[{"role": "user", "content": prompt}])
    answer = response["message"]["content"]

    chat_history.append({"role": "user", "content": message})
    chat_history.append({
        "role": "assistant",
        "content": answer + f"\n\n---\n📄 *File ({mode_tag}) via MCP*"
    })
    return "", chat_history


custom_css = """
#file-picker { border-radius: 12px; }
.gr-chatbot { border-radius: 12px; }
"""

with gr.Blocks(title="MCP File Reader") as demo:
    gr.Markdown("# 📄 MCP File Reader")
    gr.Markdown(
        "Pick a file to focus on it directly, or leave it empty and the model will "
        "look through the folder itself to find the right one."
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=260):
            gr.Markdown("### Optional: pick a file")
            file_picker = gr.File(
                label="Leave empty to let the model search on its own",
                type="filepath",
                elem_id="file-picker"
            )

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=520, show_label=False, type="messages")
            msg = gr.Textbox(placeholder="Ask a question...", show_label=False)
            msg.submit(chat_respond, inputs=[msg, chatbot, file_picker], outputs=[msg, chatbot])

demo.launch(
    server_port=7863,
    theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Inter"), "sans-serif"]),
    css=custom_css
)
