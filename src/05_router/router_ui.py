import ollama  # this connects to the local ai model
import chromadb  # this is our local vector database, used for the RAG side
import pandas as pd  # this handles the structured order data, used for the pandas side
import gradio as gr  # this builds the interface
import re  # this helps us pull a plain number out of the model's text replies
import time  # this lets us give each generated chart a unique filename
import os
import matplotlib  # this draws the charts for the plot pipeline
matplotlib.use("Agg")  # "Agg" is a non-interactive backend, it just saves images to disk,
# it doesn't try to open a window, which matters because this runs inside a web server
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHROMA_PATH = os.path.join(PROJECT_ROOT, "chroma_db")
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "orders.csv")
CHARTS_DIR = os.path.join(PROJECT_ROOT, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# --- RAG setup ---
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="notes")

# --- pandas setup ---
df = pd.read_csv(CSV_PATH)
df['Amount in transaction currency'] = (
    df['Amount in transaction currency']
    .str.replace(',', '')
    .astype(float)
)


# this describes our three pipelines as real tools, using ollama's native tool calling
# instead of asking the model to output a plain DATA/TEXT/PLOT word, we let it make an
# actual structured tool call, the same mechanism we tested in tool_callingdemo.py
ROUTING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": (
                "Search the written learning notes to answer questions about what "
                "happened, decisions made, bugs fixed, or explanations. Use this for narrative "
                "or explanatory questions, not for counting or calculating order data."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_orders",
            "description": (
                "Query the structured sales order table (columns: Order ID, Date, Amount in "
                "transaction currency, Customer account, Order Status) to count, sum, filter, "
                "or find specific records."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "make_chart",
            "description": (
                "Create a chart or graph visualizing the sales order data. Only use this when "
                "the question explicitly asks for a chart, graph, or visual representation."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]

# maps a tool name back to the route label the rest of the code already expects
TOOL_NAME_TO_ROUTE = {
    "search_notes": "TEXT",
    "query_orders": "DATA",
    "make_chart": "PLOT"
}

# the tools above take no parameters, on purpose. we already have the exact question
# text ourselves, so we don't need (and don't want) the model retyping or paraphrasing
# it into an argument, that would just be another chance for it to drift or invent things,
# same lesson as tool_callingdemo.py: don't trust the model with values we already have
def classify_question(question):
    system_message = {
        "role": "system",
        "content": (
            "You must decide which single tool would best help answer the user's question. "
            "Always call exactly one of the available tools, do not answer the question "
            "yourself directly."
        )
    }

    response = ollama.chat(
        model="llama3.1",
        messages=[system_message, {"role": "user", "content": question}],
        tools=ROUTING_TOOLS
    )

    message = response["message"]

    if message.get("tool_calls"):
        tool_name = message["tool_calls"][0]["function"]["name"]
        if tool_name in TOOL_NAME_TO_ROUTE:
            return TOOL_NAME_TO_ROUTE[tool_name]

    # if the model didn't call anything recognizable, fall back to treating it as text
    return "TEXT"


def score_relevance(question, chunk):
    prompt = f"""On a scale from 1 to 10, how relevant is the following text to answering the question below?
Respond with only a single number, nothing else.

Question: {question}

Text:
{chunk}
"""
    response = ollama.chat(model="llama3.1", messages=[{"role": "user", "content": prompt}])
    match = re.search(r"\d+", response["message"]["content"])
    return int(match.group()) if match else 0


def format_history(chat_history):
    if not chat_history:
        return "(no previous messages)"

    lines = []
    for entry in chat_history:
        lines.append(f"{entry['role']}: {entry['content']}")
    return "\n".join(lines)


def rag_pipeline(question, history_text):
    question_embedding = ollama.embed(model="nomic-embed-text", input=question)["embeddings"][0]

    results = collection.query(query_embeddings=[question_embedding], n_results=8)
    candidates = list(zip(results["documents"][0], results["metadatas"][0]))

    scored = [(score_relevance(question, doc), doc, meta) for doc, meta in candidates]
    scored.sort(key=lambda item: item[0], reverse=True)
    top_chunks = scored[:2]

    context = "\n\n".join(doc for score, doc, meta in top_chunks)

    prompt = f"""Conversation so far:
{history_text}

Answer the question using only the information in the context below.
Do not copy the context word-for-word, write your own answer in your own words.
Always answer in the same language as the question, even if the context is written in a different language.
Use the conversation above only to understand what the question is referring to (e.g. "it", "that day"),
not as a source of facts, the context below is the only source of facts.

Context:
{context}

Question: {question}
"""

    response = ollama.chat(model="llama3.1", messages=[{"role": "user", "content": prompt}])

    sources_used = sorted(set(meta["source"] for score, doc, meta in top_chunks))
    return response["message"]["content"] + "\n\n*Sources: " + ", ".join(sources_used) + "*"


def pandas_pipeline(question, history_text):
    columns_info = ", ".join(df.columns)
    status_values = df['Order Status'].unique().tolist()

    prompt = f"""Conversation so far:
{history_text}

You are given a pandas DataFrame called df with these columns: {columns_info}
The 'Order Status' column contains exactly these values: {status_values}.
Match the exact casing used in these values, do not guess based on how the question is worded.
Write a single line of pandas code that answers the question below.
Use the conversation above only to understand what the question is referring to (e.g. "that customer",
"those orders"), the actual data still comes only from df.
If the question asks "which" something (an identifier, a name, a category), make sure your code
returns that identifier itself, not a numeric value. Use idxmax()/idxmin() instead of max()/min()
when the question is asking "which one", not "what is the highest value".
Only output the code itself, nothing else, no explanation, no markdown formatting.

Question: {question}
"""

    response = ollama.chat(model="llama3.1", messages=[{"role": "user", "content": prompt}])
    code = response["message"]["content"].strip().replace("```python", "").replace("```", "").strip()

    try:
        result = eval(code)
    except Exception as e:
        return f"Could not run the generated code: {e}\n\nGenerated code was:\n`{code}`"

    return f"{result}\n\n*Generated code: `{code}`*"


# this is the new pipeline: instead of returning a number, the model writes code
# that draws a chart with matplotlib and saves it to a file, which we then show in the chat
def plot_pipeline(question, history_text):
    columns_info = ", ".join(df.columns)
    status_values = df['Order Status'].unique().tolist()

    # each chart gets its own filename, based on the current time, so the browser
    # never accidentally shows a cached (old) image instead of the new one
    chart_path = os.path.join(CHARTS_DIR, f"chart_{int(time.time())}.png")

    prompt = f"""Conversation so far:
{history_text}

You are given a pandas DataFrame called df with these columns: {columns_info}
The 'Order Status' column contains exactly these values: {status_values}.
matplotlib.pyplot is already imported as plt, and df already exists, do not re-import or re-load anything.
Write python code that creates a chart answering the question below using plt.
Rotate x-axis labels if they might overlap (plt.xticks(rotation=45)).
Save it with exactly these two lines at the end, in this order:
plt.tight_layout()
plt.savefig("{chart_path}", bbox_inches="tight")
Only output the code itself, nothing else, no explanation, no markdown formatting.

Question: {question}
"""

    response = ollama.chat(model="llama3.1", messages=[{"role": "user", "content": prompt}])
    code = response["message"]["content"].strip().replace("```python", "").replace("```", "").strip()

    try:
        # exec (not eval) because chart code is usually several lines, not a single expression
        # we hand it df and plt explicitly, this is the same execute-generated-code risk as pandas_pipeline
        exec(code, {"df": df, "plt": plt})
    except Exception as e:
        return None, f"Could not run the generated chart code: {e}\n\nGenerated code was:\n`{code}`"

    plt.close("all")  # free up the figure so the next chart starts from a clean state
    return chart_path, f"*Generated code:*\n```python\n{code}\n```"


# returns (route, content, extra) instead of a single string, since a chart needs
# to be handled differently in the chat (an image file, not just text)
def unified_answer(question, mode, history_text):
    if mode == "Force RAG":
        route = "TEXT"
    elif mode == "Force Pandas":
        route = "DATA"
    elif mode == "Force Plot":
        route = "PLOT"
    else:
        route = classify_question(question)

    if route == "PLOT":
        chart_path, extra = plot_pipeline(question, history_text)
        return route, chart_path, extra
    elif route == "DATA":
        answer = pandas_pipeline(question, history_text)
        return route, answer, None
    else:
        answer = rag_pipeline(question, history_text)
        return route, answer, None


def chat_respond(message, chat_history, mode):
    history_text = format_history(chat_history)
    route, content, extra = unified_answer(message, mode, history_text)

    chat_history.append({"role": "user", "content": message})

    if route == "PLOT":
        if content is None:
            # the chart code failed, so we just show the error text, no image to display
            chat_history.append({"role": "assistant", "content": extra})
        else:
            # gradio recognizes a dict with a "path" key as a file to render, in this
            # case an image, since the extension is .png
            chat_history.append({"role": "assistant", "content": {"path": content}})
            tag = "\n\n---\n📊 *Tool used: Plot (chart generation)*"
            chat_history.append({"role": "assistant", "content": extra + tag})
    elif route == "DATA":
        tag = "\n\n---\n🔧 *Tool used: Pandas (structured data query)*"
        chat_history.append({"role": "assistant", "content": content + tag})
    else:
        tag = "\n\n---\n🔍 *Tool used: RAG (text search over notes)*"
        chat_history.append({"role": "assistant", "content": content + tag})

    return "", chat_history


custom_css = """
.gr-dataframe table {
    table-layout: fixed;
    width: 100%;
}
.gr-dataframe th, .gr-dataframe td {
    width: 25%;
    overflow: hidden;
    text-overflow: ellipsis;
}
"""

with gr.Blocks(title="Local Assistant - Notes + Order Data") as demo:
    gr.Markdown("# Local Assistant - Notes + Order Data")
    gr.Markdown("Ask about the notes (text), the order data (numbers), or ask for a chart. Auto mode figures out which one to use, or force one manually with the dropdown.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Order Data")
            gr.Dataframe(value=df, interactive=False, wrap=True)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=500, type="messages")
            msg = gr.Textbox(label="Your question", placeholder="Type a question and press enter...")
            mode = gr.Dropdown(
                choices=["Auto", "Force RAG", "Force Pandas", "Force Plot"],
                value="Auto",
                label="Routing mode"
            )
            msg.submit(chat_respond, inputs=[msg, chatbot, mode], outputs=[msg, chatbot])

demo.launch(
    theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Inter"), "sans-serif"]),
    css=custom_css
)
