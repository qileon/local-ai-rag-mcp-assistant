from mcp.server.mcpserver import MCPServer  # this is the high-level toolkit for building an mcp server
import ollama  # this connects to the local ai model
import chromadb  # this is our local vector database
import pandas as pd  # this handles the structured order data
import re  # this helps us pull a plain number out of the model's text replies
import time  # this lets us give each generated chart a unique filename
import os  # this lets us build absolute paths regardless of the caller's working directory
import matplotlib  # this draws the charts for the make_chart tool
matplotlib.use("Agg")  # non-interactive backend, just saves images to disk, no window popup
import matplotlib.pyplot as plt

# a client like Claude's desktop app can launch this script from any working directory,
# so we can't rely on relative paths like "csvs/Orders.csv" actually pointing here.
# __file__ always points to this script's own location, so we build paths from that instead.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CHARTS_DIR = os.path.join(PROJECT_ROOT, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# --- RAG setup (same as router_ui.py) ---
chroma_client = chromadb.PersistentClient(path=os.path.join(PROJECT_ROOT, "chroma_db"))
collection = chroma_client.get_or_create_collection(name="notes")

# --- pandas setup (same as router_ui.py) ---
df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "orders.csv"))
df['Amount in transaction currency'] = (
    df['Amount in transaction currency']
    .str.replace(',', '')
    .astype(float)
)

# this is the actual mcp server object, "LocalAssistantServer" is just its name,
# a client (like Claude Code) will see this name when it connects
mcp = MCPServer("LocalAssistantServer")


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


# the @mcp.tool() decorator is what actually exposes this function to the outside world,
# without it, this would just be a normal, private python function
@mcp.tool()
def search_notes(question: str) -> str:
    """Search the learning notes to answer questions about what happened,
    decisions made, bugs fixed, or explanations from the written notes."""

    question_embedding = ollama.embed(model="nomic-embed-text", input=question)["embeddings"][0]
    results = collection.query(query_embeddings=[question_embedding], n_results=8)
    candidates = list(zip(results["documents"][0], results["metadatas"][0]))

    scored = [(score_relevance(question, doc), doc, meta) for doc, meta in candidates]
    scored.sort(key=lambda item: item[0], reverse=True)
    top_chunks = scored[:2]

    context = "\n\n".join(doc for score, doc, meta in top_chunks)

    prompt = f"""Answer the question using only the information in the context below.
Do not copy the context word-for-word, write your own answer in your own words.
Always answer in the same language as the question, even if the context is written in a different language.

Context:
{context}

Question: {question}
"""

    response = ollama.chat(model="llama3.1", messages=[{"role": "user", "content": prompt}])
    sources_used = sorted(set(meta["source"] for score, doc, meta in top_chunks))
    return response["message"]["content"] + "\n\nSources: " + ", ".join(sources_used)


@mcp.tool()
def query_orders(question: str) -> str:
    """Query the structured sales order table (columns: Order ID, Date, Amount,
    Customer account, Order Status) to count, sum, filter, or find specific records."""

    columns_info = ", ".join(df.columns)
    status_values = df['Order Status'].unique().tolist()

    prompt = f"""You are given a pandas DataFrame called df with these columns: {columns_info}
The 'Order Status' column contains exactly these values: {status_values}.
Match the exact casing used in these values, do not guess based on how the question is worded.
Write a single line of pandas code that answers the question below.
If the question asks "which" something, return that identifier itself, not a numeric value.
Use idxmax()/idxmin() instead of max()/min() when asking "which one".
Only output the code itself, nothing else, no explanation, no markdown formatting.

Question: {question}
"""

    response = ollama.chat(model="llama3.1", messages=[{"role": "user", "content": prompt}])
    code = response["message"]["content"].strip().replace("```python", "").replace("```", "").strip()

    try:
        result = eval(code)
    except Exception as e:
        return f"Could not run the generated code: {e}\n\nGenerated code was:\n{code}"

    return f"{result}\n\nGenerated code: {code}"


@mcp.tool()
def make_chart(question: str) -> str:
    """Create a chart or graph visualizing the sales order data. Only use this when
    the question explicitly asks for a chart, graph, or visual representation."""

    columns_info = ", ".join(df.columns)
    status_values = df['Order Status'].unique().tolist()

    # each chart gets its own filename, based on the current time, so a previous
    # chart is never accidentally reused or overwritten. saved next to this script,
    # not wherever the caller's working directory happens to be
    chart_path = os.path.join(CHARTS_DIR, f"chart_{int(time.time())}.png")

    prompt = f"""You are given a pandas DataFrame called df with these columns: {columns_info}
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
        exec(code, {"df": df, "plt": plt})
    except Exception as e:
        return f"Could not run the generated chart code: {e}\n\nGenerated code was:\n{code}"

    plt.close("all")
    # unlike search_notes/query_orders, this tool can't return an image as plain text,
    # so instead it returns the file path, the client is responsible for opening it
    return f"Chart saved to: {chart_path}\n\nGenerated code:\n{code}"


@mcp.tool()
def list_files(folder_path: str) -> str:
    """List the filenames inside a given folder, so you can see what's available
    before deciding which one to read with read_file. Use this when you don't
    already know which specific file contains the answer."""

    try:
        return "\n".join(os.listdir(folder_path))
    except FileNotFoundError:
        return f"Folder not found: {folder_path}"
    except NotADirectoryError:
        return f"Not a folder: {folder_path}"


@mcp.tool()
def read_file(filepath: str) -> str:
    """Read and return the raw text content of a specific file, given its full path.
    Use this when the user wants information from a specific file that isn't already
    part of the notes or order data, for example a file they picked themselves."""

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"File not found: {filepath}"
    except UnicodeDecodeError:
        return f"Could not read {filepath} as text (it may be a binary file, like an image or .docx)."


# this starts the server and makes it wait for a client to connect and call its tools,
# it doesn't return/print anything on its own, it just listens
if __name__ == "__main__":
    mcp.run()
