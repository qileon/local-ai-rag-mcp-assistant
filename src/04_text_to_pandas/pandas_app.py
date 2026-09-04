import pandas as pd  # this reads and works with tabular data (rows and columns)
import ollama  # this connects to the local ai model
import gradio as gr  # this builds the chat window in the browser
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "orders.csv")

# load the csv once when the app starts, not on every question
df = pd.read_csv(CSV_PATH)

# the amount column comes in as text with thousands separators (e.g. "81,450.00"),
# so pandas reads it as a string, we convert it to a real number here, once,
# instead of hoping the model's generated code somehow handles the conversion itself
df['Amount in transaction currency'] = (
    df['Amount in transaction currency']
    .str.replace(',', '')
    .astype(float)
)


# this runs every time the user sends a message in the chat window
def pandas_answer(question, history):
    columns_info = ", ".join(df.columns)

    # ask the model to translate the question into pandas code, not answer it directly
    prompt = f"""You are given a pandas DataFrame called df with these columns: {columns_info}
Write a single line of pandas code that answers the question below.
Only output the code itself, nothing else. No explanation, no markdown formatting.

Question: {question}
"""

    response = ollama.chat(
        model="llama3.1",
        messages=[{"role": "user", "content": prompt}]
    )

    code = response["message"]["content"].strip()

    # strip markdown fences in case the model adds them anyway
    code = code.replace("```python", "").replace("```", "").strip()

    # actually run the generated code against our real data
    try:
        result = eval(code)
    except Exception as e:
        return f"Could not run the generated code: {e}\n\nGenerated code was:\n`{code}`"

    # show both the answer and the code that produced it, for transparency
    return f"{result}\n\n*Generated code: `{code}`*"


# server_port is set explicitly so this doesn't clash with app.py (the RAG chatbot),
# which already uses the default port (7860)
gr.ChatInterface(fn=pandas_answer, title="Local Pandas Query Assistant").launch(server_port=7861)
