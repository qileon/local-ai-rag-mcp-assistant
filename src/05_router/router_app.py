import ollama  # this connects to the local ai model
import chromadb  # this is our local vector database, used for the RAG side
import pandas as pd  # this handles the structured order data, used for the pandas side
import gradio as gr  # this builds the chat window in the browser
import os
import re  # this helps us pull a plain number out of the model's text replies

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHROMA_PATH = os.path.join(PROJECT_ROOT, "chroma_db")
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "orders.csv")

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


# this is the router. before answering anything, we first ask the model
# what kind of question this is, so we know which system should handle it
def classify_question(question):
    prompt = f"""Decide whether the question below should be answered using:
- DATA: if it requires counting, summing, filtering, or calculating values from a table of sales orders
- TEXT: if it requires information from written notes, explanations, or stories, not a calculation

Respond with only one word: DATA or TEXT.

Question: {question}
"""

    response = ollama.chat(model="llama3.1", messages=[{"role": "user", "content": prompt}])
    reply = response["message"]["content"].strip().upper()

    # default to TEXT if the model's answer isn't clearly one of the two words
    return "DATA" if "DATA" in reply else "TEXT"


# this asks the model to rate, from 1 to 10, how relevant one chunk is to the question
# same reranking logic as query.py
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


# the full RAG pipeline: embed, retrieve 8 candidates, rerank, keep top 2, generate an answer
def rag_pipeline(question):
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
    return response["message"]["content"] + "\n\n*Sources: " + ", ".join(sources_used) + "*"


# the full text-to-pandas pipeline: turn the question into code, run it against the real data
def pandas_pipeline(question):
    columns_info = ", ".join(df.columns)

    prompt = f"""You are given a pandas DataFrame called df with these columns: {columns_info}
Write a single line of pandas code that answers the question below.
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


# this is the single entry point gradio calls for every message
def unified_answer(question, history):
    route = classify_question(question)

    if route == "DATA":
        answer = pandas_pipeline(question)
        tag = "\n\n---\n🔧 *Tool used: Pandas (structured data query)*"
    else:
        answer = rag_pipeline(question)
        tag = "\n\n---\n🔍 *Tool used: RAG (text search over notes)*"

    return answer + tag


gr.ChatInterface(
    fn=unified_answer,
    title="Local Assistant - Notes + Order Data",
    description="Ask about the notes (text) or the order data (numbers), it figures out which one to use.",
    examples=[
        "How many orders are Approved?",
        "Why did you move the stock-decrement logic to SOrderService?",
        "Which customer account has the highest total order amount among Approved orders?",
        "What mistake did you make with the tile button control?"
    ]
).launch()
