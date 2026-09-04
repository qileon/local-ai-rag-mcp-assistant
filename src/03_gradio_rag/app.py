import ollama  # this connects to the local ai model running on this machine
import chromadb  # this is our local vector database, stores the embeddings
import gradio as gr  # this builds the chat window in the browser, no html/css needed
import os
import re  # this helps us pull a plain number out of the model's text reply

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHROMA_PATH = os.path.join(PROJECT_ROOT, "chroma_db")

# connect to the same chromadb folder we created in ingest.py
# this doesn't create anything new, it just opens what's already there
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name="notes")


# this asks the model to rate, from 1 to 10, how relevant one chunk is to the question
# this is the "reranking" step. a slower but more careful check than plain vector similarity
def score_relevance(question, chunk):
    prompt = f"""On a scale from 1 to 10, how relevant is the following text to answering the question below?
Respond with only a single number, nothing else.

Question: {question}

Text:
{chunk}
"""

    response = ollama.chat(
        model="llama3.1",
        messages=[{"role": "user", "content": prompt}]
    )

    reply = response["message"]["content"]

    # the model might still add extra words even though we asked for just a number,
    # so we search the reply for the first number we can find instead of assuming it's clean
    match = re.search(r"\d+", reply)

    if match:
        return int(match.group())

    # if we couldn't find a number at all, treat this chunk as not relevant
    return 0


# this function runs every time the user sends a message in the chat window
# gradio calls it automatically and expects a string back as the reply
def rag_answer(question, history):
    # turn the question into a vector (a list of numbers representing its meaning)
    question_embedding = ollama.embed(model="nomic-embed-text", input=question)["embeddings"][0]

    # first pass: cheap and fast, just plain vector similarity
    # we grab more candidates than we actually need (8 instead of 2), since reranking narrows it down
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=8
    )

    candidates = list(zip(results["documents"][0], results["metadatas"][0]))

    # second pass: slower but more careful, score each candidate individually against the question
    scored_candidates = []
    for doc, meta in candidates:
        score = score_relevance(question, doc)
        scored_candidates.append((score, doc, meta))

    # sort by score, highest first, and keep only the top 2
    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    top_chunks = scored_candidates[:2]

    # only the reranked, top-scoring chunks go into the context now
    context = "\n\n".join(doc for score, doc, meta in top_chunks)

    prompt = f"""Answer the question using only the information in the context below.
Do not copy the context word-for-word — write your own answer in your own words.
Always answer in the same language as the question, even if the context is written in a different language.

Context:
{context}

Question: {question}
"""

    response = ollama.chat(
        model="llama3.1",
        messages=[{"role": "user", "content": prompt}]
    )

    # figure out which files the answer's context actually came from
    sources_used = sorted(set(meta["source"] for score, doc, meta in top_chunks))
    sources_line = "\n\n*Sources: " + ", ".join(sources_used) + "*"

    return response["message"]["content"] + sources_line


# this builds the chat interface and starts a local web server
# fn is the function gradio calls on every message, it handles the rest (bubbles, input box, etc.)
gr.ChatInterface(fn=rag_answer, title="Local RAG Assistant").launch()
