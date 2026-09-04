import ollama  # this connects to the local ai model, we use it here just for embeddings
import chromadb  # this is our local vector database
import os  # this lets us list files in a folder automatically

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHROMA_PATH = os.path.join(PROJECT_ROOT, "chroma_db")
DOCS_FOLDER = os.path.join(PROJECT_ROOT, "data", "notes")

# this creates (or opens, if it already exists) a folder on disk called chroma_db
# that's where all our embeddings actually get saved, so we don't redo this every time
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name="notes")

CHUNK_SIZE = 800  # how many characters go into one chunk
OVERLAP = 100  # how many characters we repeat at the start of the next chunk


# this splits a long piece of text into fixed-size pieces, with a bit of overlap between them
# the overlap matters: without it, a sentence that falls right on the cut point would get
# split in half, and neither chunk would have its full meaning
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # move forward by (chunk_size - overlap) instead of chunk_size,
        # so the next chunk repeats the last bit of this one
        start += chunk_size - overlap

    return chunks


# every chunk needs a unique id across ALL files, not just within one file
# so we count up as we go instead of resetting per file
chunk_id = 0

# go through every file in the docs folder, one by one
for filename in os.listdir(DOCS_FOLDER):
    # skip anything that isn't a markdown file
    if not filename.endswith(".md"):
        continue

    filepath = os.path.join(DOCS_FOLDER, filename)

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # this no longer depends on "---" being in the file at all,
    # it works the same way on any plain text regardless of how it's formatted
    chunks = chunk_text(text)

    for chunk in chunks:
        # skip empty pieces (can happen right at the start/end of a file)
        if chunk.strip() == "":
            continue

        # turn this chunk of text into a vector (a list of numbers representing its meaning)
        embedding = ollama.embed(model="nomic-embed-text", input=chunk)["embeddings"][0]

        # store the chunk's text, its vector, and which file it came from
        collection.add(
            ids=[str(chunk_id)],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": filename}]
        )

        chunk_id += 1

print(f"Stored {chunk_id} chunks from {DOCS_FOLDER}.")
