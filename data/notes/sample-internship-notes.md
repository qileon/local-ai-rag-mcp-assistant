# Sample Internship Notes

This sample file exists so the RAG examples can be tested without publishing private internship notes.

Day 1 focused on running a local language model with Ollama and understanding that the application must keep conversation history if it wants follow up questions to work.

Day 2 focused on retrieval augmented generation. Notes were split into overlapping chunks, embedded with a local embedding model, and stored in ChromaDB. Questions were embedded the same way, and the closest chunks were passed to the language model as context.

Day 3 focused on structured data. Sales order records were exported to CSV, loaded with pandas, and queried by asking the model to generate pandas expressions. This showed that RAG is useful for text, while table questions are better handled with structured data operations.

Day 4 focused on tool calling and MCP. Existing local functions were exposed as tools so another client could call them through a protocol instead of importing the Python code directly.
