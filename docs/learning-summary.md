# Learning Summary

This project started as a local AI learning exercise after the D365 internship work. The goal was not to train a model, but to understand the pieces around a model: local inference, conversation state, retrieval, structured data querying, tool calling, and MCP.

## Experiment Timeline

1. The project started with checking whether a local model could run on the available machine.
2. Ollama was installed and Llama 3.1 8B was used for the first terminal based chat test.
3. A basic Python script sent one question at a time to the local model.
4. Conversation history was added after follow up questions failed without previous messages being sent again.
5. The first RAG pipeline was built by splitting markdown notes, embedding chunks, and storing them in ChromaDB.
6. Retrieval was tested from the terminal by embedding a question, finding nearby chunks, and sending those chunks to the model as context.
7. Prompt instructions were adjusted so answers followed the language of the question and avoided copying the retrieved context directly.
8. Retrieved chunks and source filenames were printed to make the retrieval step easier to inspect.
9. The ingestion script was expanded from one document to a folder of markdown files.
10. Chunking changed from separator based chunks to fixed size overlapping chunks.
11. Reranking was added by asking the local model to score candidate chunks before selecting the final context.
12. Gradio was added so the RAG system could be used through a local browser chat interface without spending time on a custom frontend.
13. D365 order data was exported to CSV and queried with pandas because structured calculations did not fit the RAG approach.
14. The text to pandas flow was improved after discovering that amount values were being treated as strings instead of numbers.
15. A router was built to send questions to RAG, pandas, or chart generation.
16. Chart generation was added by asking the model to create matplotlib code and save a PNG output.
17. Tool calling was tested with a small `add_numbers` function to understand when the model chooses tools.
18. The same tool idea was moved into an MCP server with tools for note search, order queries, charts, and file reading.
19. A custom MCP client and Gradio UI were built to call the MCP server through the protocol instead of importing functions directly.
20. Persistent MCP sessions were benchmarked against starting a new server process for every question.
21. File discovery was tested with `list_files` and `read_file`, which exposed how multi step AI flows can fail when the wrong file is selected first.
22. Qwen 2.5 14B was tested after Llama 3.1 8B hallucinated in some file discovery cases. Qwen was slower, but better at admitting when the selected file did not contain the answer.

## Main Lessons

- Local models do not remember earlier messages unless the application sends conversation history.
- RAG does not teach the model permanently; it retrieves context at question time.
- Similarity search can return related text that still does not answer the question.
- Reranking can improve selected chunks, but only if the first retrieval step finds good candidates.
- Table questions should usually be handled as structured data problems, not text retrieval problems.
- Model generated code can run successfully while still producing the wrong answer.
- Tool calling and MCP improve structure, but they do not remove the need to validate model decisions.
- A larger local model can reduce some hallucination problems, but it can also be much slower on CPU only hardware.

## UI Choice

Gradio was chosen because it was a quick way to wrap the experiments in a usable local browser interface. The project was mainly about learning the AI pipeline, so using ready made chat and input components kept the focus on retrieval, routing, pandas queries, chart generation, and MCP.

The interfaces were kept local. Public sharing was avoided because the original version of the project used private internship notes and local files.
