# Project Scope

This repository is a cleaned public snapshot of local AI experiments built during an internship learning process.

## Included

- Local chat with Ollama
- Conversation history handling
- RAG ingestion and querying with ChromaDB
- Gradio interfaces for chat based experiments
- Text to pandas querying over exported order data
- Router logic for choosing between text, data, and chart pipelines
- Basic tool calling experiments
- MCP server and client experiments
- A small sample notes file for public RAG testing
- A sample sales order CSV used for structured data questions

## Excluded

- Real internship diary files
- Local ChromaDB database files
- Generated chart images
- Virtual environments and package caches
- Private machine paths, environment names, and company specific notes

## Safety Notes

The text to pandas and chart examples run model generated code locally. They are included to document the learning process, not as production ready execution patterns.

The MCP file reader is a local demo. It should not be exposed through a public share link or connected to untrusted clients without adding stricter file access controls.

## Why The Repository Is Structured This Way

The original folder was intentionally experimental. Each folder represented a separate attempt or learning step. This public version keeps that progression, but renames and groups the folders so the project can be understood without the original private context.
