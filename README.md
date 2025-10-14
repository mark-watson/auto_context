# AutoContext

My goal in writing this utility is to automatically generate effective one-shot prompts given:
- A query from a user.
- a Knowledge base of text documents stored in a specified directory.

AutoContext is an intelligent context retriever designed for Retrieval-Augmented Generation (RAG). It enables enhanced question-answering by retrieving relevant information from a knowledge base of text documents and formatting it as context for a Large Language Model (LLM).

This project provides one implementation:
- **AutoContextPersistent**: A persistent context retriever that uses ChromaDB to store embeddings.

## Use in other projects

Add using:

    uv init
    uv add git+https://github.com/mark-watson/auto_context.git 
    uv run python3
    >>> import auto_context
    >>> from auto_context import AutoContextPersistent

    
## Features

- **Hybrid Search**: Combines both keyword-based (BM25) and semantic (vector) search methods for more comprehensive retrieval.
- **Persistent Storage**: `AutoContextPersistent` uses ChromaDB to store vector embeddings, avoiding the need to recompute them on each run.
- **Easy Integration**: Simple API to integrate into your existing LLM workflows.
- **Automatic Processing**: Automatically loads, chunks, and indexes text documents from a directory.

## How It Works

1. **Document Loading**: AutoContext loads all `.txt` files from a specified directory and splits them into chunks (paragraphs).
2. **Index Building**: It creates two search indexes:
   - **Sparse Retriever (BM25)**: For keyword-based search.
   - **Dense Retriever (Vector Store)**: For semantic search using sentence embeddings. `AutoContextPersistent` stores these embeddings in a ChromaDB collection.
3. **Context Retrieval**: For any given query, it retrieves relevant context using both search methods and combines the results.
4. **Prompt Generation**: It formats the retrieved context along with the original query into a structured prompt ready for an LLM.

## Installation

Ensure you have the required dependencies installed:

```bash
pip install rank-bm25 sentence-transformers scikit-learn numpy chromadb
```

## Usage of AutoContextPersistent (Persistent Storage)

1. Prepare your knowledge base by placing `.txt` files in a directory.
2. Initialize `AutoContextPersistent` with the path to your directory and a path for the persistent storage:

```python
from auto_context_persistent import AutoContextPersistent

# Initialize with your document directory and a directory for the ChromaDB database
auto_context = AutoContextPersistent(directory_path="data", persist_directory="./chroma_db")
```

3. Generate augmented prompts for your queries:

```python
# Retrieve context and generate a prompt
query = "Your question here"
augmented_prompt = auto_context.get_prompt(query, num_results=3)

# Use the augmented prompt with your LLM
# llm_response = your_llm.generate(augmented_prompt)
```

## API Reference

### `AutoContext`

#### `__init__(directory_path: str, model_name: str = 'all-MiniLM-L6-v2')`

Initializes the AutoContext object.

- `directory_path`: Path to the directory containing `.txt` documents.
- `model_name`: Name of the SentenceTransformer model to use for embeddings (default: 'all-MiniLM-L6-v2').

#### `get_prompt(query: str, num_results: int = 5) -> str`

Retrieves relevant context and formats it into a prompt for an LLM.

- `query`: The user's question or prompt.
- `num_results`: The desired number of context chunks to retrieve (default: 5).
- Returns: A formatted prompt containing the context and the original query.

### `AutoContextPersistent`

#### `__init__(directory_path: str, model_name: str = "all-MiniLM-L6-v2", persist_directory: str = "./chroma_db", collection_name: str = "autocontext")`

Initializes the `AutoContextPersistent` object.

- `directory_path` (str): Path to directory containing .txt documents.
- `model_name` (str): SentenceTransformer model name for embeddings.
- `persist_directory` (str): Directory where Chroma will persist data.
- `collection_name` (str): Name of the Chroma collection to create/use.

#### `get_prompt(query: str, num_results: int = 5) -> str`

Retrieves relevant context and formats it into a prompt for an LLM.

- `query`: The user's question or prompt.
- `num_results`: The desired number of context chunks to retrieve (default: 5).
- Returns: A formatted prompt containing the context and the original query.


## Dependencies

- `rank-bm25`: For BM25 keyword-based search.
- `sentence-transformers`: For generating sentence embeddings.
- `scikit-learn`: For cosine similarity calculations.
- `numpy`: For numerical operations.
- `chromadb`: For persistent vector storage in `AutoContextPersistent`.

## Model

By default, AutoContext uses the `all-MiniLM-L6-v2` model, which provides a good balance between speed and accuracy. You can specify a different SentenceTransformer model during initialization if needed.
