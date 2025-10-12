# AutoContext

AutoContext is an intelligent context retriever designed for Retrieval-Augmented Generation (RAG). It enables enhanced question-answering by retrieving relevant information from a knowledge base of text documents and formatting it as context for a Large Language Model (LLM).

## Features

- **Hybrid Search**: Combines both keyword-based (BM25) and semantic (vector) search methods for more comprehensive retrieval
- **Easy Integration**: Simple API to integrate into your existing LLM workflows
- **Automatic Processing**: Automatically loads, chunks, and indexes text documents from a directory

## How It Works

1. **Document Loading**: AutoContext loads all `.txt` files from a specified directory and splits them into chunks (paragraphs).
2. **Index Building**: It creates two search indexes:
   - **Sparse Retriever (BM25)**: For keyword-based search
   - **Dense Retriever (Vector Store)**: For semantic search using sentence embeddings
3. **Context Retrieval**: For any given query, it retrieves relevant context using both search methods and combines the results.
4. **Prompt Generation**: It formats the retrieved context along with the original query into a structured prompt ready for an LLM.

## Installation

Ensure you have the required dependencies installed:

```bash
pip install rank-bm25 sentence-transformers scikit-learn numpy
```

## Usage

1. Prepare your knowledge base by placing `.txt` files in a directory.
2. Initialize AutoContext with the path to your directory:

```python
from auto_context import AutoContext

# Initialize with your document directory
auto_context = AutoContext(directory_path="./path/to/your/documents")
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

- `directory_path`: Path to the directory containing `.txt` documents
- `model_name`: Name of the SentenceTransformer model to use for embeddings (default: 'all-MiniLM-L6-v2')

#### `get_prompt(query: str, num_results: int = 5) -> str`

Retrieves relevant context and formats it into a prompt for an LLM.

- `query`: The user's question or prompt
- `num_results`: The desired number of context chunks to retrieve (default: 5)
- Returns: A formatted prompt containing the context and the original query

## Example

```python
from auto_context import AutoContext

# Initialize AutoContext
auto_context = AutoContext(directory_path="../data")

# Generate a prompt with context
query = "Explain the theory of relativity"
prompt = auto_context.get_prompt(query, num_results=3)

print(prompt)
```

## Dependencies

- `rank-bm25`: For BM25 keyword-based search
- `sentence-transformers`: For generating sentence embeddings
- `scikit-learn`: For cosine similarity calculations
- `numpy`: For numerical operations

## Model

By default, AutoContext uses the `all-MiniLM-L6-v2` model, which provides a good balance between speed and accuracy. You can specify a different SentenceTransformer model during initialization if needed.
```

This documentation provides a comprehensive overview of the AutoContext project, including its features, how it works, installation instructions, usage examples, and API reference. It should help users understand and effectively use the AutoContext library.
