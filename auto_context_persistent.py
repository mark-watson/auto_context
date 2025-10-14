#
# Persistent AutoContext using Chroma (chromadb) for vector storage.
#
# This file is a copy of auto_context.py but uses a persistent Chroma
# collection to store and retrieve dense embeddings instead of keeping
# them in-memory as a NumPy array.
#
# NOTE: This requires the 'chromadb' package to be installed.
# You can install it with: pip install chromadb
#
import hashlib
import pathlib
import sys
from rank_bm25 import BM25Okapi

# Optional imports for compatibility with environments that already
# have sentence-transformers installed (declared in pyproject.toml).
from sentence_transformers import SentenceTransformer

# chromadb imports
import chromadb
from chromadb.utils import embedding_functions


class AutoContextPersistent:
    """
    An intelligent context retriever for Retrieval-Augmented Generation (RAG)
    that persists dense embeddings in a Chroma (chromadb) collection.

    Key differences from the in-memory implementation:
    - Uses chromadb.Client with a persist_directory to store vectors on disk.
    - Creates or reuses a named collection to avoid recomputing embeddings
      across runs.
    - Keeps BM25 sparse retriever in-memory for keyword search.
    """

    def __enter__(self):
        """Enter the runtime context for the AutoContextPersistent object."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """This method is kept for compatibility but does nothing, as
        modern ChromaDB versions handle persistence automatically."""
        pass

    def __init__(
        self,
        directory_path: str,
        model_name: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db",
        collection_name: str = "autocontext",
    ):
        """
        Initializes the AutoContextPersistent object.

        Args:
            directory_path (str): Path to directory containing .txt documents.
            model_name (str): SentenceTransformer model name for embeddings.
            persist_directory (str): Directory where Chroma will persist data.
            collection_name (str): Name of the Chroma collection to create/use.
        """
        self._is_initialized = False
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # --- 1. Load and Chunk Documents ---
        self.chunks = self._load_and_chunk_documents(directory_path)
        if not self.chunks:
            print(
                "Heads up: No text chunks found. The directory might be empty or have no .txt files."
            )
            return

        # --- 2. Build Retrievers ---
        print("Building search indexes (BM25 and persistent Chroma vector store)...")
        self._build_retrievers(model_name)
        print("All set! AutoContextPersistent is ready to go.")

        self._is_initialized = True

    def _load_and_chunk_documents(self, directory_path: str) -> list[str]:
        """Loads .txt files and splits them into chunks (paragraphs)."""
        chunks = []
        p = pathlib.Path(directory_path)
        if not p.is_dir():
            print(f"Error: Directory not found at {directory_path}", file=sys.stderr)
            return chunks

        for file_path in p.glob("*.txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Split by paragraph and filter out empty strings
                    paragraphs = [para.strip() for para in content.split("\n\n") if para.strip()]
                    chunks.extend(paragraphs)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}", file=sys.stderr)

        print(f"Loaded {len(chunks)} text chunks from {len(list(p.glob('*.txt')))} files.")
        return chunks

    def _build_retrievers(self, model_name: str):
        """Creates the BM25 sparse index and a persistent Chroma dense index."""
        # --- Build Sparse Retriever (BM25) ---
        tokenized_chunks = [chunk.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

        # --- Build Persistent Dense Retriever (Chroma) ---
        # Initialize the persistent Chroma client.
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # Use Chroma's SentenceTransformer wrapper.
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

        # Get or create the collection.
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, embedding_function=embedding_func
        )

        # --- Synchronize Documents ---
        # Create content-addressable IDs for current chunks.
        current_chunk_ids = {
            hashlib.sha256(chunk.encode("utf-8")).hexdigest() for chunk in self.chunks
        }
        
        # Get existing IDs from the collection.
        # Note: .get() with no IDs returns all items.
        existing_items = self.collection.get()
        existing_chunk_ids = set(existing_items["ids"])

        # Determine which chunks to add or remove.
        ids_to_add = current_chunk_ids - existing_chunk_ids
        ids_to_remove = existing_chunk_ids - current_chunk_ids

        # Add new chunks to the collection.
        if ids_to_add:
            chunks_to_add = [
                chunk
                for chunk in self.chunks
                if hashlib.sha256(chunk.encode("utf-8")).hexdigest() in ids_to_add
            ]
            new_ids = [
                hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                for chunk in chunks_to_add
            ]
            self.collection.add(documents=chunks_to_add, ids=new_ids)
            print(f"Added {len(ids_to_add)} new chunks to the collection.")

        # Remove stale chunks from the collection.
        if ids_to_remove:
            self.collection.delete(ids=list(ids_to_remove))
            print(f"Removed {len(ids_to_remove)} stale chunks from the collection.")

        if not ids_to_add and not ids_to_remove:
            print("Chroma collection is already up-to-date.")

    def get_prompt(self, query: str, num_results: int = 5) -> str:
        """
        Retrieves relevant context and formats it into a prompt for an LLM.

        Args:
            query (str): The user's question or prompt.
            num_results (int): The desired number of context chunks to retrieve.

        Returns:
            str: A formatted prompt containing the context and the original query.
        """
        if not self._is_initialized:
            return f"AutoContextPersistent not properly initialized. No context available.\n\nQuery: {query}"

        print(f"\n--- Retrieving context for query: '{query}' ---")

        # --- 1. Sparse Search (BM25) ---
        tokenized_query = query.lower().split()
        bm25_results = self.bm25.get_top_n(tokenized_query, self.chunks, n=num_results)
        print(f"BM25 found {len(bm25_results)} keyword-based results.")

        # --- 2. Dense Search (Chroma) ---
        results = self.collection.query(query_texts=[query], n_results=num_results)
        # results['documents'] is a list-of-lists: one list per query
        vector_results = results.get("documents", [[]])[0]
        print(f"Chroma vector search found {len(vector_results)} semantic-based results.")

        # --- 3. Hybridization: Combine and deduplicate results ---
        combined_results = bm25_results + vector_results
        # Use a dictionary to maintain order while removing duplicates
        unique_results = list(dict.fromkeys(combined_results))
        print(f"Combined and deduplicated, we have {len(unique_results)} context chunks.")

        # --- 4. Format the final prompt ---
        context_str = "\n\n---\n\n".join(unique_results)

        final_prompt = (
            "Based on the following context, please answer the question.\n\n"
            "--- CONTEXT ---\n"
            f"{context_str}\n"
            "--- END CONTEXT ---\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

        return final_prompt
