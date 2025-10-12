import os
import pathlib
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import shutil

class AutoContext:
    """
    An intelligent context retriever for Retrieval-Augmented Generation (RAG).

    This class loads text documents from a directory, processes them into chunks,
    and builds two types of search indexes:
    1. A sparse retriever (BM25) for keyword-based search.
    2. A dense retriever (vector store) for semantic search.

    The get_prompt method uses a hybrid approach, combining results from both
    retrievers to find the most relevant context for a given query.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __init__(self, directory_path: str, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the AutoContext object.

        Args:
            directory_path (str): The path to the directory containing .txt documents.
            model_name (str): The name of the SentenceTransformer model to use for embeddings.
        """
        self._is_initialized = False
        # --- 1. Load and Chunk Documents ---
        self.chunks = self._load_and_chunk_documents(directory_path)
        if not self.chunks:
            print("Heads up: No text chunks found. The directory might be empty or have no .txt files.")
            return

        # --- 2. Build Retrievers ---
        print("Building search indexes (BM25 and vector)...")
        self._build_retrievers(model_name)
        print("All set! AutoContext is ready to go.")
        
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
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Split by paragraph and filter out empty strings
                    paragraphs = [para.strip() for para in content.split('\n\n') if para.strip()]
                    chunks.extend(paragraphs)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}", file=sys.stderr)
        
        print(f"Loaded {len(chunks)} text chunks from {len(list(p.glob('*.txt')))} files.")
        return chunks

    def _build_retrievers(self, model_name: str):
        """Creates the BM25 sparse index and the dense vector index."""
        # --- Build Sparse Retriever (BM25) ---
        tokenized_chunks = [chunk.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

        # --- Build Dense Retriever (SentenceTransformer + NumPy) ---
        # Using SentenceTransformer as it's highly effective for this task.
        # This will download the model on the first run.
        self.embedding_model = SentenceTransformer(model_name)
        self.chunk_embeddings = self.embedding_model.encode(
            self.chunks, 
            show_progress_bar=True,
            normalize_embeddings=True # Normalize for cosine similarity
        )

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
            return f"AutoContext not properly initialized. No context available.\n\nQuery: {query}"

        print(f"\n--- Retrieving context for query: '{query}' ---")

        # --- 1. Sparse Search (BM25) ---
        tokenized_query = query.lower().split()
        bm25_results = self.bm25.get_top_n(tokenized_query, self.chunks, n=num_results)
        print(f"BM25 found {len(bm25_results)} keyword-based results.")

        # --- 2. Dense Search (Vector Similarity) ---
        query_embedding = self.embedding_model.encode(query, normalize_embeddings=True)
        # Reshape for sklearn's cosine_similarity function
        query_embedding_reshaped = query_embedding.reshape(1, -1)
        
        # Calculate similarities and get top N indices
        similarities = cosine_similarity(query_embedding_reshaped, self.chunk_embeddings)[0]
        top_indices = np.argsort(similarities)[-num_results:][::-1]
        vector_results = [self.chunks[i] for i in top_indices]
        print(f"Vector search found {len(vector_results)} semantic-based results.")

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

