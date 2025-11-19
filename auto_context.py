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
import logging
import re
from typing import List, Dict, Set, Optional, Any
from rank_bm25 import BM25Okapi
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

    def __enter__(self) -> "AutoContextPersistent":
        """Enter the runtime context for the AutoContextPersistent object."""
        return self

    def __exit__(self, exc_type: Optional[type], exc_value: Optional[Exception], traceback: Optional[Any]) -> None:
        """This method is kept for compatibility but does nothing, as
        modern ChromaDB versions handle persistence automatically."""
        pass

    def __init__(
        self,
        directory_path: str,
        model_name: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db",
        collection_name: str = "autocontext",
    ) -> None:
        """
        Initializes the AutoContextPersistent object.

        Args:
            directory_path (str): Path to directory containing .txt documents.
            model_name (str): SentenceTransformer model name for embeddings.
            persist_directory (str): Directory where Chroma will persist data.
            collection_name (str): Name of the Chroma collection to create/use.
        """
        self._is_initialized: bool = False
        self.persist_directory: str = persist_directory
        self.collection_name: str = collection_name
        self.chunks: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.client: Optional[chromadb.ClientAPI] = None
        self.collection: Optional[chromadb.Collection] = None
        self.logger = logging.getLogger(__name__)

        # --- 1. Load and Chunk Documents ---
        self.chunks = self._load_and_chunk_documents(directory_path)
        if not self.chunks:
            self.logger.warning("No text chunks found. The directory might be empty or have no .txt files.")
            return

        # --- 2. Build Retrievers ---
        self.logger.info("Building search indexes (BM25 and persistent Chroma vector store)...")
        self._build_retrievers(model_name)
        self.logger.info("AutoContextPersistent is ready.")

        self._is_initialized = True

    def _load_and_chunk_documents(self, directory_path: str) -> List[str]:
        """Loads .txt files and splits them into chunks (paragraphs)."""
        chunks: List[str] = []
        self.metadatas = []
        p = pathlib.Path(directory_path)
        if not p.is_dir():
            self.logger.error(f"Directory not found at {directory_path}")
            return chunks
        for file_path in p.glob("*.txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    paragraphs = [para.strip() for para in content.split("\n\n") if para.strip()]
                    for idx, para in enumerate(paragraphs):
                        chunks.append(para)
                        self.metadatas.append({"file": file_path.name, "paragraph_index": idx})
            except Exception as e:
                self.logger.error(f"Error reading file {file_path}: {e}")
        self.logger.info(f"Loaded {len(chunks)} text chunks from {len(list(p.glob('*.txt')))} files.")
        return chunks

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _build_retrievers(self, model_name: str) -> None:
        """Creates the BM25 sparse index and a persistent Chroma dense index."""
        # --- Build Sparse Retriever (BM25) ---
        tokenized_chunks = [self._tokenize(chunk) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

        # --- Build Persistent Dense Retriever (Chroma) ---
        # Initialize the persistent Chroma client.
        pathlib.Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # Use Chroma's SentenceTransformer wrapper.
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

        # Get or create the collection.
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, embedding_function=embedding_func
        )
        all_ids = [hashlib.sha256(chunk.encode("utf-8")).hexdigest() for chunk in self.chunks]
        id_to_meta = dict(zip(all_ids, self.metadatas))
        # --- Synchronize Documents ---
        # Create content-addressable IDs for current chunks.
        current_chunk_ids: Set[str] = {
            hashlib.sha256(chunk.encode("utf-8")).hexdigest() for chunk in self.chunks
        }
        
        # Get existing IDs from the collection.
        # Note: .get() with no IDs returns all items.
        existing_items = self.collection.get()
        existing_chunk_ids: Set[str] = set(existing_items["ids"])

        # Determine which chunks to add or remove.
        ids_to_add: Set[str] = current_chunk_ids - existing_chunk_ids
        ids_to_remove: Set[str] = existing_chunk_ids - current_chunk_ids

        # Add new chunks to the collection.
        if ids_to_add:
            chunks_to_add: List[str] = [
                chunk
                for chunk in self.chunks
                if hashlib.sha256(chunk.encode("utf-8")).hexdigest() in ids_to_add
            ]
            new_ids: List[str] = [
                hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                for chunk in chunks_to_add
            ]
            metadatas_to_add: List[Dict[str, Any]] = [id_to_meta[i] for i in new_ids]
            self.collection.add(documents=chunks_to_add, ids=new_ids, metadatas=metadatas_to_add)
            self.logger.info(f"Added {len(ids_to_add)} new chunks to the collection.")

        # Remove stale chunks from the collection.
        if ids_to_remove:
            self.collection.delete(ids=list(ids_to_remove))
            self.logger.info(f"Removed {len(ids_to_remove)} stale chunks from the collection.")

        if not ids_to_add and not ids_to_remove:
            self.logger.info("Chroma collection is already up-to-date.")

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

        self.logger.info(f"Retrieving context for query: '{query}'")
        if self.bm25 is None or self.collection is None or not self.chunks:
            return f"AutoContextPersistent not properly initialized. No context available.\n\nQuery: {query}"
        tokenized_query = self._tokenize(query)
        all_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_idx = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)[:num_results]
        bm25_texts = [self.chunks[i] for i in top_bm25_idx]
        bm25_vals = [all_scores[i] for i in top_bm25_idx]
        self.logger.info(f"BM25 found {len(bm25_texts)} keyword-based results.")
        results = self.collection.query(query_texts=[query], n_results=num_results, include=["documents","distances"])
        vector_texts: List[str] = results.get("documents", [[]])[0]
        vector_dists = results.get("distances", [[]])[0]
        self.logger.info(f"Chroma vector search found {len(vector_texts)} semantic-based results.")
        def norm(vals: List[float]) -> List[float]:
            if not vals:
                return []
            vmin, vmax = min(vals), max(vals)
            if vmax == vmin:
                return [0.0 for _ in vals]
            return [(v - vmin) / (vmax - vmin) for v in vals]
        bm25_norm = norm(bm25_vals)
        vec_sims = [1.0 / (1.0 + d) for d in vector_dists] if vector_dists else []
        vec_norm = norm(vec_sims)
        scores = {}
        alpha = 0.5
        for t, s in zip(bm25_texts, bm25_norm):
            scores[t] = max(scores.get(t, 0.0), alpha * s)
        for t, s in zip(vector_texts, vec_norm):
            scores[t] = max(scores.get(t, 0.0), (1 - alpha) * s)
        ranked = [t for t, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:num_results]
        context_str: str = "\n\n---\n\n".join(ranked)
        final_prompt: str = (
            "Based on the following context, please answer the question.\n\n"
            "--- CONTEXT ---\n"
            f"{context_str}\n"
            "--- END CONTEXT ---\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        return final_prompt
