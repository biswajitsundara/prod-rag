"""
query.py
Handles: user query → embed query → similarity search → return top-K chunks
"""

# ─────────────────────────────────────────────
# STEP 1: EMBED USER QUERY
# ─────────────────────────────────────────────

def embed_query(query:str) -> list[float]:
    """
    Embed the user's query using the same model as ingestion
    Critical: query and doc embedings MUST use the same model
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = model.encode(query)
        print(f"[Embedder] Query embedded: '{query[:60]}...' " if len(query) > 60
              else f"[Embedder] Query embedded: '{query}'")
        return embedding.tolist()
    
    except ImportError:
        raise ImportError("Install sentence transformer")
    

# ─────────────────────────────────────────────
# STEP 2: SIMILARITY SEARCH (cosine similarity)
# ─────────────────────────────────────────────

import chromadb
DB_DIRECTORY = "./chromadb"
chroma_client = chromadb.PersistentClient(path=DB_DIRECTORY)

collection = chroma_client.get_or_create_collection(
    name="document_chunks",
    metadata={"hnsw:space": "cosine"} 
)
 
print(f"[Chroma] Total chunks currently in '{collection.name}': {collection.count()}")

def retrieve_relevant_chunks(query_vector: list[float], top_k: int = 3) -> list[dict]:
    """
    Takes a query vector from sentence-transformers and finds the 
    top_k most similar chunks from the global ChromaDB collection.
    """
    # Query ChromaDB directly
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )

    # Check if we actually got results back
    if not results or not results['ids'] or len(results['ids'][0]) == 0:
        print("[Search] No matches found.")
        return []

    formatted_results = []

    # Chroma returns lists of lists because it supports batch querying [0] gets our single query
    ids = results['ids'][0]
    distances = results['distances'][0]
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]

    print(f"\n[Search] Top {top_k} chunks retrieved via ChromaDB:")
    for i in range(len(ids)):
        # Convert Chroma's cosine distance back to a similarity score: 1 - distance
        similarity_score = round(1.0 - distances[i], 4)
        meta = metadatas[i] if metadatas[i] else {}
        
        chunk_data = {
            "id": ids[i],
            "text": documents[i],
            "metadata": meta,
            "score": similarity_score,
            "filename": meta.get("filename", "Unknown File")
        }
        formatted_results.append(chunk_data)

        # Print a clean log line for debugging
        chunk_idx = meta.get('chunk_index', 'N/A')
        print(f"  {i+1}. score={chunk_data['score']} | {chunk_data['filename']} | "
              f"chunk_{chunk_idx} | '{chunk_data['text'][:70]}...'")

    return formatted_results

# ─────────────────────────────────────────────
# ENTRYPOINT  →  called by chat-main.py
# ─────────────────────────────────────────────

def retrieve(user_query: str, top_k: int = 3) -> list[dict]:
    """
    Full retrieval pipeline:
    embed query → similarity search → return top-K chunks
    """
    print(f"\n{'='*50}")
    print(f" Query: {user_query}")
    print(f"{'='*50}")

  
    query_embedding = embed_query(user_query)
    top_chunks      = retrieve_relevant_chunks(query_embedding, top_k=top_k)

    return top_chunks

           