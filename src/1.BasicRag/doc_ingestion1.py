"""
Document Ingestion
Handles: document loading → chunking → embedding → storing in vector DB
"""

from pathlib import Path
import hashlib

# ─────────────────────────────────────────────
# STEP 1: DOC LOADER
# ─────────────────────────────────────────────

SUPPORTED_LOADERS = {
    ".txt": "TextLoader",
    ".md":  "TextLoader",
    ".pdf": "PyPDFLoader",
    ".docx": "Docx2txtLoader",
}

def load_document(file_path:str) -> dict:
    """
    Load a document using the appropriate LangChain loader based on file type.

    .txt / .md   → TextLoader
    .pdf         → PyPDFLoader
    .docx        → Docx2txtLoader

    Each loader returns a list of LangChain Document objects with:
      - page_content : the extracted text
      - metadata     : source, page number, etc. (loader-specific)
    
    In production: extend this to support S3Loader, ConfluenceLoader,
    NotionLoader, WebBaseLoader, etc.
    """
    from langchain_community.document_loaders import (
        TextLoader,
        PyPDFLoader,
        Docx2txtLoader,
    )

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found {file_path}")
    
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_LOADERS:
        raise ValueError(
            f"Unsupported file type {suffix}"
            f"Supported: {list(SUPPORTED_LOADERS.keys())}"
        )

    if suffix in (".txt", ".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif suffix == ".pdf":
        loader = PyPDFLoader(file_path)          # splits by page automatically
    elif suffix == ".docx":
        loader = Docx2txtLoader(file_path)

    # All loaders expose the same .load() interface → list of Documents
    documents = loader.load()

    content = "\n\n".join(doc.page_content for doc in documents)

    doc_id  = hashlib.md5(content.encode()).hexdigest()[:8]

    print(f"[Loader] '{path.name}' loaded via {SUPPORTED_LOADERS[suffix]} "
          f"({len(documents)} document(s), {len(content)} chars)")
    
    return {
        "id": doc_id,
        "filename": path.name,
        "content": content,
        "metadata": {"source": str(path), "type": suffix}
    }

# -------------------------
# CHUNKER
# -------------------------

def chunk_document(doc: dict, chunk_size: int = 300, overlap: int = 50) -> list[dict]:
    """
    Split document content into overlapping chunks.
    Note: This slices by CHARACTERS.
    """
    content = doc["content"]
    chunks = []
    start = 0
    index = 0

    while start < len(content):
        end = min(start + chunk_size, len(content))
        text = content[start:end].strip()

        if text:
            chunks.append({
                "chunk_id": f"{doc['id']}_chunk_{index}", 
                "doc_id": doc["id"],                       
                "filename": doc["filename"],              
                "text": text,
                "metadata": {**doc["metadata"], "chunk_index": index}
            })
            index += 1
        start += chunk_size - overlap 
        
    print(f"[Chunker] created {len(chunks)} chunks from {doc['filename']}")
    return chunks  


# -----------------------------------
# Embedding Model
# ----------------------------------

# Global initialization (or passed into the function) so it loads ONLY ONCE
try:
    from sentence_transformers import SentenceTransformer
    # Initialize outside the function loop for production-level speed
    _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except ImportError:
    _MODEL = None

def embedding_chunks(chunks: list[dict]) -> list[dict]:
    """
    Generate embeddings for each chunk using an embedding model.
    Returns chunks with 'embedding' field added.
    """
    if _MODEL is None:
        raise ImportError(
            "Could not look up embedding model. Please install sentence-transformers: "
            "`pip install sentence-transformers`"
        )

    if not chunks:
        return []

    # Extract all text blocks to pass to the model as a batch
    texts = [c["text"] for c in chunks]

    # Generate the vector matrices
    embeddings = _MODEL.encode(texts, show_progress_bar=True)
    
    # Assign them back to the dictionary strings
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()  # <-- FIXED: Added quotes around "embedding"
        
    print(f"[Embedder] Embedded {len(chunks)} chunks")
    return chunks    

#--------------------------------------
# VECTOR STORE (ChromaDB Local Client)
# -------------------------------------    
import chromadb

# Define the local directory where Chroma will save its database files
DB_DIRECTORY = "./chromadb"

def save_to_vector_store(chunks: list[dict]) -> None:
    """
    Persist embedded chunks to a local persistent ChromaDB instance
    """
    if not chunks:
        print("[VectorStore] No chunks provided to save")
        return
    
    #Pass the DB_DIRECTORY to the client
    client = chromadb.PersistentClient(path=DB_DIRECTORY)

    # Chroma collections have strict naming rules (alphanumeric, 3-63 chars)
    collection = client.get_or_create_collection(name="document_chunks")

    # Unpack your list of dicts into the flat lists ChromaDB expects
    ids = []
    embeddings = []
    documents = []
    metadatas = []  

    for i, chunk in enumerate(chunks):
        # Generate a unique id
        chunk_id = chunk.get("chunk_id", f"{chunk['doc_id']}_chunk_{i}")
        ids.append(chunk_id)

        # Extract the vector embeddings array
        embeddings.append(chunk["embedding"])

        # Extract the original content
        documents.append(chunk.get("text", ""))

        chunk_metadata = {
            "doc_id": str(chunk["doc_id"]),
            "source": str(chunk.get("source", "unknown"))
        } 

        metadatas.append(chunk_metadata)

    
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    print(f"[VectorStore] Upserted {len(chunks)} chunks to ChromaDB → '{DB_DIRECTORY}' "
          f"(Total collection count: {collection.count()})")
    
#---------------------------------
# ENTRYPOINT  →  POST /v1/ingest
#---------------------------------
def ingest(file_path: str) -> dict:
    """
    Full ingestion pipeline
    load -> chunk -> embed -> store
    """
    print(f"\n{'='*50}")
    print(f" Ingesting: {file_path}")  # FIX: Matches the parameter name now
    print(f"{'='*50}")

    # 1. Load the document from disk/memory
    doc = load_document(file_path)
    
    # 2. Split the document into smaller text snippets
    chunks = chunk_document(doc)
    
    # 3. Generate vectors for those text snippets
    chunks =  embedding_chunks(chunks) 
    
    # 4. Save vectors and metadata to ChromaDB
    save_to_vector_store(chunks)

    print(f"\n✅ Ingestion complete for '{doc['filename']}'")
    return {"doc_id": doc["id"], "chunks_stored": len(chunks)}    


if __name__ == "__main__":
    # Run it only once
    ingest("./docs/company_policy.txt")