from pathlib import Path
import hashlib
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────
# STEP 1: DOC LOADER
# ─────────────────────────────────────────────

SUPPORTED_LOADERS = {
    ".txt": "TextLoader",
    ".md":  "TextLoader",
    ".pdf": "PyPDFLoader",
    ".docx": "Docx2txtLoader",
}

def load_document(file_path: str) -> dict:
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
        raise ValueError(f"Unsupported file type {suffix}. Supported: {list(SUPPORTED_LOADERS.keys())}")

    if suffix in ('.txt', '.md'):
        loader = TextLoader(file_path, encoding="utf-8")
    elif suffix == ".pdf":
        loader = PyPDFLoader(file_path)         
    elif suffix == ".docx":
        loader = Docx2txtLoader(file_path)

    documents = loader.load()
    
    # Create an aggregate string for hashing, but track pages if available
    content = "\n\n".join(doc.page_content for doc in documents)
    doc_id = hashlib.md5(content.encode()).hexdigest()[:8]

    print(f"[Loader] '{path.name}' loaded via {SUPPORTED_LOADERS[suffix]} ({len(documents)} document(s))")
    
    return {
        "id": doc_id,
        "filename": path.name,
        "documents": documents, # Pass the rich document objects downstream
        "metadata": {"source": str(path), "type": suffix}
    }

# -------------------------
# CHUNKER (Upgraded to Recursive)
# -------------------------

def chunk_document(doc: dict, chunk_size: int = 300, overlap: int = 50) -> list[dict]:
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk_size to prevent infinite loops.")

    # Using RecursiveCharacterTextSplitter prevents splitting words in half
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len
    )
    
    chunks = []
    index = 0
    
    # Split the internal LangChain documents directly to preserve page numbers
    for raw_doc in doc["documents"]:
        page_text_chunks = splitter.split_text(raw_doc.page_content)
        
        for text in page_text_chunks:
            text = text.strip()
            if not text:
                continue
                
            # Merge global metadata with specific page metadata if it exists
            extended_meta = {**doc["metadata"], **raw_doc.metadata, "chunk_index": index}
            
            chunks.append({
                "chunk_id": f"{doc['id']}_chunk_{index}", 
                "doc_id": doc["id"],                                      
                "filename": doc["filename"],              
                "text": text,
                "metadata": extended_meta
            })
            index += 1
        
    print(f"[Chunker] created {len(chunks)} chunks from {doc['filename']}")
    return chunks  

# -----------------------------------
# Embedding Model
# ----------------------------------

try:
    from sentence_transformers import SentenceTransformer
    _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except ImportError:
    _MODEL = None

def embedding_chunks(chunks: list[dict]) -> list[dict]:
    if _MODEL is None:
        raise ImportError("Please install sentence-transformers: `pip install sentence-transformers`")

    if not chunks:
        return []

    texts = [c["text"] for c in chunks]
    embeddings = _MODEL.encode(texts, show_progress_bar=True)
    
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()
        
    print(f"[Embedder] Embedded {len(chunks)} chunks")
    return chunks    

#--------------------------------------
# VECTOR STORE 
# -------------------------------------    
DB_DIRECTORY = "./chromadb"

def save_to_vector_store(chunks: list[dict]) -> None:
    if not chunks:
        print("[VectorStore] No chunks provided to save")
        return
    
    client = chromadb.PersistentClient(path=DB_DIRECTORY)
    collection = client.get_or_create_collection(name="document_chunks")

    ids, embeddings, documents, metadatas = [], [], [], []

    for i, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id", f"{chunk['doc_id']}_chunk_{i}")
        ids.append(chunk_id)
        embeddings.append(chunk["embedding"])
        documents.append(chunk.get("text", ""))

        # FIX: Correctly look into the nested metadata dictionary
        chunk_metadata = {
            "doc_id": str(chunk["doc_id"]),
            "source": str(chunk["metadata"].get("source", "unknown")),
            "page": str(chunk["metadata"].get("page", 0)) # Now handles PDF pages safely!
        } 
        metadatas.append(chunk_metadata)

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    print(f"[VectorStore] Upserted {len(chunks)} chunks to ChromaDB. Total count: {collection.count()}")
    
#---------------------------------
# ENTRYPOINT
#---------------------------------
def ingest(file_path: str) -> dict:
    print(f"\n{'='*50}\n Ingesting: {file_path}\n{'='*50}")

    doc = load_document(file_path)
    chunks = chunk_document(doc)
    chunks = embedding_chunks(chunks) 
    save_to_vector_store(chunks)

    print(f"\n✅ Ingestion complete for '{doc['filename']}'")
    return {"doc_id": doc["id"], "chunks_stored": len(chunks)}    

if __name__ == "__main__":
    ingest("./docs/company_policy.txt")