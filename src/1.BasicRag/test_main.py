from doc_ingestion import load_document, chunk_document, embedding_chunks, save_to_vector_store
from sentence_transformers import util


def test_doc_loader():
    result = load_document("./docs/company_policy.txt")
    import json
    print(json.dumps(result, indent=4))

def test_chunker():
    mock_doc = {
        "id": "doc_123",
        "filename": "sample_policy.txt",
        "content": "The quick brown fox jumps over the lazy dog. Artificial Intelligence is changing how we build software pipelines every single day.",
        "metadata": {"source": "docs/sample_policy.txt", "type": ".txt"}
    }

    chunks = chunk_document(mock_doc, chunk_size=40, overlap=15)

    import json
    for chunk in chunks:
        print(f"🧩 CHUNK INDEX: {chunk['metadata']['chunk_index']}")
        print(f"ID:   {chunk['chunk_id']}")
        print(f"TEXT: \"{chunk['text']}\"")
        print("-" * 40)    

def test_embeddings():
    mock_chunks = [
        {"chunk_id": "c1", "text": "Python is an excellent language for building AI pipelines."},
        {"chunk_id": "c2", "text": "Writing code in Python makes developing machine learning apps fast."},
        {"chunk_id": "c3", "text": "The price of apples and oranges increased at the local market."}
    ]

    embedded_chunks = embedding_chunks(mock_chunks)

    first_chunk = embedded_chunks[0]
    vector_dims = len(first_chunk["embedding"])
    
    print("✅ Structure Verification:")
    print(f"   - Is 'embedding' key present? {'embedding' in first_chunk}")
    print(f"   - Vector Type: {type(first_chunk['embedding'])}")
    print(f"   - Vector Dimensions: {vector_dims} (Should be 384 for all-MiniLM-L6-v2)\n")

    # Semantic Math Check (The Vector Space Test)
    v1 = embedded_chunks[0]["embedding"]
    v2 = embedded_chunks[1]["embedding"]
    v3 = embedded_chunks[2]["embedding"]


    similarity_similar = util.cos_sim(v1, v2).item()
    similarity_different = util.cos_sim(v1, v3).item()

    print("📊 Semantic Math Verification:")
    print(f"   - Similarity (Python AI vs Python ML): {similarity_similar:.4f} (Should be HIGH ~ 0.6 - 0.9)")
    print(f"   - Similarity (Python AI vs Fruit Prices): {similarity_different:.4f} (Should be LOW ~ 0.0 - 0.2)")

    # Assert to flag an official issue if the math makes no sense
    assert similarity_similar > similarity_different, "Error: Semantic spacing is broken!"
    print("\n🎉 All embedding tests passed perfectly!")


def test_vector_store():
    import chromadb
    DB_DIRECTORY = "./chromadb"
    mock_chunks = [
        {
            "doc_id": "doc_abc",
            "text": "The quick brown fox jumps over the lazy dog.",
            "embedding": [0.1, 0.2, 0.3, 0.4],
            "source": "animal_stories.txt"
        },
        {
            "doc_id": "doc_abc",
            "text": "Artificial Intelligence is transforming the tech landscape.",
            "embedding": [0.9, 0.8, 0.7, 0.6],
            "source": "tech_news.txt"
        }
    ]
    print("\n[Test] Running save_to_vector_store()...")
    save_to_vector_store(mock_chunks)

    client = chromadb.PersistentClient(path=DB_DIRECTORY)
    collection = client.get_collection(name="document_chunks")
    
    # Fetch everything in the collection
    results = collection.get()
    
    print("\n--- Verification Results ---")
    print(f"IDs found: {results['ids']}")
    print(f"Documents found: {results['documents']}")
    print(f"Metadata found: {results['metadatas']}")

    assert len(results['ids']) >= 2, "Test Failed: Expected at least 2 items in DB."
    assert "doc_abc_chunk_0" in results['ids'], "Test Failed: ID naming convention broken."
    
    print("\n[Test Status] SUCCESS! Data successfully written and verified.")


def main():
   #test_doc_loader()
   #test_chunker()
   #test_embeddings()
   test_vector_store()



if __name__ == "__main__":
    main()
