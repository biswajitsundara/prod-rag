"""
chat-main.py
Handles: retrieved chunks → build prompt → send to LLM → return grounded answer
This is the final step that ties doc-ingestion.py and query.py together.
"""
from query_pipeline import retrieve

# ─────────────────────────────────────────────
# STEP 1: BUILD THE PROMPT
# ─────────────────────────────────────────────

def build_prompt(user_query: str, chunks: list[dict]) -> tuple[str, str]:
    """
    Construct the system + user prompt.

    System prompt  → tells the LLM to answer only from the provided context.
    User prompt    → injects the retrieved chunks + the actual question.

    Keeping system and user prompts separate is best practice with Claude.
    """

    system_prompt = """You are a helpful assistant that answers questions strictly based on the provided context.
    Rules:
    - Only use information from the context below. Do not use outside knowledge.
    - If the answer is not in the context, say: "I don't have enough information to answer this."
    - Always cite the source filename and chunk index at the end of your answer.
    - Be concise and direct.
    """

    # Format each retrieved chunk with its source
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get('metadata', {})
        filename = chunk.get('filename', metadata.get('filename', 'Unknown File'))
        chunk_idx = metadata.get('chunk_index', 'N/A')

        # FIX: Safely try 'text', fallback to 'document', fallback to empty string
        text = chunk.get('text', chunk.get('document', ''))

        source = f"{filename} (chunk_{chunk_idx})"
        context_blocks.append(f"[Source {i}: {source}]\n{text}")
    
    context_text = "\n\n".join(context_blocks)

    user_prompt = f"""
        question: {user_query}
        context: {context_text}
        Answer: 
    """

    return system_prompt, user_prompt    


# ─────────────────────────────────────────────
# STEP 2: CALL THE LLM
# ─────────────────────────────────────────────
import cohere
from dotenv import load_dotenv
load_dotenv()

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Send the augmented prompt to Cohere and get a grounded response.
    """
    # Cohere automatically reads the COHERE_API_KEY environment variable
    client = cohere.ClientV2()  

    response = client.chat(
        model="command-r-plus-08-2024",  
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # Extract the text out of Cohere's V2 response structure
    return response.message.content[0].text

# ─────────────────────────────────────────────
# STEP 3: DISPLAY RESPONSE WITH CITATIONS
# ─────────────────────────────────────────────

def display_response(user_query: str, answer: str, chunks: list[dict]) -> None:
    """
    Print the final answer with source citations.
    """
    print(f"\n{'='*50}")
    print(f" Answer")
    print(f"{'='*50}")
    print(answer)

    #print(f"\n--- Sources Used ---")
    # for i, chunk in enumerate(chunks, 1):
    #     print(f"  [{i}] {chunk['filename']} | "
    #           f"chunk_{chunk['metadata']['chunk_index']} | "
    #           f"score: {chunk['score']}")


# ─────────────────────────────────────────────
# ENTRYPOINT  →  POST /v1/query
# ─────────────────────────────────────────────

def rag_query(user_query: str, top_k: int = 3) -> dict:
    """
    Full RAG pipeline:
    retrieve chunks → build prompt → call LLM → return answer + citations
    """
    # 1. Retrieve relevant chunks (from query.py)
    chunks = retrieve(user_query, top_k=top_k)

    # 2. Build the augmented prompt
    system_prompt, user_prompt = build_prompt(user_query, chunks)

    print(f"\n[Prompt] Sending to LLM with {len(chunks)} context chunks...")

    # 3. Call the LLM
    answer = call_llm(system_prompt, user_prompt)

    # 4. Display with citations
    display_response(user_query, answer, chunks)

    return {
        "query":     user_query,
        "answer":    answer,
        "citations": [
            {
                "source":      chunk["filename"],
                # "chunk_index": chunk["metadata"]["chunk_index"],
                "score":       chunk["score"],
                "text":        chunk["text"],
            }
            for chunk in chunks
        ],
    }


def interactive_rag_chat():
    print("Welcome to the RAG Assistant! (Type 'exit' or 'quit' to stop)\n")

    while True:
        # 1. Capture the user's input from the terminal
        query = input("Ask a question: ").strip()

        # 2. Check if the user wants to break out of the loop
        if query.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break

        # 3. Skip empty inputs if the user just hits 'Enter'
        if not query:
            continue

        # 4. Process the query using your existing RAG pipeline
        print("\nThinking...")
        result = rag_query(query)
        
        # 5. Print the result (assuming rag_query returns text, or adapt to result['text'])
        print(f"\nAnswer: {result['answer']}")
        print("\n" + "─" * 50 + "\n")


def batch_static_queries(static_queries: list[str] = None):
    """
    Runs a predefined list of static queries through the RAG pipeline.
    Useful for testing, benchmarking, or automated demos.
    """
    # Default list if none is provided
    if static_queries is None:
        static_queries = [
            "What is the hotel reimbursement limit?",
            "How many days do I have to submit an expense claim?",
            "Can I fly business class for a 4-hour domestic flight?",
        ]

    print(f"🚀 Running {len(static_queries)} static test queries...\n")

    for i, query in enumerate(static_queries, 1):
        print(f"📝 [Query {i}/{len(static_queries)}]: '{query}'")
        
        # Run your RAG pipeline
        result = rag_query(query)
        
        # print(f"\n💡 [Answer]: {result}")
        print("\n" + "─" * 50 + "\n")
        
    print("✅ Static query batch execution complete.")

# ─────────────────────────────────────────────
# DEMO RUN
# ─────────────────────────────────────────────

from doc_ingestion import ingest

if __name__ == "__main__":
    # Make sure you have run doc_ingestion.py first
    interactive_rag_chat()