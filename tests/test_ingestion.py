from insurellm.rag.ingestion.chunker import chunk_documents
from insurellm.rag.ingestion.loader import load_documents

# Step 1: Load files into Documents
documents = load_documents()

# Step 2: Split the loaded Documents
chunks = chunk_documents(documents)

print(f"Documents loaded: {len(documents)}")
print(f"Chunks created: {len(chunks)}")

print("=== Sample Chunk ===")
print(chunks[0]) 
