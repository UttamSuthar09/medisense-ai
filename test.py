import chromadb
from chromadb.utils import embedding_functions

# Load RAG collection
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
client = chromadb.PersistentClient(path="chroma_db")
rag_collection = client.get_collection(
    name="medical_knowledge",
    embedding_function=embedding_fn
)
print(f"RAG loaded: {rag_collection.count()} chunks")

def retrieve_context(query, n_results=3):
    try:
        results = rag_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        docs = results['documents'][0]
        return "\n\n---\n\n".join(docs)
    except Exception as e:
        print(f"Retrieval error: {e}")
        return ""

# Test retrieval quality
test_queries = [
    "diabetes symptoms causes treatment",
    "malaria fever chills treatment",
    "tuberculosis cough chest pain",
    "dengue fever rash joint pain",
]

for query in test_queries:
    print(f"\n🔍 Query: {query}")
    print("-" * 50)
    context = retrieve_context(query, n_results=2)
    chunks = context.split("\n\n---\n\n")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk[:300]}...")