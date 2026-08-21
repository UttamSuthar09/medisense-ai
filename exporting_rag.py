import chromadb
from chromadb.utils import embedding_functions
import pickle

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("medical_knowledge", embedding_function=embedding_fn)

all_data = collection.get(include=['documents', 'metadatas', 'embeddings'])

with open('rag_knowledge.pkl', 'wb') as f:
    pickle.dump(all_data, f)

print(f"Exported {len(all_data['documents'])} chunks ✓")