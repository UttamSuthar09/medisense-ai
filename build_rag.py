# build_rag.py
import os
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import re

# ── Config ────────────────────────────────────────────────
PDF_FOLDER   = "medical_docs"   # put your PDFs here
CHROMA_PATH  = "chroma_db"      # vector DB will be saved here
CHUNK_SIZE   = 500              # characters per chunk
CHUNK_OVERLAP = 100

# ── Setup ChromaDB with local sentence-transformer embeddings ──
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"   # small, fast, runs fully offline
)

client     = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name="medical_knowledge",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)

# ── PDF text extractor ────────────────────────────────────
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# ── Text chunker ──────────────────────────────────────────
def chunk_text(text, source_name, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = re.sub(r'\s+', ' ', text).strip()
    chunks = []
    ids    = []
    metas  = []
    start  = 0
    idx    = 0
    while start < len(text):
        end   = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 100:   # skip tiny fragments
            chunks.append(chunk)
            ids.append(f"{source_name}_{idx}")
            metas.append({
                'source': source_name,
                'chunk_index': idx,
            })
            idx += 1
        start += chunk_size - overlap
    return chunks, ids, metas

# ── Process all PDFs in folder ────────────────────────────
pdf_folder = Path(PDF_FOLDER)
pdf_folder.mkdir(exist_ok=True)

pdfs = list(pdf_folder.glob("*.pdf"))
if not pdfs:
    print(f"No PDFs found in '{PDF_FOLDER}/' folder.")
    print("Please add medical PDF files there and re-run this script.")
else:
    for pdf_path in pdfs:
        print(f"Processing: {pdf_path.name}")
        text   = extract_text_from_pdf(pdf_path)
        source = pdf_path.stem  # filename without extension

        chunks, ids, metas = chunk_text(text, source)
        if not chunks:
            print(f"  No content extracted from {pdf_path.name}")
            continue

        # Add to ChromaDB in batches of 100
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            collection.add(
                documents=ids[i:i+batch_size],
                ids=ids[i:i+batch_size],
                metadatas=metas[i:i+batch_size],
            )
            # Store actual text separately since Chroma stores embeddings
            collection.upsert(
                documents=chunks[i:i+batch_size],
                ids=ids[i:i+batch_size],
                metadatas=metas[i:i+batch_size],
            )
        print(f"  Added {len(chunks)} chunks from {pdf_path.name}")

    print(f"\nRAG knowledge base built ✓")
    print(f"Total chunks: {collection.count()}")