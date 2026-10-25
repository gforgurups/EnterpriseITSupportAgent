from pathlib import Path
from app.core.config import get_settings
from app.services.ingestion import DocumentProcessor
from app.rag.vectorstore import add_documents

settings = get_settings()
docProcessor = DocumentProcessor()
folder = Path(settings.sample_kb_dir)
files = [p for p in folder.iterdir() if p.is_file()]
all_docs=[]
for file in files:
    docs = docProcessor.load_documents(["data/sample_kb/company_it_handbook.md"])
chunks = docProcessor.split_documents(docs)
ids = add_documents(chunks)
print(f"Indexed {len(files)} files -> {len(chunks)} chunks -> {len(ids)} Pinecone vectors")