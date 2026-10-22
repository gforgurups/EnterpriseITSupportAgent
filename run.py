from app.core.config import get_settings
from app.services.ingestion import DocumentProcessor
from app.rag.vectorstore import add_documents

settings = get_settings()

print(f"App Name: {settings.app_name}")

docProcessor = DocumentProcessor()
chunks = docProcessor.ingest_documents(["data/sample_kb/company_it_handbook.md"])
print(len(chunks))

docs = add_documents(chunks)

print(len(docs))

