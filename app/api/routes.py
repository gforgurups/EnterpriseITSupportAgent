from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from pydantic import BaseModel, Field
from app.core.config import get_settings
from app.rag.workflow import ask
from app.rag.vectorstore import add_documents
from app.services.ingestion import DocumentProcessor
from app.services.audit import write_audit

router = APIRouter(prefix="/api")
settings = get_settings()

class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=3000)

@router.get("/healthcheck")
def health_check():
    return {"status":"ok", "service":settings.app_name}

@router.post("/chat")
def chat(payload: ChatRequest):
    try:
        result = ask(payload.question)
        write_audit(payload.question, result["source_used"], result.get("trace",[]))

        return {
            "answer": result["answer"],
            "source_used": result["source_used"],
            "trace": result.get("trace",[]),
            "citations":result.get("citations",[]),
            "rewritten_query": result.get("current_query",payload.question)
        }

    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex)) from ex

@router.post("/ingest")
async def ingest(file: UploadFile = File(...), x_admin_key: str = Header(default="")):
    supported_file_types = [".pdf", ".txt", ".md", ".docx"]
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in supported_file_types:
        raise HTTPException(status_code=400, detail=f"Supported: {', '.join(supported_file_types)}")
    docProcessor = DocumentProcessor()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / Path(file.filename).name
    dest.write_bytes(await file.read())
    chunks =  docProcessor.ingest_documents([str(dest)])
    ids = add_documents(chunks)
    return {"message": "Document indexed", "file": dest.name, "chunks": len(chunks), "ids_created": len(ids)}