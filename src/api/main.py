from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from pathlib import Path
from typing import List

from src.generation.graceful_fallback import answer_with_fallback
from src.ingestion.loader import load_all_documents
from src.ingestion.chunker import chunk_all_documents
from src.ingestion.indexer import build_index

app = FastAPI(
    title="RAG Hybrid Search API",
    description="Production-grade RAG pipeline with hybrid dense+sparse retrieval, citation verification, and confidence scoring.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str
    strategy: str = "structure_aware"
    top_k: int = 5


class AskResponse(BaseModel):
    status: str
    answer: str
    sources: list[str] = []
    confidence_scores: dict


@app.post("/v1/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = answer_with_fallback(
            request.question,
            strategy=request.strategy,
            top_k=request.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return AskResponse(
        status=result["status"],
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence_scores=result["confidence_scores"],
    )


@app.get("/v1/documents")
def list_documents(raw_dir: str = "data/raw"):
    path = Path(raw_dir)
    if not path.exists():
        return {"documents": []}

    docs = [f.name for f in path.iterdir() if f.is_file()]
    return {"documents": docs, "count": len(docs)}


@app.post("/v1/ingest")
def ingest_documents():
    try:
        load_all_documents()
        chunk_all_documents()
        for strategy in ["fixed_size", "structure_aware", "semantic"]:
            build_index(strategy)
        return {"status": "success", "message": "Documents re-ingested and indexes rebuilt"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    allowed_extensions = {".md", ".txt", ".html", ".htm", ".pdf"}
    saved_files = []
    skipped_files = []

    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed_extensions:
            skipped_files.append(file.filename)
            continue

        dest_path = raw_dir / file.filename
        with open(dest_path, "wb") as f:
            content = await file.read()
            f.write(content)
        saved_files.append(file.filename)

    if not saved_files:
        raise HTTPException(status_code=400, detail="No valid files uploaded. Supported types: .md, .txt, .html, .pdf")

    try:
        load_all_documents()
        chunk_all_documents()
        for strategy in ["fixed_size", "structure_aware", "semantic"]:
            build_index(strategy)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Files saved but re-indexing failed: {e}")

    return {
        "status": "success",
        "uploaded": saved_files,
        "skipped": skipped_files,
        "message": f"{len(saved_files)} file(s) ingested and indexes rebuilt",
    }


@app.get("/")
def root():
    return {"message": "RAG Hybrid Search API is running. Visit /docs for API documentation."}