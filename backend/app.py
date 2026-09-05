import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from backend.rag_service import RAGService

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "backend" / "data"
FRONTEND_DIR = ROOT_DIR / "frontend"
load_dotenv(ROOT_DIR / ".env")
_gemini_model = None
_gemini_api_key = None
_query_cache = {}

app = FastAPI(title="Enterprise Knowledge Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_gemini_model():
    global _gemini_model, _gemini_api_key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or genai is None:
        return None

    if _gemini_model is not None and _gemini_api_key == api_key:
        return _gemini_model

    genai.configure(api_key=api_key)
    _gemini_api_key = api_key
    _gemini_model = genai.GenerativeModel("gemini-3.6-flash")
    return _gemini_model


rag_service = RAGService(DATA_DIR)


class QuestionRequest(BaseModel):
    question: str


@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "RAG backend is running"}


@app.post("/api/query")
def query_documents(payload: QuestionRequest):
    question = payload.question.strip()
    cache_key = question.casefold()
    if cache_key in _query_cache:
        return _query_cache[cache_key]

    matches = rag_service.retrieve(question)

    if matches and os.getenv("GOOGLE_API_KEY") and genai is not None:
        model = get_gemini_model()
        if model is not None:
            context = f"[{matches[0]['source']}]: {matches[0]['text']}"
            prompt = (
                "You are an enterprise policy assistant. Answer using ONLY the document context provided below. "
                "If the answer is not in the context, say that you could not find it in the available documents. "
                "Do not invent facts.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}"
            )
            try:
                response = model.generate_content(prompt)
                answer_text = getattr(response, "text", "").strip()
                if answer_text:
                    result = {
                        "answer": answer_text,
                        "sources": [{"file": item["source"], "score": round(float(item["score"] + item.get("intent_score", 0)), 3)} for item in matches],
                        "model": "gemini",
                    }
                    if len(_query_cache) >= 100:
                        _query_cache.pop(next(iter(_query_cache)))
                    _query_cache[cache_key] = result
                    return result
            except Exception:
                pass

    result = rag_service.answer(question)
    if len(_query_cache) >= 100:
        _query_cache.pop(next(iter(_query_cache)))
    _query_cache[cache_key] = result
    return result


def extract_text_from_pdf(file_path: Path) -> str:
    if PdfReader is None:
        raise HTTPException(status_code=500, detail="pypdf is not installed. Please install the PDF dependency.")

    reader = PdfReader(str(file_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in {".pdf", ".txt"}:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_path = DATA_DIR / file.filename
    content = await file.read()
    file_path.write_bytes(content)

    if file_ext == ".pdf":
        text = extract_text_from_pdf(file_path)
        parsed_txt = DATA_DIR / f"{file_path.stem}.txt"
        parsed_txt.write_text(text, encoding="utf-8")

    rag_service.load_documents()

    return {"message": f"Uploaded {file.filename} successfully", "status": "success"}


@app.get("/")
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", app=StaticFiles(directory=str(FRONTEND_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
