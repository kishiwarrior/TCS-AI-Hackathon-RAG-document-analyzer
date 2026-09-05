# Enterprise Knowledge Assistant (RAG Demo)

This project is a simple RAG-style Q&A app for enterprise knowledge documents.

## Project plan

1. Create a lightweight Python backend with FastAPI.
2. Add a document ingestion layer that reads text-based policy files.
3. Build a retrieval layer using TF-IDF similarity search.
4. Create a simple web frontend for chat and source display.
5. Verify the app runs locally on the machine.

## Tech stack

- Python 3.10+
- FastAPI
- Uvicorn
- scikit-learn
- HTML/CSS/JavaScript

## Project structure

- `backend/app.py` – API server and app routes
- `backend/rag_service.py` – retrieval and answer generation logic
- `backend/data/` – sample enterprise documents
- `frontend/` – chat UI

## Run locally

```bash
cd "c:/Users/ankit/Downloads/MY PROJECTS/ai project tcs hackathon"
python -m pip install -r requirements.txt
copy .env.example .env
# Edit .env and set GOOGLE_API_KEY to your Gemini API key
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

The `.env` file is ignored by Git and must never be committed. The backend loads
`GOOGLE_API_KEY` from `.env` automatically. If it is missing, the local retrieval
fallback remains available.

Then open:

- http://127.0.0.1:8000

## Upload to GitHub

From the project folder:

```bash
git init
git add .
git commit -m "Initial enterprise knowledge assistant"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

The real `.env` file is excluded from Git. Upload `.env.example` instead and
create a local `.env` file after cloning the repository.

If `GOOGLE_API_KEY` is not set, the app uses the local retrieval fallback instead of the Gemini model.

## Example questions

- How many casual leaves do employees get per year?
- What is the reimbursement policy for travel?
- Are there restrictions on internet usage?

## Goal

This is a clean demo version of the business prompt shown in the attached design: a document-grounded assistant that answers using retrieved evidence and cites the source.
