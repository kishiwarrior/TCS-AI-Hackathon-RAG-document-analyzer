from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RAGService:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.documents: List[Dict[str, str]] = []
        self.facts: List[Dict[str, str]] = []
        self.vectorizer = None
        self.matrix = None
        self.fact_vectorizer = None
        self.fact_matrix = None
        self.load_documents()

    def _split_into_sections(self, text: str) -> List[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        blocks: List[str] = []
        i = 0

        while i < len(lines):
            line = lines[i]
            is_heading = bool(re.match(r"^(?:Section|Chapter)\s*\d*\s*[:.-]?\s*[A-Za-z0-9 &/-]+$", line, re.I))

            if is_heading:
                blocks.append(line)
                i += 1
                body_lines = []
                while i < len(lines) and not re.match(r"^(?:Section|Chapter)\s*\d*\s*[:.-]?\s*[A-Za-z0-9 &/-]+$", lines[i], re.I):
                    body_lines.append(lines[i])
                    i += 1
                if body_lines:
                    blocks.append(" ".join(body_lines))
                continue

            body_lines = [line]
            i += 1
            while i < len(lines) and not re.match(r"^(?:Section|Chapter)\s*\d*\s*[:.-]?\s*[A-Za-z0-9 &/-]+$", lines[i], re.I):
                body_lines.append(lines[i])
                i += 1
            blocks.append(" ".join(body_lines))

        return blocks

    def load_documents(self):
        self.documents = []
        self.facts = []
        if not self.data_dir.exists():
            return

        for file_path in sorted(self.data_dir.glob("*.txt")):
            text = file_path.read_text(encoding="utf-8")
            for chunk in self._chunk_text(text, chunk_size=250, overlap=60):
                self.documents.append({
                    "source": file_path.name,
                    "text": chunk.strip(),
                })

            for fact in self._extract_fact_units(text):
                self.facts.append({"source": file_path.name, **fact})

        if not self.documents:
            self.vectorizer = None
            self.matrix = None
            return

        all_text = [doc["text"] for doc in self.documents]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(all_text)

        fact_text = [fact["text"] for fact in self.facts]
        if fact_text:
            self.fact_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self.fact_matrix = self.fact_vectorizer.fit_transform(fact_text)

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
        chunks: List[str] = []
        current = ""

        for paragraph in paragraphs:
            if not paragraph:
                continue
            if len(current) + len(paragraph) <= chunk_size:
                current = (current + " " + paragraph).strip()
            else:
                if current:
                    chunks.append(current)
                current = paragraph

        if current:
            chunks.append(current)

        final_chunks: List[str] = []
        for idx, chunk in enumerate(chunks):
            final_chunks.append(chunk)
            if idx < len(chunks) - 1 and overlap > 0:
                overlap_text = " ".join(chunks[idx + 1].split()[:max(1, overlap // 4)])
                if overlap_text:
                    final_chunks[-1] = f"{chunk} {overlap_text}".strip()

        return final_chunks

    def _extract_heading_and_body(self, section: str):
        lines = [line.strip() for line in section.splitlines() if line.strip()]
        if not lines:
            return None, section

        first = lines[0]
        if not re.match(r"^\s*(?:Section|Chapter)\s*\d*\s*[:.-]?\s*[A-Za-z0-9 &/-]+", first, re.I):
            return None, section

        heading = first
        body = " ".join(lines[1:]) if len(lines) > 1 else ""
        return heading, body

    def _normalize(self, text: str) -> str:
        text = text.lower()
        text = text.replace("’", "'")
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_fact_units(self, text: str) -> List[Dict[str, str]]:
        """Create small evidence units so common words cannot make whole files compete."""
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []

        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
        heading_pattern = re.compile(r"^(?:Section|Chapter)\s*\d*\s*[:.-]?\s*[A-Za-z0-9 &/()\-]+$", re.I)
        heading_indexes = [index for index, line in enumerate(lines) if heading_pattern.match(line)]
        if heading_indexes:
            units = []
            for position, heading_index in enumerate(heading_indexes):
                heading = lines[heading_index]
                body_end = heading_indexes[position + 1] if position + 1 < len(heading_indexes) else len(lines)
                body = " ".join(lines[heading_index + 1:body_end]).strip()
                units.append({"text": heading, "answer": heading, "kind": "heading"})
                if body:
                    units.append({
                        "text": f"{heading}. {body}",
                        "answer": body,
                        "kind": "body",
                    })
            return units

        numbered_units = [unit.strip() for unit in re.split(r"(?=\b\d+\.\s+)", normalized) if unit.strip()]
        if len(numbered_units) > 1:
            return [{"text": unit, "answer": unit, "kind": "body"} for unit in numbered_units]

        return [{"text": normalized, "answer": normalized, "kind": "body"}]

    def _keyword_match_score(self, question: str, text: str) -> int:
        question_terms = set(self._normalize(question).split())
        text_terms = set(self._normalize(text).split())
        ignored = {"what", "which", "where", "when", "does", "do", "can", "is", "are", "the", "a", "an", "at", "to", "of", "for", "in", "on", "how", "many", "per", "year"}
        question_terms -= ignored
        overlap = len(question_terms & text_terms)

        concept_words = {
            "leave": {"leave", "leaves", "casual", "medical", "days"},
            "travel": {"travel", "reimbursement", "claim", "claims", "expense", "submitted", "days"},
            "internet": {"internet", "website", "websites", "social", "media", "browse", "browsing", "office", "hours"},
            "attendance": {"attendance", "mark", "checkin", "check", "daily", "before"},
            "password": {"password", "passwords", "authentication", "mfa", "characters", "updated"},
            "exam": {"exam", "topics", "module", "industrial", "management", "pcc", "cs502"},
        }
        expanded = set(question_terms)
        question_text = self._normalize(question)
        for concept, words in concept_words.items():
            if concept in question_text or question_terms & words:
                expanded.update(words)

        return len(expanded & text_terms) + overlap

    def _intent_rank(self, question: str, text: str, kind: str = "body"):
        q = self._normalize(question)
        text_norm = self._normalize(text)

        if re.search(r"which section|section.*(is|does)|in which section|what section", q):
            if kind == "heading":
                return 25
            return -30

        if re.search(r"what does|what is|meaning|means|importance|why|importance of", q):
            if kind == "body":
                if len(text.split()) > 10:
                    return 20
                if re.search(r"\bmeans\b|\bimportance\b|\bhelps\b|\breduces\b|\bexposure\b|\bconfidentiality\b", text_norm):
                    return 18
            return -20

        return 0

    def retrieve(self, question: str):
        if not question or not question.strip():
            return []

        if not self.documents or self.vectorizer is None or self.matrix is None:
            return []

        query_vector = self.fact_vectorizer.transform([question]) if self.fact_vectorizer is not None else None
        semantic_scores = cosine_similarity(query_vector, self.fact_matrix).flatten() if query_vector is not None else []
        ranked = []
        for index, fact in enumerate(self.facts):
            keyword_score = self._keyword_match_score(question, fact["text"])
            semantic_score = float(semantic_scores[index]) if len(semantic_scores) else 0.0
            intent_score = self._intent_rank(question, fact["text"], fact.get("kind", "body"))
            total_score = keyword_score * 2 + semantic_score * 4 + intent_score
            if total_score > 0:
                ranked.append({
                    "source": fact["source"],
                    "text": fact.get("answer", fact["text"]),
                    "score": total_score,
                    "kind": fact.get("kind", "body"),
                    "intent_score": 0,
                })

        if not ranked:
            return []

        # Pick a winning file first, then return only its best evidence units.
        source_scores = {}
        for match in ranked:
            source_scores[match["source"]] = max(source_scores.get(match["source"], 0), match["score"])
        winning_source = max(source_scores, key=source_scores.get)
        winning_matches = [match for match in ranked if match["source"] == winning_source]
        winning_matches.sort(key=lambda item: item["score"], reverse=True)
        best_score = winning_matches[0]["score"]
        if best_score < 2:
            return []
        return winning_matches[:1]

    def answer(self, question: str) -> Dict[str, object]:
        if not question or not question.strip():
            return {
                "answer": "Please enter a valid question about the uploaded documents.",
                "sources": [],
            }

        if not self.documents or self.vectorizer is None or self.matrix is None:
            return {
                "answer": "No document data is available yet. Please upload a text file first.",
                "sources": [],
            }

        matches = self.retrieve(question)
        if not matches:
            return {
                "answer": "I could not find a relevant answer in the available documents.",
                "sources": [],
            }

        best_match = matches[0]
        answer_text = self._generate_answer(question, best_match["text"])

        return {
            "answer": answer_text,
            "sources": [
                {
                    "file": match["source"],
                    "score": round(float(match["score"] + match["intent_score"]), 3),
                }
                for match in matches
            ],
        }

    def _generate_answer(self, question: str, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if len(cleaned) > 350:
            cleaned = cleaned[:350].rsplit(" ", 1)[0] + "..."

        if re.search(r"casual|leave|leaves", self._normalize(question)) and "Casual Leave" in cleaned:
            return cleaned
        if re.search(r"travel|reimbursement|claim", self._normalize(question)) and "Travel Reimbursement" in cleaned:
            return cleaned
        if re.search(r"internet|website|social|media|browse", self._normalize(question)) and "Internet Usage" in cleaned:
            return cleaned
        if re.search(r"attendance|time", self._normalize(question)) and "Attendance" in cleaned:
            return cleaned
        return f"Based on the available documents, {cleaned}"
