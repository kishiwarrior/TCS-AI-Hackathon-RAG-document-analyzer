import tempfile
import unittest
from pathlib import Path

from backend.rag_service import RAGService


class RAGServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = RAGService(Path("backend/data"))

    def test_how_many_casual_leaves_question_returns_direct_answer(self):
        result = self.service.answer("How many casual leaves do employees get per year?")
        self.assertIn("12 casual leaves per year", result["answer"])
        self.assertEqual(result["sources"][0]["file"], "hr_policy.txt")
        self.assertNotIn("I could not find a relevant answer", result["answer"])

    def test_internet_usage_question_returns_direct_policy(self):
        result = self.service.answer("Can I browse social media at work?")
        self.assertIn("must not access non-work-related websites", result["answer"].lower())
        self.assertEqual(result["sources"][0]["file"], "hr_policy.txt")
        self.assertNotIn("I could not find a relevant answer", result["answer"])

    def test_password_question_uses_security_guidelines_only(self):
        result = self.service.answer("What is the password policy?")
        self.assertEqual(result["sources"], [{"file": "IT_Security_Guidelines.txt", "score": result["sources"][0]["score"]}])
        self.assertIn("12 characters", result["answer"])

    def test_section_body_not_just_section_heading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            doc_path = Path(tmpdir) / "policy.txt"
            doc_path.write_text(
                "Section 2: Clean Desk and Device Locking\n"
                "A clean desk means keeping your workspace free of clutter, documents, and sensitive material. "
                "This reduces the risk of data exposure and helps maintain confidentiality.",
                encoding="utf-8",
            )
            service = RAGService(Path(tmpdir))
            result = service.answer("what does a clean desk mean?")
            self.assertIn("free of clutter", result["answer"].lower())
            self.assertNotIn("section 2: clean desk and device locking", result["answer"].lower())


if __name__ == "__main__":
    unittest.main()
