# test_main.py

import unittest
from main import RAGPipeline

class TestRAGPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = RAGPipeline()
        cls.prompts = [
            "Are late payments penalized?",
            "When are customers billed?",
            "What is required for admin account login?",
        ]
        cls.results = cls.pipeline.run_batch(cls.prompts)

    def test_batch_size_matches_prompts(self):
        self.assertEqual(len(self.results), len(self.prompts))

    def test_result_structure(self):
        for result in self.results:
            for key in ["prompt", "input", "output", "reference", "label", "score", "explanation"]:
                self.assertIn(key, result)

    def test_label_validity(self):
        valid_labels = ["CORRECT", "INCORRECT", "PARTIALLY_CORRECT"]
        for result in self.results:
            self.assertIn(result["label"].upper(), valid_labels)

    def test_score_range(self):
        for result in self.results:
            self.assertIsInstance(result["score"], (int, float))
            self.assertGreaterEqual(result["score"], 0.0)
            self.assertLessEqual(result["score"], 1.0)

    def test_non_empty_output(self):
        for result in self.results:
            self.assertTrue(result["output"].strip())

if __name__ == "__main__":
    unittest.main()
