import unittest

from lazy_nlp_pipeline import NLP
from lazy_nlp_pipeline.words_analyzer import WordsAnalyzer, Word


class TestWords(unittest.TestCase):
    def setUp(self):
        self.nlp = NLP('test_project')
        self.nlp.add_doc_pipe(WordsAnalyzer())

    def test_basic(self):
        text = 'Кілька слів. Ще слова'
        doc = self.nlp(text)
        words = doc.words
        for w in words:
            self.assertIsInstance(w, Word)
        self.assertEqual(words[0].text, "Кілька".lower())
        for w in words:
            self.assertLessEqual(w.score, 1.0)
