import unittest

from lazy_nlp_pipeline import NLP, Word


class TestWords(unittest.TestCase):
    def setUp(self):
        self.nlp = NLP('test_words')
    
    def test_token_words(self):
        doc = self.nlp('Красиво як-не-як')
        for t in doc.tokens:
            for w in t.words_starting_here:
                with self.subTest(token=t, word=w):
                    self.assertIsInstance(w, Word)
