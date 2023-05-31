import unittest

from lazy_nlp_pipeline import NLP


class TestWords(unittest.TestCase):
    def setUp(self):
        self.nlp = NLP('test_words')
    
    def test_token_words(self):
        doc = self.nlp('Красиво як-не-як')
        for t in doc.tokens:
            with self.subTest(t=t):
                self.assertEqual(t.words_starting_here, [])
