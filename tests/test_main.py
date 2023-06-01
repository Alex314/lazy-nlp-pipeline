import unittest

from lazy_nlp_pipeline import NLP, Doc, Token, Word


class TestDefaultNLP(unittest.TestCase):
    def test_project_name(self):
        nlp = NLP('test_default_nlp')
        self.assertEqual(nlp.project_name, 'test_default_nlp')

    def test_nlp_call(self):
        nlp = NLP('test_default_nlp')
        text = 'test doc'
        doc = nlp(text)
        self.assertIsInstance(doc, Doc)
        self.assertEqual(doc.text, text)

    def test_tokenization(self):
        nlp = NLP('test_default_nlp')
        text = 'Test doc_1!'
        doc = nlp(text)
        for t in doc.tokens:
            self.assertIsInstance(t, Token)
        self.assertEqual([t.text for t in doc.tokens], ['Test', ' ', 'doc', '_', '1', '!'])
    
    def test_words(self):
        nlp = NLP('test_default_nlp')
        text = 'Кілька слів. Ще слова'
        doc = nlp(text)
        for w in doc.words:
            self.assertIsInstance(w, Word)
        self.assertEqual(doc.words[0].text, "Кілька".lower())
        for w in doc.words:
            self.assertLessEqual(w.score, 1.0)


if __name__ == '__main__':
    unittest.main()
