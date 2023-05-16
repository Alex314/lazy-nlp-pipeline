import unittest

from lazy_nlp_pipeline import NLP, Doc


class TestDefaultNLP(unittest.TestCase):
    def test_project_name(self):
        nlp = NLP('test_project')
        self.assertEqual(nlp.project_name, 'test_project')

    def test_nlp_call(self):
        nlp = NLP('test_project')
        text = 'test doc'
        doc = nlp(text)
        self.assertIsInstance(doc, Doc)
        self.assertEqual(doc.text, text)


if __name__ == '__main__':
    unittest.main()
