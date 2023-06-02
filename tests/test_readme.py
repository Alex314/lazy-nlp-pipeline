import unittest

from lazy_nlp_pipeline import (NLP, Pattern as P,
                               TokenPattern as TP, WordPattern as WP,
                               )


class TestReadme(unittest.TestCase):
    """Test all examples from README.md so that they are up-to-date"""

    def test_basic_example(self):
        nlp = NLP(project_name='test_readme_examples')

        pattern = P(
            TP('1'),
            TP('a'),
        )

        test_texts = [
            '1 a',
            'Something 1 a something',
            'Something Something2',
        ]

        results = []
        for span in nlp.match_patterns([pattern], texts=test_texts):
            results.append((span.text, span.start_char, span.end_char))
            # print(span)
        # Span('1 a')[0:3 doc=140386624266448]
        # Span('1 a')[10:13 doc=140386624411344]

        expected_results = [
            ('1 a', 0, 3),
            ('1 a', 10, 13),
        ]
        self.assertEqual(results, expected_results)

    def test_date_example(self):
        nlp = NLP(project_name='simplest_token_patterns')

        ymd_date = P(
            TP(isnumeric=True, min_len=4, max_len=4),
            TP('-'),
            TP(isnumeric=True, min_len=2, max_len=2),
            TP('-'),
            TP(isnumeric=True, min_len=2, max_len=2),

            allow_inbetween=None,
        )
        dmy_date = P(
            TP(isnumeric=True, min_len=2, max_len=2),
            TP('-'),
            TP(isnumeric=True, min_len=2, max_len=2),
            TP('-'),
            TP(isnumeric=True, min_len=4, max_len=4),

            allow_inbetween=None,
        )

        pattern = P(
            TP('from', ignore_case=True)[0:1],
            ymd_date | dmy_date,
            TP('to', ignore_case=True),
            ymd_date | dmy_date,
        )

        test_texts = [
            'From 2001-01-10 to 2009-01-10',
            'Something 10-01-2001 to 2009-01-10 Something2',
        ]

        results = []
        for span in nlp.match_patterns([pattern], texts=test_texts):
            results.append((span.text, span.start_char, span.end_char))
            # print(span)
        # Span('From 2001-01-10 to 2009-01-10')[0:29 doc=139710858833808]
        # Span('2001-01-10 to 2009-01-10')[5:29 doc=139710858833808]
        # Span('10-01-2001 to 2009-01-10')[10:34 doc=139710858245392]

        expected_results = [
            ('From 2001-01-10 to 2009-01-10', 0, 29),
            ('2001-01-10 to 2009-01-10', 5, 29),
            ('10-01-2001 to 2009-01-10', 10, 34),
        ]
        self.assertEqual(results, expected_results)

    def test_russian_lemmatization(self):
        nlp = NLP(project_name='simplest_token_patterns')

        pattern = P(
            WP(lemma='общедоступный'),
            TP(isspace=False)[1:],
        )

        test_texts = [
            'Википедия (англ. Wikipedia) — общедоступная интернет-энциклопедия реализованная на принципах вики',
        ]

        results = []
        for span in nlp.match_patterns([pattern], texts=test_texts):
            results.append((span.text, span.start_char, span.end_char))
            # print(span)
        # Span('общедоступная интернет')[30:52 doc=139710859896592]
        # Span('общедоступная интернет-')[30:53 doc=139710859896592]
        # Span('общедоступная интернет-энциклопедия')[30:65 doc=139710859896592]

        expected_results = [
            ('общедоступная интернет', 30, 52),
            ('общедоступная интернет-', 30, 53),
            ('общедоступная интернет-энциклопедия', 30, 65),
        ]
        self.assertEqual(results, expected_results)
