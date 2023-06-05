import unittest

from lazy_nlp_pipeline import (NLP, Pattern as P, TokenPattern as TP, WordPattern as WP)


class TestPatterns(unittest.TestCase):
    def setUp(self):
        self.nlp = NLP('test_project')

    def assert_pattern_cases(self, pattern, cases):
        for backward in [False, True]:
            for s, expected in cases:
                with self.subTest(backward=backward, source_str=s, expected_matches=expected):
                    matches = self.nlp.match_patterns(
                        patterns=[pattern], texts=[s], backward=backward)
                    str_matches = [m.text for m in matches]
                    self.assertCountEqual(str_matches, expected)

    def test_token_sequence(self):
        pattern = P(
            TP('1'),
            TP('a'),
        )

        src_match_pairs = [
            ('1a', ['1a']),
            ('something ...', []),
            ('smth 1a smth2', ['1a']),
        ]

        self.assert_pattern_cases(pattern, src_match_pairs)

    def test_subpattern(self):
        p1 = P(
            TP('1'),
            TP('a'),
        )

        pattern = P(
            p1,
            TP(':'),
            p1,
        )

        src_match_pairs = [
            ('1a:1a', ['1a:1a']),
            ('something 1a:1...a:1a', []),
        ]

        self.assert_pattern_cases(pattern, src_match_pairs)

    def test_allow_inbetween(self):
        pattern = P(
            TP('1'),
            TP('0'),
            TP('0'),

            allow_inbetween=TP(':')[1:],
        )

        src_match_pairs = [
            ('1:0:0', ['1:0:0']),
            ('Something 1:::::0:0', ['1:::::0:0']),
            ('Something 1:::::0:a::0', []),
        ]

        self.assert_pattern_cases(pattern, src_match_pairs)

    def test_allow_inbetween_none(self):
        pattern = P(
            TP('e'),
            TP('2'),
            TP(' '),
            TP('e'),
            TP('4'),

            allow_inbetween=None,
        )

        src_match_pairs = [
            ('e2 e4', ['e2 e4']),
            ('e 2 e 4', []),
        ]

        self.assert_pattern_cases(pattern, src_match_pairs)

    def test_date_example(self):
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

        src_match_pairs = [
            ('Something 10-01-2001 to 2009-01-10 Something2',
             ['10-01-2001 to 2009-01-10']),
            ('1999-01-10', []),
            ('From 2001-01-10 to 2009-01-10',
             ['2001-01-10 to 2009-01-10', 'From 2001-01-10 to 2009-01-10'])
        ]

        self.assert_pattern_cases(pattern, src_match_pairs)

    def test_uk_lemma(self):
        pattern = P(
            WP(lemma='неприбутковий', lang='uk'),
            WP(pos='NOUN', lang='uk'),
        )

        src_match_pairs = [
            ('Вікіпе́дія (англ. Wikipedia) — загальнодоступна неприбуткова багатомовна онлайн-енциклопедія, '
             'якою опікується неприбуткова організація «Фонд Вікімедіа».',
             ['неприбуткова організація']),
        ]

        self.assert_pattern_cases(pattern, src_match_pairs)

    def test_ru_lemma(self):
        pattern = P(
            WP(lemma='общедоступный', lang='ru'),
            TP(isspace=False)[1:],
        )

        src_match_pairs = [
            ('Википедия (англ. Wikipedia) — общедоступная интернет-энциклопедия реализованная на принципах вики',
             ['общедоступная интернет-энциклопедия', 'общедоступная интернет-', 'общедоступная интернет']),
        ]

        self.assert_pattern_cases(pattern, src_match_pairs)


if __name__ == '__main__':
    unittest.main()
