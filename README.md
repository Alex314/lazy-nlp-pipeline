# lazy_nlp_pipeline

NLP pipeline for rule-based pattern matching, heavily inspired by Spacy. This project have very limited subset of patterns available, so it is just a showcase of possibilities to improve over Spacy, not a viable ready-to-use replacement.

Main advantages of lazy_nlp_pipeline over Spacy are lazy evaluation and ability to create nested patterns.

Lazy evaluation allows to skip computationally expensive unnessessary steps while matching patterns. For example, in Spacy full pipeline is applied to all documents. lazy_nlp_pipeline apply steps when necessary. I.e. if there is a pattern which starts with some match of any token of length between X and Y and ends with a token of particular lemma, then there's no need to lemmatize all tokens in all documents - only those which weren't filtered out by a much quicker to check first part of a pattern.

Nested patterns provides a nice opportunity to create more sofisticated rules than mere list of all possible end-to-end sequences of token rules as in Spacy. It also helps to define some subpatterns in seperate variables and reuse them in different parts of different patterns.

See basic usage examples in BasicUsage.ipynb

See example and comparison with Spacy in Example.ipynb

## Key examples

- Basic example

```python
from lazy_nlp_pipeline import (NLP, Pattern as P,
                               TokenPattern as TP, WordPattern as WP,
                               )

nlp = NLP(project_name='simplest_token_patterns')

pattern = P(
    TP('1'),
    TP('a'),
)

test_texts = [
    '1 a',
    'Something 1 a something',
    'Something Something2',
]
for span in nlp.match_patterns([pattern], texts=test_texts):
    print(span)
# Span('1 a')[0:3 doc=140386624266448]
# Span('1 a')[10:13 doc=140386624411344]
```

- Date example

```python
from lazy_nlp_pipeline import NLP, Pattern as P, TokenPattern as TP

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
for span in nlp.match_patterns([pattern], texts=test_texts):
    print(span)
# Span('From 2001-01-10 to 2009-01-10')[0:29 doc=139710858833808]
# Span('2001-01-10 to 2009-01-10')[5:29 doc=139710858833808]
# Span('10-01-2001 to 2009-01-10')[10:34 doc=139710858245392]
```

- Russian lemmatization

```python
from lazy_nlp_pipeline import NLP, Pattern as P, TokenPattern as TP

nlp = NLP(project_name='simplest_token_patterns')

pattern = P(
    WP(lemma='общедоступный'),
    TP(isspace=False)[1:],
)

test_texts = [
    'Википедия (англ. Wikipedia) — общедоступная интернет-энциклопедия реализованная на принципах вики',
]
for span in nlp.match_patterns([pattern], texts=test_texts):
    print(span)
# Span('общедоступная интернет')[30:52 doc=139710859896592]
# Span('общедоступная интернет-')[30:53 doc=139710859896592]
# Span('общедоступная интернет-энциклопедия')[30:65 doc=139710859896592]
```
