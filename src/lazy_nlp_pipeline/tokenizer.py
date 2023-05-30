from __future__ import annotations
import logging
import re

from lazy_nlp_pipeline.doc import Doc, DocPosition, Span


class Tokenizer:
    targets = [
        (Doc, 'tokens'),
    ]

    def __init__(self):
        self.token_regex = re.compile(r'''(?x)
        (\w(?<![\d_]))+  # word
        |\d+             # number
        |[.,?!:"']       # punctuation
        |\s+             # space
        ''')

    def eval_attribute(self, target: Doc, name: str) -> None:
        if name == 'tokens':
            self.eval_tokens(target)

    def eval_tokens(self, doc: Doc) -> None:
        """Set doc.tokens by splitting doc.text"""
        if 'tokens' in doc.lazy_attributes:
            raise Exception(f'Tokenizer dont work with already tokenized docs')
        doc.lazy_attributes['tokens'] = []

        if len(doc.text) == 0:
            logging.warning('Got empty doc.text')
            return

        token_boundaries = [0]
        for match in self.token_regex.finditer(doc.text):
            start, end = match.span()
            if start != token_boundaries[-1]:
                token_boundaries.append(start)
            if end != token_boundaries[-1]:
                token_boundaries.append(end)
        if len(doc.text) != token_boundaries[-1]:
            token_boundaries.append(len(doc.text))

        tokens: list[Token] = []
        for start, end in zip(token_boundaries[:-1], token_boundaries[1:]):
            t = Token(text=doc.text[start:end], doc=doc,
                      start_char=start, end_char=end)
            if len(tokens) > 0:
                tokens[-1].next_token = t
                t.previous_token = tokens[-1]
            tokens.append(t)
        doc.lazy_attributes['tokens'] = tokens


class Token:
    def __init__(self,
                 text: str,
                 doc: Doc,
                 start_char: int,
                 end_char: int,
                 ):
        self.text = text
        self.doc = doc
        self.start_char = start_char
        self.end_char = end_char
        self.next_token: Token | None = None
        self.previous_token: Token | None = None

    @property
    def start_position(self) -> DocPosition:
        return DocPosition(self.doc, self.start_char)

    @property
    def end_position(self) -> DocPosition:
        return DocPosition(self.doc, self.end_char)

    def to_span(self) -> Span:
        return Span(self.doc, self.start_char, self.end_char)

    def __repr__(self) -> str:
        flags = [
            f'{self.start_char}:{self.end_char}',
        ]
        ans = f'{type(self).__name__}({self.text!r})[{" ".join(flags)}]'
        return ans
