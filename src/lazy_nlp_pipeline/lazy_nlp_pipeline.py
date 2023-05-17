from __future__ import annotations
from collections.abc import Sequence
import logging
import re
from typing import Collection, Generator, Iterable, Mapping

import pymorphy3
from tqdm import tqdm

from lazy_nlp_pipeline.patterns import TokenPattern


class NLP:
    def __init__(self,
                 project_name: str = 'default_project',
                 ):
        self.project_name = project_name
        self.tokenizer = Tokenizer()
        self._lemmatizer = None
        self.doc_pipes = {}

    def add_doc_pipe(self, pipe):
        for attr in pipe.attributes_provided:
            if attr in self.doc_pipes:
                logging.warning(
                    f'Attribute {attr!r} already defined in {self}')
            self.doc_pipes[attr] = pipe

    @property
    def lemmatizer(self):
        """Hardcoded Russian pymorphy3 for now"""
        if self._lemmatizer is None:
            self._lemmatizer = pymorphy3.MorphAnalyzer()
        return self._lemmatizer

    def tokenize(self, doc: Doc) -> None:
        """Set tokens of given doc with current tokenizer"""
        self.tokenizer.tokenize(doc)

    def lemmatize(self, token: Token) -> None:
        """Set lemma of given token.

        Works only with pymorphy-like lemmatizers for now"""
        token._lemma = self.lemmatizer.parse(token.text)[0].normal_form

    def match_patterns(self,
                       patterns: Iterable,
                       texts: Collection[Doc | str],
                       backward: bool = False,  # TODO: rewrite
                       ) -> Generator[Span, None, None]:
        """Yield all different occurences of each pattern in each text

        If text is str - Doc will be created from that string.
        if `backward` - will match patterns in reverse (last subpattern first).
            That could speed-up parsing if last subpatterns take less time
            to filter out non-matching sequences.
        """
        # for doc in tqdm(texts, total=len(texts), desc='match_patterns'):
        for doc in texts:
            if isinstance(doc, str):
                doc = self(doc)
            for p in patterns:
                yield from p.match(doc, forward=not backward)

    def get_pattern(self, pattern_name: str):
        """Get pre-registered pattern by name

        The idea is to have registered patterns, including default ones
        (like SPACES, NUMBER, DATE, etc.) so that user don't have to define them every time.
        So far only "SPACES" is hardcoded, to use as default `allow_inbetween` in Pattern class.
        """
        if pattern_name == 'SPACES':
            return TokenPattern(isspace=True)

    def __call__(self, text: str) -> Doc:
        """Create Doc from str"""
        return Doc(text, self)


class Doc(Sequence):
    def __init__(self,
                 text: str,
                 nlp: NLP,
                 ):
        self.text = text
        self.nlp = nlp
        self.tokens = None
        self.attrs = {}  # For custom attributes

    def __getattr__(self, name):
        # logging.info(f'Doc.__getattr__({name!r}) called')
        if name in self.attrs:
            return self.attrs[name]
        if name in self.nlp.doc_pipes:
            value = self.nlp.doc_pipes[name].get_attribute(self, name)
            self.attrs[name] = value
            return value
        raise AttributeError

    def __getitem__(self, key) -> Token | Span:
        """Get token by index or Span by slice

        If key is an int - it will be treated as token-level index
        If key is a slice - it will be treated as character-level slice of text
        """
        if self.tokens is None:
            self.nlp.tokenize(self)
        match key:
            case int():
                return self.tokens[key]
            case slice(start=lower, stop=upper):
                if lower is None:
                    lower = 0
                if upper is None:
                    upper = -1
                else:
                    upper -= 1
                return Span(doc=self, start_char=self[lower].start_char, end_char=self[upper].end_char)
        return NotImplemented

    def __len__(self) -> int:
        """Number of tokens"""
        if self.tokens is None:
            self.nlp.tokenize(self)
        return len(self.tokens)

    def __repr__(self) -> str:
        flags = [self.nlp.project_name]
        if self.tokens is not None:
            flags.append('tokenized')
        ans = f'{type(self).__name__}({self.text!r})[{" ".join(flags)}]'
        return ans


class DocPosition:
    """Character-level position in the Doc"""

    def __init__(self, doc: Doc, position: int):
        self.doc = doc
        self.position = position
        self._token_ahead = None
        self._token_behind = None

    @property
    def token_ahead(self) -> Token | None:
        """Return token which starts from the next character"""
        if self._token_ahead is None:
            for t in self.doc:
                if t.start_char == self.position:
                    self._token_ahead = t
                    return t
        return self._token_ahead

    @property
    def token_behind(self) -> Token | None:
        """Return token which ends by the previous character"""
        if self._token_behind is None:
            for t in self.doc:
                if t.end_char == self.position:
                    self._token_behind = t
                    return t
        return self._token_behind

    def to_span(self) -> Span:
        """Create empty Span on current position"""
        return Span(self.doc, self.position, self.position)

    def __repr__(self) -> str:
        flags = [
            f'{self.doc.text[:self.position][-4:]!r}|{self.doc.text[self.position:][:4]!r}',
            f'{self.position}',
            f'doc_id={id(self.doc)}',
        ]
        ans = f'{type(self).__name__}[{" ".join(flags)}]'
        return ans


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
        self.next_token = None
        self.previous_token = None
        self._lemma = None

    @property
    def start_position(self) -> DocPosition:
        return DocPosition(self.doc, self.start_char)

    @property
    def end_position(self) -> DocPosition:
        return DocPosition(self.doc, self.end_char)

    @property
    def lemma(self) -> str:
        if self._lemma is None:
            self.doc.nlp.lemmatize(self)
        return self._lemma

    def to_span(self) -> Span:
        return Span(self.doc, self.start_char, self.end_char)

    def __repr__(self) -> str:
        flags = [
            f'{self.start_char}:{self.end_char}',
        ]
        if self._lemma is not None:
            flags.append('lemmatized')
        ans = f'{type(self).__name__}({self.text!r})[{" ".join(flags)}]'
        return ans


class Span:
    def __init__(self,
                 doc: Doc,
                 start_char: int,
                 end_char: int,
                 attributes: Mapping | None = None,
                 ):
        self.doc = doc
        self.start_char = start_char
        self.end_char = end_char
        self.text = self.doc.text[start_char:end_char]
        self.attributes = {}
        if attributes is not None:
            self.attributes |= attributes

    def set_attribute(self, attribute_name: str, value) -> None:
        self.attributes[attribute_name] = value

    def __len__(self) -> int:
        return self.end_char - self.start_char

    def __add__(self, other: Span) -> Span:
        """Concatenate two Spans if they exactly follow each other

        Attributes are merged, first Span have priority
        """
        if not isinstance(other, Span):
            return NotImplemented
        if self.doc is not other.doc:
            raise ValueError(
                f"To add two Span's they should be from the same Doc")
        if self.end_char != other.start_char:
            raise ValueError(f"To add two Span's they should follow each other"
                             f" Got {self} and {other}")
        return Span(self.doc, self.start_char, other.end_char,
                    attributes=self.attributes | other.attributes)

    def __eq__(self, other: Span) -> bool:
        if isinstance(other, Span):
            if self.doc is not other.doc:
                return False
            if self.start_char != other.start_char:
                return False
            if self.end_char != other.end_char:
                return False
            if self.attributes != other.attributes:
                return False
            return True
        return NotImplemented

    def __repr__(self) -> str:
        flags = [
            f'{self.start_char}:{self.end_char}',
            f'doc={id(self.doc)}',
        ]
        if len(self.attributes) > 0:
            flags.append('{' + ', '.join(f'{k}: {v!r}' for k,
                         v in self.attributes.items()) + '}')
        ans = f'{type(self).__name__}({self.text!r})[{" ".join(flags)}]'
        return ans


class Tokenizer:
    """Hardcoded regexp-based tokenizer"""

    def __init__(self):
        self.token_regex = re.compile(r'''(?x)
        (\w(?<![\d_]))+  # word
        |\d+             # number
        |[.,?!:"']       # punctuation
        |\s+             # space
        ''')

    def tokenize(self, doc: Doc) -> None:
        """Set tokens attribute of doc by splitting doc.text"""
        if len(doc.text) == 0:
            doc.tokens = []
            logging.warning('Got empty Doc text')
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

        tokens = []
        for start, end in zip(token_boundaries[:-1], token_boundaries[1:]):
            t = Token(text=doc.text[start:end], doc=doc,
                      start_char=start, end_char=end)
            if len(tokens) > 0:
                tokens[-1].next_token = t
                t.previous_token = tokens[-1]
            tokens.append(t)
        doc.tokens = tokens
