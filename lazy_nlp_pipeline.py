from __future__ import annotations
from collections.abc import Sequence
import logging
import re

import pymorphy3
from tqdm import tqdm


class NLP:
    def __init__(self,
                 project_name: str = 'default_project',
                ):
        self.project_name = project_name
        self.tokenizer = Tokenizer()
        self._lemmatizer = None
    
    @property
    def lemmatizer(self):
        """Hardcoded Russian pymorphy3 for now"""
        if self._lemmatizer is None:
            self._lemmatizer = pymorphy3.MorphAnalyzer()
        return self._lemmatizer
    
    def tokenize(self, doc:Doc) -> None:
        """Set tokens of given doc with current tokenizer"""
        self.tokenizer.tokenize(doc)

    def lemmatize(self, token:Token) -> None:
        """Set lemma of given token.
        
        Works only with pymorphy-like lemmatizers for now"""
        token._lemma = self.lemmatizer.parse(token.text)[0].normal_form
    
    def match_patterns(self,
                       patterns: Iterable,
                       texts: Iterable[Doc|str],
                       backward: bool = False,  # TODO: rewrite
                      ) -> Generator[Span, None, None]:
        """Yield all different occurences of each pattern in each text
        
        If text is str - Doc will be created from that string.
        if `backward` - will match patterns in reverse (last subpattern first).
            That could speed-up parsing if last subpatterns take less time
            to filter out non-matching sequences.
        """
        for doc in tqdm(texts, total=len(texts), desc='match_patterns'):
            if isinstance(doc, str):
                doc = self(doc)
            for p in patterns:
                yield from p.match(doc, forward = not backward)
    
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

    def __getitem__(self, key) -> Token|Span:
        """Get token by index or Span by slice
        
        If key is an int - it will be treated as token-level index
        If key is a slice - it will be treated as character-level slice of text
        """
        if self.tokens  is None:
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
        if self.tokens  is None:
            self.tokens = self.nlp.tokenize(self.text)
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
    def token_ahead(self) -> Token|None:
        """Return token which starts from the next character"""
        if self._token_ahead is None:
            for t in self.doc:
                if t.start_char == self.position:
                    self._token_ahead = t
                    return t
        return self._token_ahead

    @property
    def token_behind(self) -> Token|None:
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
                 start_char:int,
                 end_char:int,
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
                 attributes: Mapping|None = None,
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
            raise ValueError(f"To add two Span's they should be from the same Doc")
        if self.end_char != other.start_char:
            raise ValueError(f"To add two Span's they should follow each other"
                             f" Got {self} and {other}")
        return Span(self.doc, self.start_char, other.end_char,
                    attributes=self.attributes|other.attributes)

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
            flags.append('{' + ', '.join(f'{k}: {v!r}' for k, v in self.attributes.items()) + '}')
        ans = f'{type(self).__name__}({self.text!r})[{" ".join(flags)}]'
        return ans

class Tokenizer:
    """Hardcoded regexp-based tokenizer"""
    def __init__(self):
        self.token_regex = re.compile(r'''(?x)
        (\w(?<![\d_]))+  # word
        |\d+             # number
        |[.,?!:"']       # punctuation
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
            t = Token(text=doc.text[start:end], doc=doc, start_char=start, end_char=end)
            if len(tokens) > 0:
                tokens[-1].next_token = t
                t.previous_token = tokens[-1]
            tokens.append(t)
        doc.tokens = tokens

class RepeatedPattern:
    def __init__(self,
                 subpattern: TokenPattern|Pattern|OrPattern,
                 arity_min: int|None,
                 arity_max: int|None,
                ):
        self.subpattern = subpattern
        if arity_min is None:
            arity_min = 0
        self.arity_min = arity_min
        self.arity_max = arity_max

    def match_from(self, doc_position:DocPosition,
                   forward: bool = True,
                   previously_matched = None,
                   min_matches: int|None = None,
                   max_matches: int|None = None,
                  ) -> Generator[tuple[Span, DocPosition], None, None]:
        """Match pattern starting from doc_position either forward or backward"""
        if min_matches is None:
            min_matches = self.arity_min
            max_matches = self.arity_max
        if previously_matched is None:
            previously_matched = doc_position.to_span()
        for matched_sp, next_position in self.subpattern.match_from(doc_position, forward=forward):
            # print(f'{self.subpattern=} {matched_sp=}, {next_position=}')
            if forward:
                matched = previously_matched + matched_sp
            else:
                matched = matched_sp + previously_matched
            
            if min_matches <= 1:  # Should yield current match
                yield matched, next_position

            local_min_matches = max(1, min_matches-1)
            if max_matches is None:
                yield from self.match_from(next_position, forward, matched,
                                           local_min_matches, max_matches)
            elif max_matches > 1:
                yield from self.match_from(next_position, forward, matched,
                                           local_min_matches, max_matches-1)
        if min_matches == 0:
            yield previously_matched, doc_position

class RepeatablePattern:
    def __getitem__(self, arity):
        """Repeat pattern from X to Y times, including X and Y"""
        match arity:
            case int():
                return RepeatedPattern(self, arity, arity)
            case slice(start=lower, stop=upper):
                if lower is None:
                    lower = 0
                return RepeatedPattern(self, lower, upper)
            case _:
                raise TypeError(f'Pattern quantifier should be int or slice. Got {arity!r}')
        return self
    

class TokenPattern(RepeatablePattern):
    def __init__(self,
                 text: str|None = None,
                 lemma: str|None = None,

                 ignore_case: bool = False,

                 isalpha: bool|None = None,
                 isnumeric: bool|None = None,
                 isspace: bool|None = None,

                 min_len: int|None = None,
                 max_len: int|None = None,
                ):
        self.to_check = []

        self.text = text
        if text is not None:
            self.to_check.append('text')

        self.lemma = lemma
        if lemma is not None:
            self.to_check.append('lemma')

        self.ignore_case = ignore_case

        self.isalpha = isalpha
        if isalpha is not None:
            self.to_check.append('isalpha')
        self.isnumeric = isnumeric
        if isnumeric is not None:
            self.to_check.append('isnumeric')
        self.isspace = isspace
        if isspace is not None:
            self.to_check.append('isspace')

        self.min_len = min_len
        if min_len is not None:
            self.to_check.append('min_len')
        self.max_len = max_len
        if max_len is not None:
            self.to_check.append('max_len')

    def match_from(self, doc_position:DocPosition,
                   forward: bool = True,
                  ) -> Generator[tuple[Span, DocPosition], None, None]:
        """Match pattern starting from doc_position either forward or backward"""
        if forward:
            token = doc_position.token_ahead
        else:
            token = doc_position.token_behind
        if token is None:
            return
        for rule in self.to_check:
            match rule:
                case 'text':
                    pattern_text = self.text
                    token_text = token.text
                    if self.ignore_case:
                        pattern_text = pattern_text.lower()
                        token_text = token_text.lower()
                    if token_text != pattern_text:
                        return
                case 'lemma':
                    if token.lemma != self.lemma:
                        return
                case 'isalpha':
                    if token.text.isalpha() != self.isalpha:
                        return
                case 'isnumeric':
                    if token.text.isnumeric() != self.isnumeric:
                        return
                case 'isspace':
                    if token.text.isspace() != self.isspace:
                        return
                case 'min_len':
                    if len(token.text) < self.min_len:
                        return
                case 'max_len':
                    if len(token.text) > self.max_len:
                        return
                case _:
                    raise Exception(f'Unexpected rule to check: {rule!r}')
        next_position = token.start_position
        if forward:
            next_position = token.end_position
        yield token.to_span(), next_position

    def __or__(self, other: TokenPattern|Pattern) -> OrPattern:
        if isinstance(other, (TokenPattern, Pattern)):
            return OrPattern(self, other)
        return NotImplemented

    def __repr__(self) -> str:
        flags = []
        if self.text is not None:
            flags.append(f'text={self.text!r}')
        if self.lemma is not None:
            flags.append(f'lemma={self.lemma!r}')
        if self.isalpha is not None:
            flags.append('isalpha' if self.isalpha else '~isalpha')
        if self.isnumeric is not None:
            flags.append('isnumeric' if self.isalpha else '~isnumeric')
        if self.isspace is not None:
            flags.append('isspace' if self.isalpha else '~isspace')
        if self.min_len is not None or self.max_len is not None:
            flags.append(f'len={self.min_len}:{self.max_len}')
        ans = f'{type(self).__name__}[{" ".join(flags)}]'
        return ans

class OrPattern(RepeatablePattern):
    def __init__(self, *subpatterns: Pattern|TokenPattern):
        self.subpatterns = subpatterns

    def match_from(self, doc_position:DocPosition,
                   forward: bool = True,
                  ) -> Generator[tuple[Span, DocPosition], None, None]:
        """Match pattern starting from doc_position either forward or backward"""
        for sp in self.subpatterns:
            yield from sp.match_from(doc_position, forward=forward)

    def __or__(self, other: TokenPattern|Pattern) -> OrPattern:
        if isinstance(other, (TokenPattern, Pattern)):  # TODO: just check that other is Matchable or something
            return OrPattern(*self.subpatterns, other)
        return NotImplemented

class Pattern(RepeatablePattern):
    def __init__(self, *subpatterns: Pattern|TokenPattern,
                 allow_inbetween: Pattern|str|None = 'SPACES',
                 # force_inbetween: bool = False, # TODO: force `allow_inbetween` to match exactly once
                 as_attribute: str|None = None,
                ):
        """Matches sequence of subpatterns with any quantity of `allow_inbetween` between them
        
        allow_inbetween:
            if None - no tokens are allowed between consequent subpatterns
            if pattern - match any quantity (pattern[:]) of such patterns between consequent subpatterns 
            if str: use pre-registered pattern from NLP object by that name
        as_attribute: save text of current pattern as attribute of that name in the matched Span
        """
        if len(subpatterns) == 0:
            raise ValueError(f'Should be at least one subpattern')
        self.subpatterns = subpatterns

        if isinstance(allow_inbetween, (Pattern, TokenPattern, OrPattern)):
            allow_inbetween = allow_inbetween[:]
        self.allow_inbetween = allow_inbetween
        self.as_attribute = as_attribute

    def match(self, doc: Doc, forward=True) -> Span:
        """Yield all matches within the Doc"""
        # TODO: check guards w\o tokenizing

        for token in doc:
            if token.start_char == 0:
                position = token.start_position
                for matched, next_token in self.match_from(position, forward=forward):
                    yield matched
            position = token.end_position
            for matched, next_token in self.match_from(position, forward=forward):
                yield matched

    def match_from(self, doc_position:DocPosition,
                   forward: bool = True,
                   subpatterns_to_match: Collection[DirectionMatchable]|None = None,
                  ) -> Generator[tuple[Span, DocPosition], None, None]:
        """Match pattern starting from doc_position either forward or backward"""
        if subpatterns_to_match is None:
            for matched, next_position in self.match_from(doc_position, forward=forward,
                                                          subpatterns_to_match=self.subpatterns):
                if self.as_attribute is not None:
                    matched.set_attribute(self.as_attribute, matched.text)
                yield matched, next_position
            return
        yielded = []
        sp = subpatterns_to_match[-1]
        if forward:
            sp = subpatterns_to_match[0]
        for matched, next_position in sp.match_from(doc_position, forward=forward):
            if len(subpatterns_to_match) == 1:  # Last subpattern
                if any(i == matched for i in yielded):  # check duplicates
                    continue
                yielded.append(matched)
                yield matched, next_position
                continue

            # match rest subpatterns
            stm = subpatterns_to_match[:-1]
            if forward:
                stm = subpatterns_to_match[1:]
            if isinstance(self.allow_inbetween, str):
                self.allow_inbetween = doc_position.doc.nlp.get_pattern(self.allow_inbetween)[:]
            if self.allow_inbetween is None:
                for matched_next, after_next_position in self.match_from(next_position, forward=forward,
                                                                         subpatterns_to_match=stm):
                    if forward:
                        span = matched + matched_next
                    else:
                        span = matched_next + matched
                    if any(i == span for i in yielded):  # check duplicates
                        continue
                    yielded.append(span)
                    yield span, after_next_position
            else:
                for matched_inbetween, after_inbetween_position in self.allow_inbetween.match_from(
                                                                    next_position, forward=forward):
                    for matched_next, after_next_position in self.match_from(after_inbetween_position,
                                                                             forward=forward,
                                                                             subpatterns_to_match=stm):
                        if forward:
                            span = matched + matched_inbetween + matched_next
                        else:
                            span = matched_next + matched_inbetween + matched
                        if len(matched_next) == 0 and len(matched_inbetween) != 0:
                            #  disallow inbetween matches when they aren't around non-empty match
                            continue
                        if len(matched) == 0 and len(matched_inbetween) != 0:
                            #  disallow inbetween matches when they aren't around non-empty match
                            continue
                        if any(i == span for i in yielded):  # check duplicates
                            continue
                        yielded.append(span)
                        yield span, after_next_position

    def __or__(self, other: TokenPattern|Pattern) -> OrPattern:
        if isinstance(other, (TokenPattern, Pattern)):
            return OrPattern(self, other)
        return NotImplemented
