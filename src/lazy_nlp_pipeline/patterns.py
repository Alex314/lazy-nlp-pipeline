from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence
from typing import TYPE_CHECKING

from lazy_nlp_pipeline.doc import DocPosition
from lazy_nlp_pipeline.words_analyzer import Word


if TYPE_CHECKING:
    from lazy_nlp_pipeline.doc import Doc, Span


class RepeatedPattern:
    def __init__(self,
                 subpattern: TokenPattern | Pattern | OrPattern,
                 arity_min: int | None,
                 arity_max: int | None,
                 ):
        self.subpattern = subpattern
        if arity_min is None:
            arity_min = 0
        self.arity_min = arity_min
        self.arity_max = arity_max

    def match_from(self, doc_position: DocPosition,
                   forward: bool = True,
                   previously_matched=None,
                   min_matches: int | None = None,
                   max_matches: int | None = None,
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
                raise TypeError(
                    f'Pattern quantifier should be int or slice. Got {arity!r}')
        return self


class OrPattern(RepeatablePattern):
    def __init__(self, *subpatterns: DisjunctivePattern):
        self.subpatterns = subpatterns

    def match_from(self, doc_position: DocPosition,
                   forward: bool = True,
                   ) -> Generator[tuple[Span, DocPosition], None, None]:
        """Match pattern starting from doc_position either forward or backward"""
        for sp in self.subpatterns:
            yield from sp.match_from(doc_position, forward=forward)

    def __or__(self, other: DisjunctivePattern) -> OrPattern:
        if isinstance(other, DisjunctivePattern):
            return OrPattern(*self.subpatterns, other)
        return NotImplemented


class DisjunctivePattern(ABC):
    def __or__(self, other: DisjunctivePattern) -> OrPattern:
        if isinstance(other, DisjunctivePattern):
            return OrPattern(self, other)
        return NotImplemented

    @abstractmethod
    def match_from(self, doc_position: DocPosition,
                   forward: bool = True,
                   ) -> Generator[tuple[Span, DocPosition], None, None]: ...


class TokenPattern(RepeatablePattern, DisjunctivePattern):
    def __init__(self,
                 text: str | None = None,

                 ignore_case: bool = False,

                 isalpha: bool | None = None,
                 isnumeric: bool | None = None,
                 isspace: bool | None = None,

                 min_len: int | None = None,
                 max_len: int | None = None,
                 ):
        self.to_check = []

        self.text = text
        if text is not None:
            self.to_check.append('text')

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

    def match_from(self, doc_position: DocPosition,
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
                        pattern_text = pattern_text.lower()  # type: ignore
                        token_text = token_text.lower()
                    if token_text != pattern_text:
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
                    if len(token.text) < self.min_len:  # type: ignore
                        return
                case 'max_len':
                    if len(token.text) > self.max_len:  # type: ignore
                        return
                case _:
                    raise Exception(f'Unexpected rule to check: {rule!r}')
        next_position = token.start_position
        if forward:
            next_position = token.end_position
        yield token.to_span(), next_position

    def __repr__(self) -> str:
        flags = []
        if self.text is not None:
            flags.append(f'text={self.text!r}')
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


class WordPattern(RepeatablePattern, DisjunctivePattern):
    def __init__(self,
                 text: str | None = None,
                 ignore_case: bool = False,
                 lemma: str | None = None,
                 pos: str | None = None,
                 lang: str | None = None,
                 ):
        self.to_check = []

        self.text = text
        if text is not None:
            self.to_check.append('text')
        self.ignore_case = ignore_case

        self.lemma = lemma
        if lemma is not None:
            self.to_check.append('lemma')
        self.pos = pos
        if pos is not None:
            self.to_check.append('pos')
        self.lang = lang
        if lang is not None:
            self.to_check.append('lang')

    def match_from(self, doc_position: DocPosition,
                   forward: bool = True,
                   ) -> Generator[tuple[Span, DocPosition], None, None]:
        """Match pattern starting from doc_position either forward or backward"""
        if forward:
            token = doc_position.token_ahead
            if token is None:
                return
            words = token.words_starting_here
        else:
            token = doc_position.token_behind
            if token is None:
                return
            words = token.words_ending_here
        for w in words:
            if self.pass_guards(w):
                next_position = doc_position.doc[w.start_char]
                if forward:
                    next_position = doc_position.doc[w.end_char]
                yield doc_position.doc[w.start_char:w.end_char], next_position

    def pass_guards(self, word: Word) -> bool:
        for rule in self.to_check:
            match rule:
                case 'text':
                    pattern_text = self.text
                    word_text = word.text
                    if self.ignore_case:
                        pattern_text = pattern_text.lower()  # type: ignore
                        word_text = word_text.lower()
                    if word_text != pattern_text:
                        return False
                case 'lemma':
                    if word.lemma != self.lemma:
                        return False
                case 'pos':
                    if word.pos != self.pos:
                        return False
                case 'lang':
                    if word.lang != self.lang:
                        return False
                case _:
                    raise Exception(f'Unexpected rule to check: {rule!r}')
        return True


class Pattern(RepeatablePattern, DisjunctivePattern):
    def __init__(self, *subpatterns: Pattern | TokenPattern,
                 allow_inbetween: Pattern | str | None = 'SPACES',
                 # force_inbetween: bool = False, # TODO: force `allow_inbetween` to match exactly once
                 as_attribute: str | None = None,
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

    def match(self, doc: Doc, forward=True) -> Generator[Span, None, None]:
        """Yield all matches within the Doc"""
        # TODO: check guards w\o tokenizing

        for token in doc.tokens:
            if token.start_char == 0:
                position = token.start_position
                for matched, next_token in self.match_from(position, forward=forward):
                    yield matched
            position = token.end_position
            for matched, next_token in self.match_from(position, forward=forward):
                yield matched

    def match_from(self, doc_position: DocPosition,
                   forward: bool = True,
                   subpatterns_to_match: Sequence | None = None,
                   ) -> Generator[tuple[Span, DocPosition], None, None]:
        """Match pattern starting from doc_position either forward or backward"""
        if subpatterns_to_match is None:
            for matched, next_position in self.match_from(doc_position, forward=forward,
                                                          subpatterns_to_match=self.subpatterns):
                if self.as_attribute is not None:
                    matched.set_attribute(self.as_attribute, matched.text)
                yield matched, next_position
            return
        yielded: list[Span] = []
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
                self.allow_inbetween = doc_position.doc.nlp.get_pattern(self.allow_inbetween)[
                    :]
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
