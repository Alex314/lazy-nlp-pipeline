from __future__ import annotations
from collections.abc import Iterable
import dataclasses
from typing import TYPE_CHECKING

from pymorphy3 import MorphAnalyzer


if TYPE_CHECKING:
    from lazy_nlp_pipeline import Doc


class WordsAnalyzer:
    attributes_provided = ['words']

    def __init__(self):
        self.morph_uk = MorphAnalyzer(lang='uk')
        self.morph_ru = MorphAnalyzer(lang='ru')

    def get_attribute(self, doc: Doc, name: str):
        if name == 'words':
            return self.get_words(doc)

    def get_words(self, doc: Doc):
        ans = []
        for t in doc:
            words = self.words_from_token(t)
            words = Word.squeeze(words)
            ans.extend(words)
        return ans

    def words_from_token(self, token, prefix='', start_char=None):
        if start_char is None:
            start_char = token.start_char
        ans = []
        prefix += token.text.lower()
        for lang, morph in [('uk', self.morph_uk), ('ru', self.morph_ru)]:
            matched_prefixes = []
            for match in morph.iter_known_word_parses(prefix):
                if match.word == prefix:
                    if prefix in matched_prefixes:
                        continue
                    ans.extend([self.match_to_word(
                        m, start_char=start_char, lang=lang) for m in morph.parse(prefix)])
                    matched_prefixes.append(prefix)
                else:
                    if token.next_token is not None:
                        ans.extend(self.words_from_token(
                            token.next_token, prefix=prefix, start_char=start_char))
                    break
        return ans

    def match_to_word(self, match, start_char, lang=None):
        return Word(match.word, start_char, lemma=match.normal_form, pos=match.tag.POS, lang=lang, score=match.score)


@dataclasses.dataclass(frozen=True)
class Word:
    text: str
    start_char: int
    lemma: str | None = None
    pos: str | None = None
    lang: str | None = None
    score: float = dataclasses.field(default=1.0, compare=False)

    def __repr__(self) -> str:
        flags = [f'{self.start_char}:{self.start_char+len(self.text)}']
        if self.lemma is not None:
            flags.append(self.lemma)
        if self.pos is not None:
            flags.append(self.pos)
        if round(self.score, 5) != 1:
            flags.append(f'{self.score:.2g}')
        if self.lang is not None:
            flags.append(self.lang)
        ans = f'{type(self).__name__}({self.text})[{" ".join(flags)}]'
        return ans

    @classmethod
    def squeeze(cls, words: Iterable[Word], lemma=True, pos=True, lang=True):
        dct: dict[Word, Word] = {}
        for w in words:
            nw = cls(w.text, w.start_char,
                     lemma=w.lemma if lemma else None,
                     pos=w.pos if pos else None,
                     lang=w.lang if lang else None,
                     score=w.score,
                     )
            if nw in dct:
                new_score = min(1.0, dct[nw].score+nw.score)
                dct[nw] = dataclasses.replace(dct[nw], score=new_score)
            else:
                dct[nw] = nw
        return list(dct.values())
