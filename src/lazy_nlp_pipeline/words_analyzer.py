from __future__ import annotations
from collections.abc import Iterable
import dataclasses

from pymorphy3 import MorphAnalyzer
from pymorphy3.units.by_shape import PunctuationAnalyzer
from pymorphy3.units.by_analogy import KnownSuffixAnalyzer, UnknownPrefixAnalyzer
from pymorphy3.units.unkn import UnknAnalyzer
from lazy_nlp_pipeline.base_classes import WithLazyAttributes

from lazy_nlp_pipeline.doc import Doc
from lazy_nlp_pipeline.tokenizer import Token


class WordsAnalyzer:
    targets: list[tuple[type[WithLazyAttributes], str]] = [
        (Doc, 'words'),
        (Token, 'words_starting_here'),
        (Token, 'words_ending_here'),
        (Token, 'words'),
    ]
    DEFAULT_WORD_CHARS = {
        'uk': "'-.абвгдежзийклмнопрстуфхцчшщьюяєіїґ",
        'ru': "'-.0123456789абвгдежзийклмнопрстуфхцчшщъыьэюяё’"
    }

    def __init__(self,
                 lang: str = 'uk',
                 word_chars: str | None = None,
                 exclude_analyzers=(PunctuationAnalyzer, KnownSuffixAnalyzer, UnknownPrefixAnalyzer, UnknAnalyzer),
                 ):
        self.lang = lang
        if word_chars is None:
            word_chars = self.DEFAULT_WORD_CHARS[lang]
        self.word_chars = word_chars
        units = [
            u if not isinstance(u, list)
            else [su for su in u if not isinstance(su, exclude_analyzers)]
            for u in MorphAnalyzer(lang=lang)._config_value('DEFAULT_UNITS', MorphAnalyzer.DEFAULT_UNITS)
            if not isinstance(u, exclude_analyzers)
        ]
        units = [u for u in units if not u == []]
        self.morph = MorphAnalyzer(lang=lang, units=units)

    def eval_attribute(self, target: Doc | Token, name: str) -> None:
        match target, name:
            # TODO: remove `ignore` when mypy will support tuples in `case`
            # https://github.com/python/mypy/issues/12364
            case Doc as doc, 'words':
                self.eval_words(doc)  # type: ignore
            case Token as t, 'words_starting_here':
                self.eval_words_forward(t)  # type: ignore
            case Token as t, 'words_ending_here':
                self.eval_words_backward(t)  # type: ignore
            case Token as t, 'words':
                self.eval_token_words(t)  # type: ignore

    def eval_words_forward(self, t: Token, prefix: str = '',
                           start_token: Token | None = None, start_char: int | None = None):
        if start_token is None:
            start_token = t
        if 'words_starting_here' not in start_token.lazy_attributes:
            start_token.lazy_attributes['words_starting_here'] = []
        if any(ch not in self.word_chars for ch in t.text.lower()):
            return
        if start_char is None:
            start_char = t.start_char
        prefix += t.text
        words = [self.match_to_word(m, start_char, t.end_char) for m in self.morph.parse(prefix)]
        start_token.lazy_attributes['words_starting_here'].extend(words)
        start_token.lazy_attributes['words_starting_here'] = Word.squeeze(
            start_token.lazy_attributes['words_starting_here'])
        if t.next_token is None:
            return
        self.eval_words_forward(t.next_token, prefix=prefix,
                                start_token=start_token, start_char=start_char)

    def eval_words_backward(self, t: Token, suffix: str = '',
                            end_token: Token | None = None, end_char: int | None = None):
        if end_token is None:
            end_token = t
        if 'words_ending_here' not in end_token.lazy_attributes:
            end_token.lazy_attributes['words_ending_here'] = []
        if any(ch not in self.word_chars for ch in t.text.lower()):
            return
        if end_char is None:
            end_char = t.end_char
        suffix = t.text
        words = [self.match_to_word(m, t.start_char, end_char) for m in self.morph.parse(suffix)]
        end_token.lazy_attributes['words_ending_here'].extend(words)
        end_token.lazy_attributes['words_ending_here'] = Word.squeeze(
            end_token.lazy_attributes['words_ending_here'])
        if t.previous_token is None:
            return
        self.eval_words_backward(t.previous_token, suffix=suffix,
                                 end_token=end_token, end_char=end_char)

    def eval_token_words(self, t: Token):
        raise NotImplementedError

    def eval_words(self, doc: Doc):
        if 'words' not in doc.lazy_attributes:
            doc.lazy_attributes['words'] = []
        for t in doc.tokens:
            doc.lazy_attributes['words'].extend(t.words_starting_here)
        doc.lazy_attributes['words'] = Word.squeeze(doc.lazy_attributes['words'])

    def match_to_word(self, match, start_char: int, end_char: int):
        return Word(match.word, start_char, end_char,
                    lemma=match.normal_form, pos=str(match.tag.POS),
                    lang=self.lang, score=match.score)


@dataclasses.dataclass(frozen=True)
class Word:
    text: str
    start_char: int
    end_char: int
    lemma: str | None = None
    pos: str | None = None
    lang: str | None = None
    score: float = dataclasses.field(default=1.0, compare=False)

    def __repr__(self) -> str:
        flags = [f'{self.start_char}:{self.end_char}']
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
            nw = cls(w.text, w.start_char, w.end_char,
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
