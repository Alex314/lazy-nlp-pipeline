from __future__ import annotations
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from pymorphy3 import MorphAnalyzer
from pymorphy3.units.by_shape import PunctuationAnalyzer
from pymorphy3.units.unkn import UnknAnalyzer
from lazy_nlp_pipeline.base_classes import WithLazyAttributes

from lazy_nlp_pipeline.doc import Doc
from lazy_nlp_pipeline.tokenizer import Token


if TYPE_CHECKING:
    from lazy_nlp_pipeline.nlp import NLP


class Word(WithLazyAttributes):
    def __init__(self,
                 nlp: NLP,
                 text: str,
                 doc: Doc,
                 start_char: int,
                 end_char: int,
                 normalized_text: str | None = None,
                 ):
        super().__init__(nlp)
        self.text = text
        self.doc = doc
        self.start_char = start_char
        self.end_char = end_char
        if normalized_text is None:
            normalized_text = text
        self.normalized_text = normalized_text
        self.is_capitalized = self.text[:1].isupper()

    def __repr__(self) -> str:
        flags = [
            f'{self.start_char}:{self.end_char}',
        ]
        if self.normalized_text != self.text:
            flags.append(f'norm={self.normalized_text!r}')
        if self.lazy_attributes.get('is_known') is False:
            flags.append('unknown')
        # show value of str attributes
        for k, v in self.lazy_attributes.items():
            if isinstance(v, str):
                flags.append(v)
        ans = f'{type(self).__name__}({self.text!r})[{" ".join(flags)}]'
        return ans


class WordsAnalyzer:
    targets: list[tuple[type[WithLazyAttributes], str]] = [
        (Doc, 'words'),
        (Token, 'words_starting_here'),
        (Token, 'words_ending_here'),
    ]
    DEFAULT_WORD_CHARS = {
        'uk': "'-.абвгдежзийклмнопрстуфхцчшщьюяєіїґ",
        'ru': "'-.0123456789абвгдежзийклмнопрстуфхцчшщъыьэюяё’",
    }
    DEFAULT_FIRST_CHARS = {
        'uk': "абвгдежзийклмнопрстуфхцчшщюяєіїґ",
        'ru': "абвгдежзийклмнопрстуфхцчшщыэюяё",
    }
    DEFAULT_LAST_CHARS = {
        'uk': ".абвгдежзийклмнопрстуфхцчшщьюяєіїґ",
        'ru': "абвгдежзийклмнопрстуфхцчшщъыьэюяё",
    }
    DEFAULT_REPLACEMENTS: dict[str, dict[str, str]] = {
        'uk': {
            '`': "'",
            '́': '',
        },
        'ru': {},
    }

    def __init__(self,
                 lang: str = 'uk',
                 word_chars: str | None = None,
                 first_chars: str | None = None,
                 last_chars: str | None = None,
                 char_replacements: Mapping[str, str] | None = None,
                 exclude_analyzers=(PunctuationAnalyzer,
                                    UnknAnalyzer),
                 ):
        self.lang = lang
        if word_chars is None:
            word_chars = self.DEFAULT_WORD_CHARS[lang]
        self.word_chars = word_chars
        if first_chars is None:
            first_chars = self.DEFAULT_FIRST_CHARS[lang]
        self.first_chars = first_chars
        if last_chars is None:
            last_chars = self.DEFAULT_LAST_CHARS[lang]
        self.last_chars = last_chars
        if char_replacements is None:
            char_replacements = self.DEFAULT_REPLACEMENTS[lang]
        self.char_replacements = char_replacements
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
            case Doc() as doc, 'words':
                self.eval_words(doc)
            case Token() as t, 'words_starting_here':
                self.eval_words_forward(t)
            case Token() as t, 'words_ending_here':
                self.eval_words_backward(t)

    def eval_words_forward(self, t: Token, prefix: str = '',
                           start_token: Token | None = None, start_char: int | None = None):
        if start_token is None:
            start_token = t
        if 'words_starting_here' not in start_token.lazy_attributes:
            start_token.lazy_attributes['words_starting_here'] = []
        if start_char is None:
            start_char = t.start_char
        prefix += t.text
        for k, v in self.char_replacements.items():
            prefix = prefix.replace(k, v)
        if any(ch not in self.word_chars for ch in prefix.lower()) or not prefix:
            return
        if prefix[:1].lower() not in self.first_chars:
            return
        if prefix[-1:].lower() in self.last_chars:
            words = self.parse_words(
                nlp=t.nlp, text=prefix, doc=t.doc,
                start_char=start_token.start_char, end_char=t.end_char,
            )
            start_token.lazy_attributes['words_starting_here'].extend(words)
            start_token.lazy_attributes['words_starting_here'] = squeeze_words(
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
        if end_char is None:
            end_char = t.end_char
        suffix = t.text + suffix
        for k, v in self.char_replacements.items():
            suffix = suffix.replace(k, v)
        if any(ch not in self.word_chars for ch in suffix.lower()) or not suffix:
            return
        if suffix[-1:].lower() not in self.last_chars:
            return
        if suffix[:1].lower() in self.first_chars:
            words = self.parse_words(
                nlp=t.nlp, text=suffix, doc=t.doc,
                start_char=t.start_char, end_char=end_token.end_char,
            )
            end_token.lazy_attributes['words_ending_here'].extend(words)
            end_token.lazy_attributes['words_ending_here'] = squeeze_words(
                end_token.lazy_attributes['words_ending_here'])
        if t.previous_token is None:
            return
        self.eval_words_backward(t.previous_token, suffix=suffix,
                                 end_token=end_token, end_char=end_char)

    def eval_words(self, doc: Doc):
        if 'words' not in doc.lazy_attributes:
            doc.lazy_attributes['words'] = []
        for t in doc.tokens:
            doc.lazy_attributes['words'].extend(t.words_starting_here)
        doc.lazy_attributes['words'] = squeeze_words(doc.lazy_attributes['words'])

    def parse_words(self, nlp: NLP, text: str, doc: Doc,
                    start_char: int, end_char: int):
        parses = self.morph.parse(text)
        words = []
        for p in parses:
            w = Word(
                nlp=nlp, text=doc.text[start_char:end_char], doc=doc,
                start_char=start_char, end_char=end_char,
                normalized_text=text,
            )
            w.lazy_attributes['lang'] = self.lang

            w.lazy_attributes['is_known'] = p.is_known
            w.lazy_attributes['lemma'] = p.normal_form
            w.lazy_attributes['score'] = p.score

            w.lazy_attributes['animacy'] = p.tag.animacy
            w.lazy_attributes['aspect'] = p.tag.aspect
            w.lazy_attributes['case_plus'] = p.tag.case
            w.lazy_attributes['case'] = p.tag.case
            if p.tag.case in p.tag.RARE_CASES:
                w.lazy_attributes['case'] = p.tag.RARE_CASES[p.tag.case]
            w.lazy_attributes['gender'] = p.tag.gender
            w.lazy_attributes['involvement'] = p.tag.involvement
            w.lazy_attributes['mood'] = p.tag.mood
            w.lazy_attributes['number'] = p.tag.number
            w.lazy_attributes['pos'] = p.tag.POS
            w.lazy_attributes['person'] = p.tag.person
            w.lazy_attributes['tense'] = p.tag.tense
            w.lazy_attributes['transitivity'] = p.tag.transitivity
            w.lazy_attributes['voice'] = p.tag.voice

            w.lazy_attributes['role'] = None
            for gramemma in p.tag.grammemes:
                if gramemma in ['Name', 'Surn', 'Patr']:
                    w.lazy_attributes['role'] = gramemma
            words.append(w)
        return words


def squeeze_words(words: Iterable[Word],
                  keep_only: Iterable[str] | None = None,
                  ignore_only: Iterable[str] | None = None,
                  ) -> list[Word]:
    """Keep only words with unique attributes

    By default filter words using all attributes.
    If keep_only is specified - use only attributes with given names
    If ignore_only is specified - use all but given attributes
    """
    if keep_only is not None and ignore_only is not None:
        raise ValueError('At least one of `keep_only` `ignore_only` should be None')
    state_to_word = {}
    for w in words:
        state = (
            ('doc', w.doc),
            ('text', w.text),
            ('start_char', w.start_char),
            ('end_char', w.end_char),
            ('normalized_text', w.normalized_text),
            *sorted([(k, v) for k, v in w.lazy_attributes.items()
                     if not k in ['score']])
        )
        if keep_only is not None:
            state = tuple((k, v) for (k, v) in state if k in keep_only)
        if ignore_only is not None:
            state = tuple((k, v) for (k, v) in state if k not in ignore_only)
        if state in state_to_word:
            continue
        # TODO: replace ignored attributes with None
        state_to_word[state] = w
    return list(state_to_word.values())
