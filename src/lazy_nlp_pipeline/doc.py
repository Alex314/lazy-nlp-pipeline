from __future__ import annotations
from collections import defaultdict
from collections.abc import Mapping, Collection
from typing import TYPE_CHECKING, Any, overload

from spacy import displacy

from lazy_nlp_pipeline.base_classes import WithLazyAttributes


if TYPE_CHECKING:
    from lazy_nlp_pipeline.nlp import NLP
    from lazy_nlp_pipeline.tokenizer import Token


class Doc(WithLazyAttributes):
    def __init__(self,
                 nlp: NLP,
                 text: str,
                 ):
        super().__init__(nlp)
        self.text = text
        self.tags: defaultdict[str, list[Span]] = defaultdict(list)

    def render(self, tags: Collection[str] | None = None):
        spans_to_show = []
        breaking_chars = [0]
        # TODO: split_by_tokens: bool, if True, add boundaries of tokens
        for tag, tag_spans in self.tags.items():
            if tags is not None and tag not in tags:
                continue
            for s in tag_spans:
                spans_to_show.append((tag, s))
                breaking_chars.extend([s.start_char, s.end_char])
        breaking_chars.append(len(self.text))
        breaking_chars = sorted(set(breaking_chars))

        ch_to_idx = {}
        for i, bch in enumerate(breaking_chars):
            ch_to_idx[bch] = i

        spans = []
        for tag, s in spans_to_show:
            spans.append({"start_token": ch_to_idx[s.start_char],
                          "end_token": ch_to_idx[s.end_char],
                          "label": tag})

        tokens = [self.text[s:e]
                  for s, e in zip(breaking_chars[:-1], breaking_chars[1:])]

        return displacy.render({'text': self.text, 'spans': spans,
                                'tokens': tokens}, style='span', manual=True)

    @overload
    def __getitem__(self, key: int) -> DocPosition: ...

    @overload
    def __getitem__(self, key: slice) -> Span: ...

    def __getitem__(self, key: int | slice) -> DocPosition | Span:
        """Get character-level Span of this Doc"""
        match key:
            case int():
                return DocPosition(self, key)
            case slice(start=lower, stop=upper):
                if lower is None:
                    lower = 0
                if upper is None:
                    upper = len(self.text)
                return Span(doc=self, start_char=lower, end_char=upper)
        return NotImplemented

    def __repr__(self) -> str:
        flags = [self.nlp.project_name]
        flags.append(' '.join(self.lazy_attributes))
        text = repr(self.text)
        if len(self.text) > 100:
            text = repr(self.text[:97]) + '...'
        ans = f'{type(self).__name__}({text})[{" ".join(flags)}]'
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
            for t in self.doc.tokens:
                if t.start_char == self.position:
                    self._token_ahead = t
                    return t
        return self._token_ahead

    @property
    def token_behind(self) -> Token | None:
        """Return token which ends by the previous character"""
        if self._token_behind is None:
            for t in self.doc.tokens:
                if t.end_char == self.position:
                    self._token_behind = t
                    return t
        return self._token_behind

    def to_span(self) -> Span:
        """Create empty Span on current position"""
        return Span(self.doc, self.position, self.position)

    def __repr__(self) -> str:
        flags = [
            f'{self.doc.text[:self.position][-4:]!r}_{self.doc.text[self.position:][:4]!r}',
            f'{self.position}',
            f'doc_id={id(self.doc)}',
        ]
        ans = f'{type(self).__name__}[{" ".join(flags)}]'
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
        self.attributes: dict[str, Any] = {}
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

    def __eq__(self, other) -> bool:
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
