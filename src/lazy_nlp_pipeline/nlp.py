from collections import defaultdict
from functools import partial
from typing import Any, Callable, Collection, Generator, Iterable

from lazy_nlp_pipeline.doc import Doc, Span
from lazy_nlp_pipeline.patterns import TokenPattern
from lazy_nlp_pipeline.tokenizer import Tokenizer
from lazy_nlp_pipeline.words_analyzer import WordsAnalyzer


class NLP:
    DEFAULT_ANALYZERS = (
        Tokenizer(),
        WordsAnalyzer(lang='uk'),
        WordsAnalyzer(lang='ru'),
    )

    def __init__(self,
                 project_name: str = 'default_project',
                 analyzers: Iterable = DEFAULT_ANALYZERS,
                 ):
        self.project_name = project_name
        analyzers_dict_factory: Callable[[], defaultdict] = partial(defaultdict, list)
        self.analyzers: defaultdict[type, defaultdict[str, list[Any]]  # TODO: list[Analyzer]
                                ] = defaultdict(analyzers_dict_factory)
        for a in analyzers:
            self.add_analyzer(a)

    def add_analyzer(self, analyzer) -> None:
        """Saves factories for custom attributes of document-related classes"""
        for target_class, target_attribute in analyzer.targets:
            self.analyzers[target_class][target_attribute].append(analyzer)

    def eval_lazy_attribute(self, target, attribute_name: str) -> None:
        is_attr_evaluated = False
        for target_class, attr_factories in self.analyzers.items():
            if not isinstance(target, target_class):
                continue
            for analyzer in attr_factories[attribute_name]:
                analyzer.eval_attribute(target, attribute_name)
                is_attr_evaluated = True
        if not is_attr_evaluated:
            raise AttributeError(f'{self} have no analyzer for attribute '
                                 f'{attribute_name!r} of class {target.__class__.__name__}')
    
    def read_lines(self, fpath) -> Generator[Doc, None, None]:
        """Read file, make Doc from every line"""
        with open(fpath) as f:
            for L in f:
                yield self(L.removesuffix('\n'))

    def match_patterns(self,
                       patterns: Iterable,
                       texts: Collection[Doc | str],
                       backward: bool = False,  # TODO: rewrite match_patterns
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
            return TokenPattern(isspace=True)[:]

    def __call__(self, text: str) -> Doc:
        """Create Doc from str"""
        return Doc(self, text)

    def __repr__(self) -> str:
        flags: list[str] = []
        flags_repr = f'[{" ".join(flags)}]' if flags else ''
        ans = f'{type(self).__name__}({self.project_name!r}){flags_repr}'
        return ans
