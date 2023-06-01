from __future__ import annotations
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from lazy_nlp_pipeline.nlp import NLP


class WithLazyAttributes:
    def __init__(self, nlp: NLP):
        self.nlp = nlp
        self.lazy_attributes: dict[str, Any] = {}

    def __getattr__(self, name: str):
        if name in self.lazy_attributes:
            return self.lazy_attributes[name]
        self.nlp.eval_lazy_attribute(self, name)
        return self.lazy_attributes[name]
    
    # Required to be able to pickle objects with custom __getattr__
    # which is needed to use multiprocessing
    def __setstate__(self, data):
        self.__dict__.update(data)
