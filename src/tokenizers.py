from __future__ import annotations
from abc import ABC, abstractmethod
import re
from typing import Optional


class BaseTokenizer(ABC):
    @abstractmethod
    def count(self, text: str) -> int:
        """Returns the number of token units in the text."""
        pass


class RegexProxyTokenizer(BaseTokenizer):
    """
    Exact proxy unit counter matching the thesis draft:
    counts sequences of alphanumeric characters or single punctuation symbols.
    Regex: r'\\w+|[^\\w\\s]'
    """
    def __init__(self, pattern: str = r"\w+|[^\w\s]"):
        self.regex = re.compile(pattern)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self.regex.findall(text))


class WhitespaceTokenizer(BaseTokenizer):
    """Simple whitespace-separated word counter."""
    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(text.strip().split())


class CharacterTokenizer(BaseTokenizer):
    """Character counter for raw character budgeting."""
    def count(self, text: str) -> int:
        return len(text) if text else 0


class TiktokenTokenizer(BaseTokenizer):
    """
    Standard BPE tokenizer using OpenAI's tiktoken for accurate token economics.
    Falls back gracefully to RegexProxyTokenizer if tiktoken is not installed.
    """
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self._encoder = None
        self._fallback = RegexProxyTokenizer()
        try:
            import tiktoken
            self._encoder = tiktoken.encoding_for_model(model_name)
        except Exception:
            self._encoder = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(text))
            except Exception:
                return self._fallback.count(text)
        return self._fallback.count(text)
