from abc import ABC, abstractmethod


class BaseTextExtractor(ABC):
    """Abstract base class for file text extractors."""

    @abstractmethod
    def extract(self, content_bytes: bytes, filename: str = "") -> str:
        """Extract text from file content bytes."""
        pass

