import os
from typing import Dict, Type
from app.ingestion.extractors.base import BaseTextExtractor
from app.ingestion.extractors.pdf_extractor import PdfExtractor
from app.ingestion.extractors.docx_extractor import DocxExtractor
from app.ingestion.extractors.txt_extractor import TxtExtractor

_EXTRACTOR_REGISTRY: Dict[str, Type[BaseTextExtractor]] = {
    ".pdf": PdfExtractor,
    ".docx": DocxExtractor,
    ".txt": TxtExtractor,
    ".md": TxtExtractor,
    ".markdown": TxtExtractor,
    ".csv": TxtExtractor,
    ".json": TxtExtractor,
    ".log": TxtExtractor,
    ".py": TxtExtractor,
    ".js": TxtExtractor,
    ".ts": TxtExtractor,
    ".html": TxtExtractor,
}


def get_extractor(filename: str, content_type: str = "") -> BaseTextExtractor:
    """Factory to retrieve appropriate extractor based on extension or MIME type."""
    ext = os.path.splitext(filename.lower())[1]
    extractor_cls = _EXTRACTOR_REGISTRY.get(ext)

    if extractor_cls:
        return extractor_cls()

    # Fallback checking content_type
    if "pdf" in content_type.lower():
        return PdfExtractor()
    if "word" in content_type.lower() or "officedocument" in content_type.lower():
        return DocxExtractor()

    # Default to text extractor
    return TxtExtractor()


__all__ = ["BaseTextExtractor", "PdfExtractor", "DocxExtractor", "TxtExtractor", "get_extractor"]

