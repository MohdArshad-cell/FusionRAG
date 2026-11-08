from app.ingestion.extractors.base import BaseTextExtractor
from app.utils.logger import logger


class TxtExtractor(BaseTextExtractor):
    """Extracts text from UTF-8, Latin-1, or ASCII text, markdown, CSV, or code files."""

    def extract(self, content_bytes: bytes, filename: str = "") -> str:
        for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                return content_bytes.decode(encoding).strip()
            except (UnicodeDecodeError, LookupError):
                continue
        logger.error(f"Failed to decode text file: {filename}")
        raise ValueError(f"Could not decode text file '{filename}' with supported encodings.")

