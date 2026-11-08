import io
from app.ingestion.extractors.base import BaseTextExtractor
from app.utils.logger import logger


class DocxExtractor(BaseTextExtractor):
    """Extracts text from DOCX files using python-docx."""

    def extract(self, content_bytes: bytes, filename: str = "") -> str:
        try:
            import docx
            doc = docx.Document(io.BytesIO(content_bytes))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        paragraphs.append(row_text)
            return "\n\n".join(paragraphs).strip()
        except Exception as e:
            logger.error(f"DocxExtractor failed on {filename}: {e}")
            raise ValueError(f"Could not extract text from DOCX '{filename}': {e}")

