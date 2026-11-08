import io
from app.ingestion.extractors.base import BaseTextExtractor
from app.utils.logger import logger


class PdfExtractor(BaseTextExtractor):
    """Extracts text from PDF files using PyMuPDF (fitz) or pypdf fallback."""

    def extract(self, content_bytes: bytes, filename: str = "") -> str:
        # Try PyMuPDF (fitz) first
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=content_bytes, filetype="pdf")
            text_parts = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                if page_text:
                    text_parts.append(page_text)
            doc.close()
            extracted = "\n\n".join(text_parts).strip()
            if extracted:
                return extracted
        except Exception as e:
            logger.warning(f"PyMuPDF failed on {filename}: {e}. Falling back to pypdf.")

        # Fallback to pypdf
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content_bytes))
            text_parts = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
            return "\n\n".join(text_parts).strip()
        except Exception as e:
            logger.error(f"pypdf also failed on {filename}: {e}")
            raise ValueError(f"Could not extract text from PDF '{filename}': {e}")

