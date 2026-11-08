import re
import unicodedata


class Preprocessor:
    """Cleans and normalizes raw extracted text."""

    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return ""

        # Normalize unicode (NFKC replaces compatibility characters)
        text = unicodedata.normalize("NFKC", text)

        # Replace non-breaking spaces and irregular spaces with standard space
        text = text.replace("\u00a0", " ").replace("\u200b", "")

        # Normalize carriage returns
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Replace 3 or more consecutive newlines with two
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Replace tabs and multiple horizontal spaces with single space, except on line breaks
        lines = []
        for line in text.split("\n"):
            cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
            lines.append(cleaned_line)

        return "\n".join(lines).strip()

