from typing import List, Optional
from app.core.models import Chunk


class RecursiveCharacterChunker:
    """Splits text recursively using a hierarchy of separators to preserve semantic cohesion."""

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using the given list of separators."""
        final_chunks: List[str] = []

        if not text:
            return final_chunks

        # Pick the first separator present in text
        separator = separators[-1]
        new_separators: List[str] = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        # Split on the chosen separator
        splits = text.split(separator) if separator else list(text)

        # Now merge splits respecting chunk_size and chunk_overlap
        good_splits: List[str] = []
        for s in splits:
            if s:
                good_splits.append(s)

        current_doc: List[str] = []
        current_len = 0

        for s in good_splits:
            item_len = len(s) + (len(separator) if current_doc else 0)

            # If a single piece is longer than chunk_size, recursively split it with finer separators
            if len(s) > self.chunk_size:
                if current_doc:
                    doc_str = separator.join(current_doc)
                    final_chunks.append(doc_str)
                    current_doc = []
                    current_len = 0
                if new_separators:
                    sub_chunks = self._split_text(s, new_separators)
                    final_chunks.extend(sub_chunks)
                else:
                    # Character slice fallback
                    for start in range(0, len(s), self.chunk_size - self.chunk_overlap):
                        final_chunks.append(s[start:start + self.chunk_size])
                continue

            if current_len + item_len <= self.chunk_size:
                current_doc.append(s)
                current_len += item_len
            else:
                if current_doc:
                    doc_str = separator.join(current_doc)
                    final_chunks.append(doc_str)

                    # Build overlap window
                    overlap_doc: List[str] = []
                    overlap_len = 0
                    for prev_item in reversed(current_doc):
                        added_len = len(prev_item) + (len(separator) if overlap_doc else 0)
                        if overlap_len + added_len <= self.chunk_overlap:
                            overlap_doc.insert(0, prev_item)
                            overlap_len += added_len
                        else:
                            break
                    current_doc = overlap_doc
                    current_len = overlap_len

                current_doc.append(s)
                current_len += len(s) + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            final_chunks.append(separator.join(current_doc))

        return [c.strip() for c in final_chunks if c.strip()]

    def chunk(self, text: str, doc_id: str, source_file: str = "") -> List[Chunk]:
        """Produces a list of structured Chunk objects with character offsets."""
        raw_chunks = self._split_text(text, self.separators)
        chunks: List[Chunk] = []

        current_search_idx = 0
        for i, chunk_text in enumerate(raw_chunks):
            # Find char offset in source text
            pos = text.find(chunk_text, current_search_idx)
            if pos == -1:
                pos = text.find(chunk_text)
            char_start = pos if pos != -1 else 0
            char_end = char_start + len(chunk_text)
            current_search_idx = max(0, char_start + 1)

            chunks.append(
                Chunk(
                    id=f"{doc_id}_{i}",
                    doc_id=doc_id,
                    text=chunk_text,
                    source_file=source_file,
                    chunk_index=i,
                    char_start=char_start,
                    char_end=char_end,
                    metadata={"source_file": source_file, "chunk_index": i}
                )
            )

        return chunks

