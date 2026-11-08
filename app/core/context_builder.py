from typing import List
from app.core.models import ScoredChunk
from app.utils.logger import logger


class ContextBuilder:
    """
    Constructs the prompt context from retrieved chunks, applying the
    'Lost in the Middle' reordering algorithm to optimize LLM attention.
    """

    @staticmethod
    def reorder_lost_in_the_middle(chunks: List[ScoredChunk]) -> List[ScoredChunk]:
        """
        Reorders chunks so that the most relevant documents are placed at the
        extremes (beginning and end) of the prompt context.
        
        Input sorted by rank: [1, 2, 3, 4, 5]
        Output reordered:     [1, 3, 5, 4, 2]
        """
        if len(chunks) <= 2:
            return chunks

        reordered: List[ScoredChunk] = []
        top_half: List[ScoredChunk] = []
        bottom_half: List[ScoredChunk] = []

        for idx, chunk in enumerate(chunks):
            if idx % 2 == 0:
                top_half.append(chunk)
            else:
                bottom_half.append(chunk)

        # Reverse bottom half so second best document is at the very end
        bottom_half.reverse()
        reordered = top_half + bottom_half

        logger.debug(
            f"Lost-in-the-middle reordering: original ranks {[c.rank for c in chunks]} "
            f"-> new ranks {[c.rank for c in reordered]}"
        )
        return reordered

    @classmethod
    def build_context_string(cls, chunks: List[ScoredChunk], apply_lost_in_middle: bool = True) -> str:
        """
        Builds a structured context string from scored chunks for inclusion in the prompt.
        """
        if not chunks:
            return "No relevant context found."

        ordered_chunks = cls.reorder_lost_in_the_middle(chunks) if apply_lost_in_middle else chunks

        context_blocks = []
        for i, sc in enumerate(ordered_chunks, start=1):
            c = sc.chunk
            header = f"[Source {i}: {c.source_file or 'Document'} | Chunk {c.chunk_index} | Rank {sc.rank}]"
            block = f"{header}\n{c.text.strip()}"
            context_blocks.append(block)

        return "\n\n---\n\n".join(context_blocks)

