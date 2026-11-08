import asyncio
from typing import List, Optional
from app.core.generator import BaseLLMGenerator
from app.utils.logger import logger


class QueryProcessor:
    """
    Advanced query enhancement module providing:
    1. HyDE (Hypothetical Document Embeddings): Generates a hypothetical answer
       passage to align query embedding closer to document space.
    2. Multi-Query Expansion: Generates diverse paraphrases of the query
       to maximize recall across dense and sparse retrievers.
    """

    def __init__(self, llm: BaseLLMGenerator):
        self.llm = llm

    async def generate_hypothetical_document(self, query: str) -> str:
        """
        HyDE technique: Prompt LLM to write a speculative document passage that answers the query.
        Returns the hypothetical passage to be embedded instead of the bare question.
        """
        prompt = (
            f"Write a short, highly technical and informative paragraph answering this query. "
            f"Include relevant terms, identifiers, and domain keywords:\n\nQuery: {query}"
        )
        try:
            hypo_doc = await self.llm.generate(query=query, context=prompt)
            # Clean response to ensure it's a solid document snippet
            cleaned = hypo_doc.replace("[FusionRAG Answer]", "").strip()
            logger.info(f"HyDE generated passage ({len(cleaned)} chars) for query: '{query}'")
            return cleaned or query
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}. Falling back to original query.")
            return query

    async def expand_query(self, query: str, num_variants: int = 3) -> List[str]:
        """
        Generates alternative formulations/synonyms of the user query
        to improve lexical and semantic coverage.
        """
        prompt = (
            f"Generate {num_variants} alternative search queries or paraphrases for the query below. "
            f"Include synonyms, technical terms, and concise keyword variants.\n"
            f"Output one variant per line, without numbering or bullets.\n\nQuery: {query}"
        )
        try:
            res = await self.llm.generate(query=query, context=prompt)
            lines = [line.strip().lstrip("0123456789.-) ") for line in res.split("\n") if line.strip()]
            variants = [v for v in lines if v and v.lower() != query.lower()][:num_variants]
            # Ensure the original query is always included as primary
            all_queries = [query] + variants
            logger.info(f"Expanded query '{query}' into {len(all_queries)} variants: {all_queries}")
            return all_queries
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}. Using original query.")
            return [query]

