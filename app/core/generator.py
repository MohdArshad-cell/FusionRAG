import os
import asyncio
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, List
from app.config import Settings
from app.utils.logger import logger


SYSTEM_PROMPT = """You are FusionRAG, an accurate and concise production AI assistant.
Answer the user's question based strictly on the provided Context.
Rules:
1. Ground your answer in the facts from the Context.
2. If the Context contains exact codes (e.g. ERR_404_AUTH, X-500) or specific names, cite them accurately.
3. If the context does not contain enough information to answer the question, clearly state that the answer is not present in the indexed documents.
4. Do not hallucinate or extrapolate beyond the provided text.
"""


class BaseLLMGenerator(ABC):
    """Abstract interface for LLM response generators."""

    @abstractmethod
    async def generate(self, query: str, context: str) -> str:
        """Generate a complete text response."""
        pass

    @abstractmethod
    async def generate_stream(self, query: str, context: str) -> AsyncIterator[str]:
        """Stream response tokens as an async iterator."""
        pass


class GroqGenerator(BaseLLMGenerator):
    """LLM Generator using Groq API."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile", temperature: float = 0.2):
        from groq import AsyncGroq
        self.client = AsyncGroq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        logger.info(f"GroqGenerator initialized with model '{self.model}'")

    async def generate(self, query: str, context: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        return response.choices[0].message.content or ""

    async def generate_stream(self, query: str, context: str) -> AsyncIterator[str]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True
        )
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content


class GeminiGenerator(BaseLLMGenerator):
    """LLM Generator using Google Gemini API."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", temperature: float = 0.2):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config={"temperature": temperature}
        )
        logger.info(f"GeminiGenerator initialized with model '{model}'")

    async def generate(self, query: str, context: str) -> str:
        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}"
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self.model.generate_content, prompt)
        return response.text or ""

    async def generate_stream(self, query: str, context: str) -> AsyncIterator[str]:
        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}"
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: self.model.generate_content(prompt, stream=True))
        for chunk in response:
            if chunk.text:
                yield chunk.text


class MockGenerator(BaseLLMGenerator):
    """
    Mock generator for testing, local offline development, or CI.
    Synthesizes an answer directly from the top context snippets without external API keys.
    """

    def __init__(self):
        logger.info("MockGenerator initialized (offline/testing mode).")

    async def generate(self, query: str, context: str) -> str:
        await asyncio.sleep(0.05)
        if not context or "No relevant context" in context:
            return "Based on the provided documents, there is no relevant information to answer this query."

        # Extract the first sentence or lines from context
        lines = [line.strip() for line in context.split("\n") if line.strip() and not line.startswith("[Source") and not line.startswith("---")]
        snippet = " ".join(lines[:3]) if lines else "Relevant context was retrieved."
        return f"[FusionRAG Answer] Query: '{query}'. Based on the verified context: {snippet}"

    async def generate_stream(self, query: str, context: str) -> AsyncIterator[str]:
        full_text = await self.generate(query, context)
        words = full_text.split(" ")
        for word in words:
            await asyncio.sleep(0.02)
            yield word + " "


def get_llm_generator(settings: Settings) -> BaseLLMGenerator:
    """Factory to construct the configured LLM provider."""
    provider = settings.LLM_PROVIDER.lower()

    if provider == "groq" and settings.GROQ_API_KEY:
        try:
            return GroqGenerator(
                api_key=settings.GROQ_API_KEY,
                model=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE
            )
        except Exception as e:
            logger.warning(f"Could not initialize GroqGenerator: {e}. Falling back to MockGenerator.")

    if provider == "gemini" and settings.GEMINI_API_KEY:
        try:
            return GeminiGenerator(
                api_key=settings.GEMINI_API_KEY,
                model=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE
            )
        except Exception as e:
            logger.warning(f"Could not initialize GeminiGenerator: {e}. Falling back to MockGenerator.")

    # Default / fallback
    logger.info("Using MockGenerator (no active API key or provider set to mock).")
    return MockGenerator()

