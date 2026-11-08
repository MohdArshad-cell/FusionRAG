import re
from typing import List, Dict, Any, Tuple
from app.core.models import ScoredChunk
from app.utils.logger import logger


class RetrievalConfidenceGate:
    """
    Corrective RAG (CRAG) Confidence Gate:
    Evaluates the quality of retrieved candidates before calling the LLM.
    Prevents hallucinating answers when no relevant information exists in the index.
    """

    def __init__(self, high_confidence_threshold: float = 0.50, low_confidence_threshold: float = -2.0):
        self.high_threshold = high_confidence_threshold
        self.low_threshold = low_confidence_threshold

    def evaluate_candidates(self, candidates: List[ScoredChunk]) -> Dict[str, Any]:
        """
        Assesses top Cross-Encoder candidate scores:
        - HIGH: At least one strong evidence chunk found.
        - AMBIGUOUS: Weak evidence; pass context with a strict grounding warning.
        - INSUFFICIENT: No relevant chunks found; safe abstention recommended.
        """
        if not candidates:
            return {
                "decision": "INSUFFICIENT",
                "top_score": None,
                "reason": "Retrieval returned 0 candidate chunks."
            }

        top_score = candidates[0].score

        # ms-marco cross-encoder produces logits typically ranging from -10 to +10
        if top_score >= self.high_threshold:
            decision = "HIGH"
            reason = f"Top candidate score ({top_score:.2f}) meets high confidence threshold."
        elif top_score >= self.low_threshold:
            decision = "AMBIGUOUS"
            reason = f"Top candidate score ({top_score:.2f}) is borderline."
        else:
            decision = "INSUFFICIENT"
            reason = f"Top candidate score ({top_score:.2f}) is below minimum threshold ({self.low_threshold})."

        logger.info(f"CRAG Gate evaluation: {decision} ({reason})")
        return {
            "decision": decision,
            "top_score": top_score,
            "reason": reason
        }


class GroundednessVerifier:
    """
    Post-generation Faithfulness & Groundedness Guardrail.
    Verifies that the generated answer is strictly supported by the retrieved context.
    """

    @staticmethod
    def extract_key_tokens(text: str) -> set:
        """Extracts substantive content words (ignoring stopwords and punctuation)."""
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "and", "or", "in", "on", "at",
            "to", "for", "with", "by", "from", "of", "it", "this", "that", "these",
            "those", "as", "be", "have", "has", "had", "do", "does", "did", "but", "not"
        }
        tokens = re.findall(r"[a-z0-9_\-]+", text.lower())
        return {t for t in tokens if len(t) > 2 and t not in stopwords}

    @classmethod
    def verify(cls, answer: str, context: str) -> Dict[str, Any]:
        """
        Computes a groundedness score based on token and entity overlap between
        the generated answer and the source context.
        """
        if not answer.strip() or not context.strip():
            return {
                "groundedness_score": 0.0,
                "status": "UNGROUNDED",
                "hallucination_risk": "HIGH"
            }

        # Check if model responded that no context was available
        abstain_phrases = [
            "no relevant context",
            "not present in the indexed",
            "does not contain enough information",
            "no verifiable evidence"
        ]
        if any(phrase in answer.lower() for phrase in abstain_phrases):
            return {
                "groundedness_score": 1.0,
                "status": "SAFE_ABSTENTION",
                "hallucination_risk": "NONE"
            }

        answer_tokens = cls.extract_key_tokens(answer)
        if not answer_tokens:
            return {
                "groundedness_score": 1.0,
                "status": "HIGHLY_GROUNDED",
                "hallucination_risk": "LOW"
            }

        context_tokens = cls.extract_key_tokens(context)
        supported_tokens = answer_tokens.intersection(context_tokens)

        overlap_ratio = len(supported_tokens) / len(answer_tokens)
        score = round(min(1.0, overlap_ratio * 1.2), 2)  # slight scaling for sentence connective terms

        if score >= 0.70:
            status = "HIGHLY_GROUNDED"
            risk = "LOW"
        elif score >= 0.45:
            status = "MODERATELY_GROUNDED"
            risk = "MEDIUM"
        else:
            status = "POTENTIAL_HALLUCINATION"
            risk = "HIGH"

        logger.info(f"Groundedness verification: score={score}, status={status}, risk={risk}")
        return {
            "groundedness_score": score,
            "status": status,
            "hallucination_risk": risk
        }

