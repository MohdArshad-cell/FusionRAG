import os
import pickle
import threading
from typing import List, Dict, Optional
from app.core.models import Chunk, ScoredChunk
from app.utils.logger import logger


class ParentDocumentStore:
    """
    Manages parent chunks for 'Small-to-Big' retrieval.
    Stores larger, highly-contextual document sections that are looked up
    when smaller, high-precision child chunks match the user query.
    """

    def __init__(self, persist_dir: str = "./data"):
        self.persist_dir = persist_dir
        self.file_path = os.path.join(self.persist_dir, "parent_store.pkl")
        self._lock = threading.Lock()
        self.parents: Dict[str, Chunk] = {}

        os.makedirs(self.persist_dir, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "rb") as f:
                    self.parents = pickle.load(f)
                logger.info(f"Loaded {len(self.parents)} parent chunks from store.")
            except Exception as e:
                logger.warning(f"Could not load parent store: {e}. Starting fresh.")
                self.parents = {}

    def save(self) -> None:
        with self._lock:
            tmp = self.file_path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(self.parents, f)
            os.replace(tmp, self.file_path)

    def add_parents(self, parents: List[Chunk]) -> None:
        if not parents:
            return
        with self._lock:
            for p in parents:
                self.parents[p.id] = p
        self.save()
        logger.info(f"Stored {len(parents)} parent chunks in ParentDocumentStore.")

    def get_parent(self, parent_id: str) -> Optional[Chunk]:
        return self.parents.get(parent_id)

    def resolve_to_parent_chunks(self, scored_children: List[ScoredChunk]) -> List[ScoredChunk]:
        """
        Takes top-ranked child chunks and resolves each to its parent chunk.
        Deduplicates parent chunks while preserving the best score and highest rank.
        """
        seen_parent_ids = set()
        resolved: List[ScoredChunk] = []

        for sc in scored_children:
            child = sc.chunk
            parent_id = child.parent_id

            # If child has a parent registered in store, resolve to parent
            if parent_id and parent_id in self.parents:
                if parent_id in seen_parent_ids:
                    continue  # Deduplicate: don't include same parent twice
                seen_parent_ids.add(parent_id)
                parent_chunk = self.parents[parent_id]

                resolved.append(
                    ScoredChunk(
                        chunk=parent_chunk,
                        score=sc.score,
                        rank=len(resolved) + 1,
                        retriever=f"{sc.retriever}->parent({parent_id})"
                    )
                )
            else:
                # Fallback to child chunk itself if no parent linked
                if child.id not in seen_parent_ids:
                    seen_parent_ids.add(child.id)
                    resolved.append(sc)

        logger.info(f"Resolved {len(scored_children)} child chunks to {len(resolved)} parent context blocks.")
        return resolved

    def count(self) -> int:
        return len(self.parents)

