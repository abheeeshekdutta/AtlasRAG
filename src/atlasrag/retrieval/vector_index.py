import math
from dataclasses import dataclass
from uuid import UUID

from atlasrag.ingestion.chunker import ChunkedDocument, DocumentChunk
from atlasrag.ingestion.embedder import (
    EmbeddedDocument,
    QueryEmbedder,
    create_embedder_fingerprint,
)
from atlasrag.ingestion.models import DocumentVersion


class VectorIndexError(RuntimeError):
    """Raised when artifacts or query vectors are incompatible with an index."""


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A scored chunk returned from semantic search."""

    document_version: DocumentVersion
    chunk: DocumentChunk
    score: float


@dataclass(frozen=True, slots=True)
class _IndexedChunk:
    document_version: DocumentVersion
    chunk: DocumentChunk
    vector: tuple[float, ...]


class InMemoryVectorIndex:
    """Small deterministic cosine index for local retrieval and testing."""

    def __init__(self, embedder: QueryEmbedder) -> None:
        self._embedder = embedder
        self._embedder_fingerprint = create_embedder_fingerprint(
            embedder.config,
            provider_version=embedder.provider_version,
        )
        self._chunks: dict[UUID, _IndexedChunk] = {}

    def add(
        self,
        *,
        chunked: ChunkedDocument,
        embedded: EmbeddedDocument,
    ) -> None:
        """Add or replace every chunk from one embedded document."""
        if embedded.version != chunked.version:
            raise VectorIndexError("embedded document version does not match chunks")
        if embedded.chunker_fingerprint != chunked.chunker_fingerprint:
            raise VectorIndexError("embedded chunker fingerprint does not match chunks")
        if embedded.embedder_fingerprint != self._embedder_fingerprint:
            raise VectorIndexError("embedding configuration does not match the index")
        if len(embedded.embeddings) != len(chunked.chunks):
            raise VectorIndexError("embedding count does not match chunk count")

        indexed: list[_IndexedChunk] = []
        for chunk, embedding in zip(
            chunked.chunks,
            embedded.embeddings,
            strict=True,
        ):
            if embedding.chunk_id != chunk.chunk_id:
                raise VectorIndexError("embedding chunk_id does not match chunks")
            if embedding.chunk_index != chunk.chunk_index:
                raise VectorIndexError("embedding chunk_index does not match chunks")
            self._validate_vector(embedding.values)
            indexed.append(
                _IndexedChunk(
                    document_version=chunked.version,
                    chunk=chunk,
                    vector=embedding.values,
                )
            )

        for item in indexed:
            self._chunks[item.chunk.chunk_id] = item

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        minimum_score: float = -1.0,
    ) -> tuple[SearchHit, ...]:
        """Return the highest-scoring chunks for a natural-language query."""
        if not query.strip():
            raise ValueError("query must not be blank")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not math.isfinite(minimum_score) or not -1.0 <= minimum_score <= 1.0:
            raise ValueError("minimum_score must be finite and between -1 and 1")
        if not self._chunks:
            return ()

        query_vector = self._embedder.embed_query(query)
        self._validate_vector(query_vector)

        scored = (
            SearchHit(
                document_version=item.document_version,
                chunk=item.chunk,
                score=self._cosine_similarity(query_vector, item.vector),
            )
            for item in self._chunks.values()
        )
        eligible = (hit for hit in scored if hit.score >= minimum_score)
        ordered = sorted(
            eligible,
            key=lambda hit: (-hit.score, str(hit.chunk.chunk_id)),
        )
        return tuple(ordered[:top_k])

    def _validate_vector(self, vector: tuple[float, ...]) -> None:
        if len(vector) != self._embedder.config.dimensions:
            raise VectorIndexError(
                f"vector has {len(vector)} dimensions; "
                f"expected {self._embedder.config.dimensions}"
            )
        if not all(math.isfinite(value) for value in vector):
            raise VectorIndexError("vector values must be finite")
        norm = math.hypot(*vector)
        if not math.isfinite(norm) or norm == 0.0:
            raise VectorIndexError("vector norm must be finite and positive")

    @staticmethod
    def _cosine_similarity(
        left: tuple[float, ...],
        right: tuple[float, ...],
    ) -> float:
        dot_product = math.fsum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.hypot(*left)
        right_norm = math.hypot(*right)
        score = dot_product / (left_norm * right_norm)
        return max(-1.0, min(1.0, score))
