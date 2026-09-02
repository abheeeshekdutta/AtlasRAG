import json
from typing import Literal, Protocol, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from atlasrag.ingestion.chunker import ChunkedDocument
from atlasrag.ingestion.identity import calculate_content_sha256
from atlasrag.ingestion.models import DocumentVersion

EMBEDDER_ALGORITHM_VERSION = 1


class EmbeddingConfig(BaseModel):
    """Settings that identify an embedding model and its output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    model: str
    model_revision: str
    dimensions: int = Field(gt=0)
    normalize: bool = True

    @model_validator(mode="after")
    def reject_blank_names(self) -> Self:
        if not self.provider.strip():
            raise ValueError("provider must not be blank")
        if not self.model.strip():
            raise ValueError("model must not be blank")
        if not self.model_revision.strip():
            raise ValueError("model_revision must not be blank")
        return self


def create_embedder_fingerprint(
    config: EmbeddingConfig,
    *,
    provider_version: str,
) -> str:
    """Hash every setting that can change embedding output."""
    if not provider_version.strip():
        raise ValueError("provider_version must not be blank")

    payload = {
        "algorithm_version": EMBEDDER_ALGORITHM_VERSION,
        "config": config.model_dump(mode="json"),
        "provider_version": provider_version,
    }
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return calculate_content_sha256(canonical_json.encode("utf-8"))


class DocumentEmbedder(Protocol):
    """Provider-neutral boundary for document embedding implementations."""

    config: EmbeddingConfig
    provider_version: str

    def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]: ...


class ChunkEmbedding(BaseModel):
    """Embedding vector coupled to the chunk that produced it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    chunk_id: UUID
    chunk_index: int = Field(ge=0)
    values: tuple[FiniteFloat, ...]


class EmbeddedDocument(BaseModel):
    """Complete embedding result for one chunked document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    version: DocumentVersion
    chunker_fingerprint: str
    config: EmbeddingConfig
    provider_version: str
    embedder_fingerprint: str
    embeddings: tuple[ChunkEmbedding, ...]

    @model_validator(mode="after")
    def validate_artifact(self) -> Self:
        if not self.chunker_fingerprint.strip():
            raise ValueError("chunker_fingerprint must not be blank")
        if not self.embeddings:
            raise ValueError("embeddings must not be empty")

        expected_fingerprint = create_embedder_fingerprint(
            self.config,
            provider_version=self.provider_version,
        )
        if self.embedder_fingerprint != expected_fingerprint:
            raise ValueError(
                "embedder_fingerprint does not match the embedding configuration"
            )

        expected_indexes = tuple(range(len(self.embeddings)))
        actual_indexes = tuple(item.chunk_index for item in self.embeddings)
        if actual_indexes != expected_indexes:
            raise ValueError("embedding indexes must be contiguous and ordered")

        if any(len(item.values) != self.config.dimensions for item in self.embeddings):
            raise ValueError("embedding dimensions do not match the configuration")

        return self


class DocumentEmbeddingError(RuntimeError):
    """Raised when an embedding provider returns an invalid batch."""


def embed_chunked_document(
    chunked: ChunkedDocument,
    embedder: DocumentEmbedder,
) -> EmbeddedDocument:
    """Embed contextualized chunk text and validate provider output."""
    if not embedder.provider_version.strip():
        raise DocumentEmbeddingError("provider_version must not be blank")

    texts = tuple(chunk.contextualized_text for chunk in chunked.chunks)
    vectors = embedder.embed_documents(texts)

    if len(vectors) != len(chunked.chunks):
        raise DocumentEmbeddingError(
            f"Provider returned {len(vectors)} vectors for {len(chunked.chunks)} chunks"
        )

    embeddings: list[ChunkEmbedding] = []
    for chunk, vector in zip(chunked.chunks, vectors, strict=True):
        if len(vector) != embedder.config.dimensions:
            raise DocumentEmbeddingError(
                f"Provider returned {len(vector)} dimensions for chunk "
                f"{chunk.chunk_index}; expected {embedder.config.dimensions}"
            )
        embeddings.append(
            ChunkEmbedding(
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                values=vector,
            )
        )

    return EmbeddedDocument(
        version=chunked.version,
        chunker_fingerprint=chunked.chunker_fingerprint,
        config=embedder.config,
        provider_version=embedder.provider_version,
        embedder_fingerprint=create_embedder_fingerprint(
            embedder.config,
            provider_version=embedder.provider_version,
        ),
        embeddings=tuple(embeddings),
    )
