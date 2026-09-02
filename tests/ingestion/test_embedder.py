from pathlib import Path

import pytest
from pydantic import ValidationError

from atlasrag.ingestion.chunker import ChunkedDocument, DoclingHybridChunker
from atlasrag.ingestion.embedder import (
    DocumentEmbeddingError,
    EmbeddedDocument,
    EmbeddingConfig,
    create_embedder_fingerprint,
    embed_chunked_document,
)
from atlasrag.ingestion.loader import load_local_document
from atlasrag.ingestion.models import DocumentSource
from atlasrag.ingestion.parser import DoclingParser


class RecordingEmbedder:
    def __init__(
        self,
        *,
        dimensions: int = 3,
        output_dimensions: int | None = None,
        vector_count_offset: int = 0,
    ) -> None:
        self.config = EmbeddingConfig(
            provider="test-provider",
            model="test-model",
            dimensions=dimensions,
        )
        self.provider_version = "1.2.3"
        self.output_dimensions = output_dimensions or dimensions
        self.vector_count_offset = vector_count_offset
        self.received_texts: tuple[str, ...] = ()

    def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        self.received_texts = texts
        count = len(texts) + self.vector_count_offset
        return tuple(
            tuple(
                float(index + dimension) for dimension in range(self.output_dimensions)
            )
            for index in range(max(0, count))
        )


@pytest.fixture(scope="module")
def chunked_policy() -> ChunkedDocument:
    loaded = load_local_document(
        path=Path("tests/fixtures/documents/security_policy.md"),
        source=DocumentSource(
            source_namespace="test-fixtures",
            source_key="security/policy",
        ),
    )
    return DoclingHybridChunker().chunk(DoclingParser().parse(loaded))


@pytest.mark.parametrize("field", ["provider", "model"])
@pytest.mark.parametrize("value", ["", "   "])
def test_embedding_config_rejects_blank_names(field: str, value: str) -> None:
    data = {
        "provider": "test-provider",
        "model": "test-model",
        "dimensions": 3,
    }
    data[field] = value

    with pytest.raises(ValidationError, match=f"{field} must not be blank"):
        EmbeddingConfig.model_validate(data)


def test_embedder_fingerprint_is_deterministic_and_configuration_sensitive() -> None:
    config = EmbeddingConfig(provider="test", model="model-a", dimensions=3)

    first = create_embedder_fingerprint(config, provider_version="1.0")
    repeated = create_embedder_fingerprint(config, provider_version="1.0")
    changed = create_embedder_fingerprint(
        config.model_copy(update={"normalize": False}),
        provider_version="1.0",
    )

    assert first == repeated
    assert first != changed


def test_embed_chunked_document_preserves_order_and_lineage(
    chunked_policy: ChunkedDocument,
) -> None:
    embedder = RecordingEmbedder()

    result = embed_chunked_document(chunked_policy, embedder)

    assert embedder.received_texts == tuple(
        chunk.contextualized_text for chunk in chunked_policy.chunks
    )
    assert result.version == chunked_policy.version
    assert result.chunker_fingerprint == chunked_policy.chunker_fingerprint
    assert [item.chunk_id for item in result.embeddings] == [
        chunk.chunk_id for chunk in chunked_policy.chunks
    ]
    assert [item.chunk_index for item in result.embeddings] == list(
        range(len(chunked_policy.chunks))
    )
    assert all(len(item.values) == 3 for item in result.embeddings)


def test_embedded_document_round_trips_through_json(
    chunked_policy: ChunkedDocument,
) -> None:
    original = embed_chunked_document(chunked_policy, RecordingEmbedder())

    restored = EmbeddedDocument.model_validate_json(
        original.model_dump_json(exclude_computed_fields=True)
    )

    assert restored == original


@pytest.mark.parametrize("offset", [-1, 1])
def test_embed_chunked_document_rejects_wrong_vector_count(
    chunked_policy: ChunkedDocument,
    offset: int,
) -> None:
    with pytest.raises(DocumentEmbeddingError, match="vectors for"):
        embed_chunked_document(
            chunked_policy,
            RecordingEmbedder(vector_count_offset=offset),
        )


def test_embed_chunked_document_rejects_wrong_dimensions(
    chunked_policy: ChunkedDocument,
) -> None:
    with pytest.raises(DocumentEmbeddingError, match="expected 4"):
        embed_chunked_document(
            chunked_policy,
            RecordingEmbedder(dimensions=4, output_dimensions=3),
        )


def test_embedded_document_rejects_non_finite_values(
    chunked_policy: ChunkedDocument,
) -> None:
    original = embed_chunked_document(chunked_policy, RecordingEmbedder())
    data = original.model_dump(mode="json", exclude_computed_fields=True)
    data["embeddings"][0]["values"][0] = float("nan")

    with pytest.raises(ValidationError, match="finite number"):
        EmbeddedDocument.model_validate(data)
