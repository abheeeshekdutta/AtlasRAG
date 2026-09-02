from pathlib import Path
from uuid import UUID

import pytest

from atlasrag.ingestion.chunker import ChunkedDocument, DoclingHybridChunker
from atlasrag.ingestion.embedder import EmbeddingConfig, embed_chunked_document
from atlasrag.ingestion.loader import load_local_document
from atlasrag.ingestion.models import DocumentSource
from atlasrag.ingestion.parser import DoclingParser
from atlasrag.retrieval.vector_index import InMemoryVectorIndex, VectorIndexError


class SemanticTestEmbedder:
    def __init__(self, model: str = "semantic-test-v1") -> None:
        self.config = EmbeddingConfig(
            provider="test",
            model=model,
            model_revision="revision-1",
            dimensions=3,
            normalize=True,
        )
        self.provider_version = "1.0.0"
        self.query_calls: list[str] = []

    def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        basis = (
            (1.0, 0.0, 0.0),
            (0.9, 0.1, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        return tuple(basis[index] for index, _ in enumerate(texts))

    def embed_query(self, text: str) -> tuple[float, ...]:
        self.query_calls.append(text)
        if "report" in text.casefold():
            return (1.0, 0.0, 0.0)
        return (0.0, 0.0, 1.0)


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


def test_vector_index_returns_ranked_chunks_with_lineage(
    chunked_policy: ChunkedDocument,
) -> None:
    embedder = SemanticTestEmbedder()
    embedded = embed_chunked_document(chunked_policy, embedder)
    index = InMemoryVectorIndex(embedder)
    index.add(chunked=chunked_policy, embedded=embedded)

    hits = index.search("How should I report an incident?", top_k=2)

    assert [hit.chunk.chunk_index for hit in hits] == [0, 1]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].document_version == chunked_policy.version
    assert hits[0].chunk.contextualized_text
    assert embedder.query_calls == ["How should I report an incident?"]


def test_vector_index_applies_score_threshold(
    chunked_policy: ChunkedDocument,
) -> None:
    embedder = SemanticTestEmbedder()
    index = InMemoryVectorIndex(embedder)
    index.add(
        chunked=chunked_policy,
        embedded=embed_chunked_document(chunked_policy, embedder),
    )

    hits = index.search("reporting", top_k=4, minimum_score=0.5)

    assert [hit.chunk.chunk_index for hit in hits] == [0, 1]


def test_empty_vector_index_returns_without_embedding_query() -> None:
    embedder = SemanticTestEmbedder()
    index = InMemoryVectorIndex(embedder)

    assert index.search("reporting") == ()
    assert embedder.query_calls == []


def test_vector_index_rejects_different_embedding_configuration(
    chunked_policy: ChunkedDocument,
) -> None:
    index = InMemoryVectorIndex(SemanticTestEmbedder("model-a"))
    other_embedder = SemanticTestEmbedder("model-b")

    with pytest.raises(
        VectorIndexError,
        match="embedding configuration does not match the index",
    ):
        index.add(
            chunked=chunked_policy,
            embedded=embed_chunked_document(chunked_policy, other_embedder),
        )


def test_vector_index_rejects_misaligned_chunk_ids(
    chunked_policy: ChunkedDocument,
) -> None:
    embedder = SemanticTestEmbedder()
    embedded = embed_chunked_document(chunked_policy, embedder)
    bad_first = embedded.embeddings[0].model_copy(
        update={"chunk_id": UUID("16624b43-95c7-55d5-a277-ecec22198a2c")}
    )
    misaligned = embedded.model_copy(
        update={"embeddings": (bad_first, *embedded.embeddings[1:])}
    )

    with pytest.raises(VectorIndexError, match="chunk_id does not match"):
        InMemoryVectorIndex(embedder).add(
            chunked=chunked_policy,
            embedded=misaligned,
        )


def test_vector_index_rejects_vectors_with_overflowing_norm(
    chunked_policy: ChunkedDocument,
) -> None:
    embedder = SemanticTestEmbedder()
    embedded = embed_chunked_document(chunked_policy, embedder)
    oversized_first = embedded.embeddings[0].model_copy(
        update={"values": (1.7e308, 1.7e308, 1.7e308)}
    )
    oversized = embedded.model_copy(
        update={"embeddings": (oversized_first, *embedded.embeddings[1:])}
    )

    with pytest.raises(VectorIndexError, match="norm must be finite and positive"):
        InMemoryVectorIndex(embedder).add(
            chunked=chunked_policy,
            embedded=oversized,
        )


@pytest.mark.parametrize("query", ["", "   "])
def test_vector_index_rejects_blank_query(query: str) -> None:
    with pytest.raises(ValueError, match="query must not be blank"):
        InMemoryVectorIndex(SemanticTestEmbedder()).search(query)


@pytest.mark.parametrize("top_k", [0, -1])
def test_vector_index_rejects_nonpositive_top_k(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k must be positive"):
        InMemoryVectorIndex(SemanticTestEmbedder()).search("query", top_k=top_k)


@pytest.mark.parametrize("minimum_score", [-1.1, 1.1, float("nan")])
def test_vector_index_rejects_invalid_minimum_score(minimum_score: float) -> None:
    with pytest.raises(ValueError, match="minimum_score"):
        InMemoryVectorIndex(SemanticTestEmbedder()).search(
            "query",
            minimum_score=minimum_score,
        )
