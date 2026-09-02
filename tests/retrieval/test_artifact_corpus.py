from pathlib import Path

import pytest

from atlasrag.ingestion.artifact_store import (
    save_chunked_document,
    save_embedded_document,
)
from atlasrag.ingestion.chunker import ChunkedDocument, DoclingHybridChunker
from atlasrag.ingestion.embedder import embed_chunked_document
from atlasrag.ingestion.loader import load_local_document
from atlasrag.ingestion.models import DocumentSource
from atlasrag.ingestion.parser import DoclingParser
from atlasrag.ingestion.pipeline import (
    build_chunk_artifact_path,
    build_embedding_artifact_path,
)
from atlasrag.retrieval.artifact_corpus import (
    ArtifactCorpusError,
    load_artifact_corpus,
)

from .test_vector_index import SemanticTestEmbedder


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


def _save_retrieval_artifacts(
    root: Path,
    chunked: ChunkedDocument,
    embedder: SemanticTestEmbedder,
) -> tuple[Path, Path]:
    embedded = embed_chunked_document(chunked, embedder)
    chunk_path = build_chunk_artifact_path(
        artifact_root=root,
        chunked=chunked,
    )
    embedding_path = build_embedding_artifact_path(
        artifact_root=root,
        embedded=embedded,
    )
    save_chunked_document(chunked, chunk_path)
    save_embedded_document(embedded, embedding_path)
    return chunk_path, embedding_path


def test_load_artifact_corpus_hydrates_searchable_index(
    tmp_path: Path,
    chunked_policy: ChunkedDocument,
) -> None:
    embedder = SemanticTestEmbedder()
    _save_retrieval_artifacts(tmp_path, chunked_policy, embedder)

    corpus = load_artifact_corpus(artifact_root=tmp_path, embedder=embedder)
    hits = corpus.index.search("How do I report an incident?", top_k=2)

    assert corpus.document_count == 1
    assert corpus.chunk_count == len(chunked_policy.chunks)
    assert [hit.chunk.chunk_index for hit in hits] == [0, 1]
    assert hits[0].document_version == chunked_policy.version


def test_load_artifact_corpus_ignores_other_embedding_configurations(
    tmp_path: Path,
    chunked_policy: ChunkedDocument,
) -> None:
    _save_retrieval_artifacts(
        tmp_path,
        chunked_policy,
        SemanticTestEmbedder("other-model"),
    )

    corpus = load_artifact_corpus(
        artifact_root=tmp_path,
        embedder=SemanticTestEmbedder("requested-model"),
    )

    assert corpus.document_count == 0
    assert corpus.chunk_count == 0
    assert corpus.index.search("reporting") == ()


def test_load_artifact_corpus_rejects_missing_chunk_artifact(
    tmp_path: Path,
    chunked_policy: ChunkedDocument,
) -> None:
    embedder = SemanticTestEmbedder()
    chunk_path, _ = _save_retrieval_artifacts(tmp_path, chunked_policy, embedder)
    chunk_path.unlink()

    with pytest.raises(ArtifactCorpusError, match="Missing chunk artifact"):
        load_artifact_corpus(artifact_root=tmp_path, embedder=embedder)


def test_load_artifact_corpus_reports_invalid_embedding_artifact(
    tmp_path: Path,
    chunked_policy: ChunkedDocument,
) -> None:
    embedder = SemanticTestEmbedder()
    _, embedding_path = _save_retrieval_artifacts(
        tmp_path,
        chunked_policy,
        embedder,
    )
    embedding_path.write_text("not json", encoding="utf-8")

    with pytest.raises(ArtifactCorpusError, match="Invalid embedding artifact"):
        load_artifact_corpus(artifact_root=tmp_path, embedder=embedder)


def test_load_artifact_corpus_rejects_missing_root(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="Artifact root does not exist"):
        load_artifact_corpus(
            artifact_root=missing,
            embedder=SemanticTestEmbedder(),
        )
