from pathlib import Path

from atlasrag.ingestion.artifact_store import (
    load_chunked_document,
    load_parsed_document,
)
from atlasrag.ingestion.chunker import ChunkingConfig, DoclingHybridChunker
from atlasrag.ingestion.models import DocumentSource
from atlasrag.ingestion.parser import DoclingParser
from atlasrag.ingestion.pipeline import (
    build_chunk_artifact_path,
    ingest_local_document,
)


def test_ingest_local_document_runs_complete_pipeline(
    tmp_path: Path,
) -> None:
    source = DocumentSource(
        source_namespace="test-fixtures",
        source_key="security/policy",
    )

    result = ingest_local_document(
        path=Path("tests/fixtures/documents/security_policy.md"),
        source=source,
        artifact_root=tmp_path / "artifacts",
        parser=DoclingParser(),
        chunker=DoclingHybridChunker(),
    )

    assert result.artifact_path.exists()
    assert result.chunk_artifact_path.exists()

    assert str(result.parsed.version.source.document_id) in str(result.artifact_path)
    assert str(result.parsed.version.document_version_id) in str(result.artifact_path)

    restored = load_parsed_document(result.artifact_path)

    assert restored.version == result.parsed.version
    assert restored.document.model_dump() == result.parsed.document.model_dump()

    restored_chunks = load_chunked_document(result.chunk_artifact_path)

    assert restored_chunks == result.chunked
    assert restored_chunks.version == result.parsed.version
    assert result.chunk_artifact_path == build_chunk_artifact_path(
        artifact_root=tmp_path / "artifacts",
        chunked=result.chunked,
    )


def test_chunk_artifact_path_changes_with_chunking_configuration(
    tmp_path: Path,
) -> None:
    source = DocumentSource(
        source_namespace="test-fixtures",
        source_key="security/policy",
    )
    first = ingest_local_document(
        path=Path("tests/fixtures/documents/security_policy.md"),
        source=source,
        artifact_root=tmp_path / "artifacts",
        parser=DoclingParser(),
        chunker=DoclingHybridChunker(),
    )
    second = ingest_local_document(
        path=Path("tests/fixtures/documents/security_policy.md"),
        source=source,
        artifact_root=tmp_path / "artifacts",
        parser=DoclingParser(),
        chunker=DoclingHybridChunker(ChunkingConfig(chunk_max_tokens=199)),
    )

    assert first.chunk_artifact_path != second.chunk_artifact_path
    assert first.chunk_artifact_path.exists()
    assert second.chunk_artifact_path.exists()
