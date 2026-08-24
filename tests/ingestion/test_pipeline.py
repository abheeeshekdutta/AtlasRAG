from pathlib import Path

from atlasrag.ingestion.artifact_store import load_parsed_document
from atlasrag.ingestion.models import DocumentSource
from atlasrag.ingestion.parser import DoclingParser
from atlasrag.ingestion.pipeline import ingest_local_document


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
    )

    assert result.artifact_path.exists()

    assert str(result.parsed.version.source.document_id) in str(result.artifact_path)
    assert str(result.parsed.version.document_version_id) in str(result.artifact_path)

    restored = load_parsed_document(result.artifact_path)

    assert restored.version == result.parsed.version
    assert restored.document.model_dump() == result.parsed.document.model_dump()
