from pathlib import Path

from atlasrag.ingestion.artifact_store import (
    load_parsed_document,
    save_parsed_document,
)
from atlasrag.ingestion.loader import load_local_document
from atlasrag.ingestion.models import DocumentSource
from atlasrag.ingestion.parser import DoclingParser


def test_parsed_document_can_be_saved_and_loaded(
    tmp_path: Path,
) -> None:
    document_path = Path("tests/fixtures/documents/security_policy.md")
    source = DocumentSource(
        source_namespace="test-fixtures",
        source_key="security/policy",
    )
    loaded = load_local_document(
        path=document_path,
        source=source,
    )
    original = DoclingParser().parse(loaded)

    artifact_path = tmp_path / "parsed" / "security_policy.json"

    save_parsed_document(
        original,
        artifact_path,
    )
    restored = load_parsed_document(artifact_path)

    assert artifact_path.exists()
    assert restored.version == original.version
    assert restored.parser_name == original.parser_name
    assert restored.parser_version == original.parser_version
    assert restored.document.model_dump() == original.document.model_dump()
