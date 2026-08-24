import pytest
from pydantic import ValidationError

from atlasrag.ingestion.identity import create_document_id
from atlasrag.ingestion.models import DocumentSource


def test_document_source_derives_document_id() -> None:
    source = DocumentSource(
        source_namespace="local-manifest",
        source_key="security/policy",
    )

    expected = create_document_id(
        "local-manifest",
        "security/policy",
    )

    assert source.document_id == expected


@pytest.mark.parametrize(
    ("source_namespace", "source_key"),
    [
        ("", "security/policy"),
        ("local-manifest", ""),
    ],
)
def test_document_source_rejects_blank_identity(
    source_namespace: str,
    source_key: str,
) -> None:
    with pytest.raises(ValidationError):
        DocumentSource(
            source_namespace=source_namespace,
            source_key=source_key,
        )
