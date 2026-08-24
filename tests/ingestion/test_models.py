import pytest
from pydantic import ValidationError

from atlasrag.ingestion.identity import (
    calculate_content_sha256,
    create_document_id,
    create_document_version_id,
)
from atlasrag.ingestion.models import (
    DocumentFormat,
    DocumentSource,
    DocumentVersion,
)


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


def test_document_version_from_content_derives_metadata() -> None:
    source = DocumentSource(
        source_namespace="local-manifest",
        source_key="security/policy",
    )
    content = b"Security incidents must be reported immediately."

    version = DocumentVersion.from_content(
        source=source,
        source_format=DocumentFormat.MARKDOWN,
        content=content,
    )

    assert version.content_sha256 == calculate_content_sha256(content)
    assert version.size_bytes == len(content)
    assert version.document_version_id == create_document_version_id(
        source.document_id,
        version.content_sha256,
    )


def test_document_version_rejects_malformed_content_hash() -> None:
    source = DocumentSource(
        source_namespace="local-manifest",
        source_key="security/policy",
    )

    with pytest.raises(ValidationError):
        DocumentVersion(
            source=source,
            source_format=DocumentFormat.PDF,
            content_sha256="not-a-sha256-digest",
            size_bytes=100,
        )


def test_document_version_rejects_negative_size() -> None:
    source = DocumentSource(
        source_namespace="local-manifest",
        source_key="security/policy",
    )
    content_sha256 = calculate_content_sha256(b"content")

    with pytest.raises(ValidationError):
        DocumentVersion(
            source=source,
            source_format=DocumentFormat.PDF,
            content_sha256=content_sha256,
            size_bytes=-1,
        )


def test_document_version_serializes_with_derived_ids() -> None:
    source = DocumentSource(
        source_namespace="local-manifest",
        source_key="security/policy",
    )
    version = DocumentVersion.from_content(
        source=source,
        source_format=DocumentFormat.MARKDOWN,
        content=b"Security policy",
    )

    payload = version.model_dump(mode="json")

    assert payload["source"]["document_id"] == str(source.document_id)
    assert payload["document_version_id"] == str(version.document_version_id)
    assert payload["source_format"] == "markdown"


def test_document_version_round_trips_through_canonical_json() -> None:
    source = DocumentSource(
        source_namespace="local-manifest",
        source_key="security/policy",
    )
    original = DocumentVersion.from_content(
        source=source,
        source_format=DocumentFormat.MARKDOWN,
        content=b"Security policy",
    )

    serialized = original.model_dump_json(
        exclude_computed_fields=True,
    )
    restored = DocumentVersion.model_validate_json(serialized)

    assert restored == original
    assert restored.source.document_id == original.source.document_id
    assert restored.document_version_id == original.document_version_id
