import pytest

from atlasrag.ingestion.identity import (
    calculate_content_sha256,
    create_document_id,
    create_document_version_id,
)


def test_calculate_content_sha256_returns_expected_digest() -> None:
    result = calculate_content_sha256(b"abc")

    assert result == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_create_document_id_is_deterministic() -> None:
    first = create_document_id("local-manifest", "security/policy")
    second = create_document_id("local-manifest", "security/policy")

    assert first == second


def test_create_document_id_changes_with_source_namespace() -> None:
    local_id = create_document_id("local-manifest", "security/policy")
    sharepoint_id = create_document_id("sharepoint", "security/policy")

    assert local_id != sharepoint_id


def test_create_document_id_changes_with_source_key() -> None:
    policy_id = create_document_id("local-manifest", "security/policy")
    runbook_id = create_document_id("local-manifest", "security/runbook")

    assert policy_id != runbook_id


@pytest.mark.parametrize(
    ("source_namespace", "source_key"),
    [
        ("", "security/policy"),
        ("   ", "security/policy"),
        ("local-manifest", ""),
        ("local-manifest", "   "),
    ],
)
def test_create_document_id_rejects_blank_identifiers(
    source_namespace: str,
    source_key: str,
) -> None:
    with pytest.raises(ValueError):
        create_document_id(source_namespace, source_key)


def test_content_sha256_changes_when_content_changes() -> None:
    first = calculate_content_sha256(b"security policy")
    second = calculate_content_sha256(b"security policz")

    assert first != second


def test_document_version_id_is_deterministic() -> None:
    document_id = create_document_id("local-manifest", "security/policy")
    content_sha256 = calculate_content_sha256(b"version one")

    first = create_document_version_id(document_id, content_sha256)
    second = create_document_version_id(document_id, content_sha256)

    assert first == second


def test_document_version_id_changes_with_content() -> None:
    document_id = create_document_id("local-manifest", "security/policy")
    first_hash = calculate_content_sha256(b"version one")
    second_hash = calculate_content_sha256(b"version two")

    first = create_document_version_id(document_id, first_hash)
    second = create_document_version_id(document_id, second_hash)

    assert first != second


def test_same_content_has_different_versions_for_different_documents() -> None:
    policy_id = create_document_id("local-manifest", "security/policy")
    runbook_id = create_document_id("local-manifest", "security/runbook")
    content_sha256 = calculate_content_sha256(b"shared content")

    policy_version = create_document_version_id(policy_id, content_sha256)
    runbook_version = create_document_version_id(runbook_id, content_sha256)

    assert policy_version != runbook_version


@pytest.mark.parametrize(
    "content_sha256",
    [
        "",
        "abc",
        "g" * 64,
        "A" * 64,
        "0" * 63,
        "0" * 65,
    ],
)
def test_document_version_id_rejects_malformed_sha256(
    content_sha256: str,
) -> None:
    document_id = create_document_id("local-manifest", "security/policy")

    with pytest.raises(ValueError):
        create_document_version_id(document_id, content_sha256)
