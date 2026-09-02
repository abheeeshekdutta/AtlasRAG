from uuid import UUID

import pytest
from pydantic import ValidationError

from atlasrag.ingestion.chunker import (
    ChunkingConfig,
    DocumentChunk,
    create_chunker_fingerprint,
)


def test_chunking_config_has_safe_defaults() -> None:
    config = ChunkingConfig()

    assert config.tokenizer_model == ("sentence-transformers/all-MiniLM-L6-v2")
    assert config.embedding_max_tokens == 256
    assert config.chunk_max_tokens == 200
    assert config.merge_peers is True


def test_chunking_config_rejects_blank_tokenizer_model() -> None:
    with pytest.raises(
        ValidationError,
        match="tokenizer_model must not be blank",
    ):
        ChunkingConfig(tokenizer_model="   ")


@pytest.mark.parametrize(
    "field_name",
    [
        "embedding_max_tokens",
        "chunk_max_tokens",
    ],
)
@pytest.mark.parametrize("invalid_value", [0, -1])
def test_chunking_config_rejects_non_positive_token_limits(
    field_name: str,
    invalid_value: int,
) -> None:
    data = {
        field_name: invalid_value,
    }

    with pytest.raises(ValidationError):
        ChunkingConfig.model_validate(data)


def test_chunking_config_rejects_chunk_limit_above_embedding_limit() -> None:
    with pytest.raises(
        ValidationError,
        match="chunk_max_tokens must not exceed embedding_max_tokens",
    ):
        ChunkingConfig(
            embedding_max_tokens=100,
            chunk_max_tokens=101,
        )


def test_chunking_config_rejects_extra_fields() -> None:
    with pytest.raises(
        ValidationError,
        match="Extra inputs are not permitted",
    ):
        ChunkingConfig.model_validate(
            {
                "unknown_setting": True,
            }
        )


def test_chunking_config_is_immutable() -> None:
    config = ChunkingConfig()

    with pytest.raises(
        ValidationError,
        match="Instance is frozen",
    ):
        config.chunk_max_tokens = 100


def test_chunker_fingerprint_is_deterministic_sha256() -> None:
    config = ChunkingConfig()

    first = create_chunker_fingerprint(config, docling_version="2.70.0")
    second = create_chunker_fingerprint(config, docling_version="2.70.0")

    assert first == second
    assert len(first) == 64
    assert first == first.lower()
    assert all(character in "0123456789abcdef" for character in first)


def test_chunker_fingerprint_changes_with_docling_version() -> None:
    config = ChunkingConfig()

    first = create_chunker_fingerprint(config, docling_version="2.70.0")
    second = create_chunker_fingerprint(config, docling_version="2.71.0")

    assert first != second


@pytest.mark.parametrize(
    "changed_config",
    [
        ChunkingConfig(tokenizer_model="sentence-transformers/another-model"),
        ChunkingConfig(embedding_max_tokens=512),
        ChunkingConfig(chunk_max_tokens=199),
        ChunkingConfig(merge_peers=False),
    ],
)
def test_chunker_fingerprint_changes_with_configuration(
    changed_config: ChunkingConfig,
) -> None:
    baseline = create_chunker_fingerprint(
        ChunkingConfig(),
        docling_version="2.70.0",
    )

    changed = create_chunker_fingerprint(
        changed_config,
        docling_version="2.70.0",
    )

    assert changed != baseline


@pytest.mark.parametrize("docling_version", ["", "   "])
def test_chunker_fingerprint_rejects_blank_docling_version(
    docling_version: str,
) -> None:
    with pytest.raises(ValueError, match="docling_version must not be blank"):
        create_chunker_fingerprint(
            ChunkingConfig(),
            docling_version=docling_version,
        )


def test_document_chunk_round_trips_through_json() -> None:
    original = DocumentChunk(
        chunk_id=UUID("6d684806-4336-5a68-9e75-0d537a0865f7"),
        document_id=UUID("27d3cb46-8666-5d77-b1b2-01308688364f"),
        document_version_id=UUID("a65fdb92-8893-5c37-abc3-45e6d681df72"),
        chunk_index=0,
        text="Employees must report incidents immediately.",
        contextualized_text=(
            "Security Incident Policy\nEmployees must report incidents immediately."
        ),
        token_count=11,
        headings=("Security Incident Policy",),
        source_item_refs=("#/texts/1",),
        page_numbers=(1,),
    )

    serialized = original.model_dump_json()
    restored = DocumentChunk.model_validate_json(serialized)

    assert restored == original
    assert restored.schema_version == 1
    assert restored.chunk_index == 0
    assert restored.headings == ("Security Incident Policy",)
    assert restored.page_numbers == (1,)


def _valid_chunk_data() -> dict[str, object]:
    return {
        "chunk_id": UUID("6d684806-4336-5a68-9e75-0d537a0865f7"),
        "document_id": UUID("27d3cb46-8666-5d77-b1b2-01308688364f"),
        "document_version_id": UUID("a65fdb92-8893-5c37-abc3-45e6d681df72"),
        "chunk_index": 0,
        "text": "Employees must report incidents immediately.",
        "contextualized_text": (
            "Security Incident Policy\nEmployees must report incidents immediately."
        ),
        "token_count": 11,
        "headings": ("Security Incident Policy",),
        "source_item_refs": ("#/texts/1",),
        "page_numbers": (1,),
    }


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("text", ""),
        ("text", "   "),
        ("contextualized_text", ""),
        ("contextualized_text", "   "),
        ("chunk_index", -1),
        ("token_count", 0),
        ("token_count", -1),
        ("page_numbers", (0,)),
        ("page_numbers", (-1,)),
        ("schema_version", 2),
    ],
)
def test_document_chunk_rejects_invalid_values(
    field_name: str,
    invalid_value: object,
) -> None:
    data = _valid_chunk_data()
    data[field_name] = invalid_value

    with pytest.raises(ValidationError):
        DocumentChunk.model_validate(data)


def test_document_chunk_is_immutable() -> None:
    chunk = DocumentChunk.model_validate(_valid_chunk_data())

    with pytest.raises(
        ValidationError,
        match="Instance is frozen",
    ):
        chunk.text = "Changed text"
