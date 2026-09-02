from importlib.metadata import version as distribution_version
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from atlasrag.ingestion.chunker import (
    ChunkedDocument,
    ChunkingConfig,
    DoclingHybridChunker,
    DocumentChunk,
    DocumentChunkingError,
    create_chunker_fingerprint,
)
from atlasrag.ingestion.identity import create_chunk_id
from atlasrag.ingestion.loader import load_local_document
from atlasrag.ingestion.models import (
    DocumentFormat,
    DocumentSource,
    DocumentVersion,
)
from atlasrag.ingestion.parser import DoclingParser, ParsedDocument


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


def _valid_chunked_document() -> ChunkedDocument:
    version = DocumentVersion.from_content(
        source=DocumentSource(
            source_namespace="test-fixtures",
            source_key="security/policy",
        ),
        source_format=DocumentFormat.MARKDOWN,
        content=b"security policy",
    )
    config = ChunkingConfig()
    docling_version = "2.70.0"
    fingerprint = create_chunker_fingerprint(
        config,
        docling_version=docling_version,
    )
    contextualized_text = (
        "Security Incident Policy\nEmployees must report incidents immediately."
    )
    chunk = DocumentChunk(
        chunk_id=create_chunk_id(
            version.document_version_id,
            fingerprint,
            0,
            contextualized_text,
        ),
        document_id=version.source.document_id,
        document_version_id=version.document_version_id,
        chunk_index=0,
        text="Employees must report incidents immediately.",
        contextualized_text=contextualized_text,
        token_count=11,
        headings=("Security Incident Policy",),
        source_item_refs=("#/texts/1",),
    )

    return ChunkedDocument(
        version=version,
        config=config,
        docling_version=docling_version,
        chunker_fingerprint=fingerprint,
        chunks=(chunk,),
    )


def test_chunked_document_round_trips_through_json() -> None:
    original = _valid_chunked_document()

    restored = ChunkedDocument.model_validate_json(
        original.model_dump_json(exclude_computed_fields=True)
    )

    assert restored == original
    assert restored.schema_version == 1
    assert restored.chunks[0].chunk_index == 0


@pytest.mark.parametrize("docling_version", ["", "   "])
def test_chunked_document_rejects_blank_docling_version(
    docling_version: str,
) -> None:
    data = _valid_chunked_document().model_dump(exclude_computed_fields=True)
    data["docling_version"] = docling_version

    with pytest.raises(ValidationError, match="docling_version must not be blank"):
        ChunkedDocument.model_validate(data)


def test_chunked_document_rejects_empty_chunks() -> None:
    data = _valid_chunked_document().model_dump(exclude_computed_fields=True)
    data["chunks"] = ()

    with pytest.raises(ValidationError, match="chunks must not be empty"):
        ChunkedDocument.model_validate(data)


@pytest.mark.parametrize("chunk_index", [1, 2])
def test_chunked_document_rejects_noncontiguous_indexes(
    chunk_index: int,
) -> None:
    data = _valid_chunked_document().model_dump(exclude_computed_fields=True)
    data["chunks"][0]["chunk_index"] = chunk_index

    with pytest.raises(
        ValidationError,
        match="chunk indexes must be contiguous and ordered",
    ):
        ChunkedDocument.model_validate(data)


def test_chunked_document_rejects_wrong_fingerprint() -> None:
    data = _valid_chunked_document().model_dump(exclude_computed_fields=True)
    data["chunker_fingerprint"] = "0" * 64

    with pytest.raises(
        ValidationError,
        match="chunker_fingerprint does not match the chunking configuration",
    ):
        ChunkedDocument.model_validate(data)


def test_chunked_document_rejects_wrong_document_id() -> None:
    data = _valid_chunked_document().model_dump(exclude_computed_fields=True)
    data["chunks"][0]["document_id"] = UUID("16624b43-95c7-55d5-a277-ecec22198a2c")

    with pytest.raises(
        ValidationError,
        match="chunk document_id does not match the document",
    ):
        ChunkedDocument.model_validate(data)


def test_chunked_document_rejects_wrong_document_version_id() -> None:
    data = _valid_chunked_document().model_dump(exclude_computed_fields=True)
    data["chunks"][0]["document_version_id"] = UUID(
        "cf0da800-84df-542f-bce8-4004d9dc42fd"
    )

    with pytest.raises(
        ValidationError,
        match="chunk document_version_id does not match the document version",
    ):
        ChunkedDocument.model_validate(data)


def test_chunked_document_rejects_embedding_token_overflow() -> None:
    data = _valid_chunked_document().model_dump(exclude_computed_fields=True)
    data["chunks"][0]["token_count"] = 257

    with pytest.raises(
        ValidationError,
        match="chunk exceeds the embedding token limit",
    ):
        ChunkedDocument.model_validate(data)


def test_chunked_document_rejects_wrong_chunk_id() -> None:
    data = _valid_chunked_document().model_dump(exclude_computed_fields=True)
    data["chunks"][0]["chunk_id"] = UUID("c2f34067-4713-5ab8-8b91-b48cf594b96d")

    with pytest.raises(
        ValidationError,
        match="chunk_id does not match the chunk content",
    ):
        ChunkedDocument.model_validate(data)


@pytest.fixture(scope="module")
def parsed_policy() -> ParsedDocument:
    loaded = load_local_document(
        path=Path("tests/fixtures/documents/security_policy.md"),
        source=DocumentSource(
            source_namespace="test-fixtures",
            source_key="security/policy",
        ),
    )
    return DoclingParser().parse(loaded)


@pytest.fixture(scope="module")
def chunked_policy(parsed_policy: ParsedDocument) -> ChunkedDocument:
    return DoclingHybridChunker().chunk(parsed_policy)


def test_docling_hybrid_chunker_preserves_structure_and_lineage(
    parsed_policy: ParsedDocument,
    chunked_policy: ChunkedDocument,
) -> None:
    assert chunked_policy.version == parsed_policy.version
    assert chunked_policy.docling_version == distribution_version("docling-core")
    assert len(chunked_policy.chunks) == 4
    assert all(
        chunk.document_id == parsed_policy.version.source.document_id
        for chunk in chunked_policy.chunks
    )
    assert all(
        chunk.document_version_id == parsed_policy.version.document_version_id
        for chunk in chunked_policy.chunks
    )

    reporting_chunk = chunked_policy.chunks[1]
    assert reporting_chunk.headings == (
        "Security Incident Policy",
        "Reporting procedure",
    )
    assert reporting_chunk.source_item_refs == (
        "#/texts/3",
        "#/texts/4",
        "#/texts/5",
    )
    assert reporting_chunk.contextualized_text.startswith(
        "Security Incident Policy\nReporting procedure"
    )

    table_chunk = chunked_policy.chunks[2]
    assert table_chunk.source_item_refs == ("#/tables/0",)
    assert "Active compromise" in table_chunk.text
    assert all(not chunk.page_numbers for chunk in chunked_policy.chunks)


def test_docling_hybrid_chunker_is_deterministic(
    parsed_policy: ParsedDocument,
    chunked_policy: ChunkedDocument,
) -> None:
    repeated = DoclingHybridChunker().chunk(parsed_policy)

    assert repeated.chunker_fingerprint == chunked_policy.chunker_fingerprint
    assert [chunk.chunk_id for chunk in repeated.chunks] == [
        chunk.chunk_id for chunk in chunked_policy.chunks
    ]
    assert [chunk.contextualized_text for chunk in repeated.chunks] == [
        chunk.contextualized_text for chunk in chunked_policy.chunks
    ]


def test_docling_hybrid_chunker_rejects_final_token_overflow(
    parsed_policy: ParsedDocument,
) -> None:
    chunker = DoclingHybridChunker(
        ChunkingConfig(
            embedding_max_tokens=25,
            chunk_max_tokens=25,
        )
    )

    with pytest.raises(
        DocumentChunkingError,
        match=r"Chunk \d+ has \d+ tokens; the embedding limit is 25",
    ):
        chunker.chunk(parsed_policy)


def test_docling_hybrid_chunker_splits_long_section(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "long-policy.md"
    body = " ".join(
        f"Security control number {index} must be reviewed annually."
        for index in range(50)
    )
    document_path.write_text(
        f"# Long Security Policy\n\n{body}\n",
        encoding="utf-8",
    )
    loaded = load_local_document(
        path=document_path,
        source=DocumentSource(
            source_namespace="test-fixtures",
            source_key="security/long-policy",
        ),
    )
    parsed = DoclingParser().parse(loaded)
    config = ChunkingConfig(
        embedding_max_tokens=100,
        chunk_max_tokens=40,
    )

    result = DoclingHybridChunker(config).chunk(parsed)

    assert len(result.chunks) > 1
    assert all(chunk.token_count <= config.chunk_max_tokens for chunk in result.chunks)
    assert all(chunk.headings == ("Long Security Policy",) for chunk in result.chunks)
    assert all(chunk.source_item_refs == ("#/texts/1",) for chunk in result.chunks)


def test_docling_hybrid_chunker_rejects_document_without_content(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "heading-only.md"
    document_path.write_text("# Heading only\n", encoding="utf-8")
    loaded = load_local_document(
        path=document_path,
        source=DocumentSource(
            source_namespace="test-fixtures",
            source_key="security/heading-only",
        ),
    )
    parsed = DoclingParser().parse(loaded)

    with pytest.raises(
        DocumentChunkingError,
        match="Docling produced no chunks",
    ):
        DoclingHybridChunker().chunk(parsed)
