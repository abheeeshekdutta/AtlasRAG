import json
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from atlasrag.ingestion.identity import calculate_content_sha256

CHUNKER_NAME = "docling-hybrid"
CHUNKER_ALGORITHM_VERSION = 1


class ChunkingConfig(BaseModel):
    """Configuration that controls repeatable document chunking."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    tokenizer_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_max_tokens: int = Field(default=256, gt=0)
    chunk_max_tokens: int = Field(default=200, gt=0)
    merge_peers: bool = True

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        if not self.tokenizer_model.strip():
            raise ValueError("tokenizer_model must not be blank")

        if self.chunk_max_tokens > self.embedding_max_tokens:
            raise ValueError("chunk_max_tokens must not exceed embedding_max_tokens")

        return self


def create_chunker_fingerprint(
    config: ChunkingConfig,
    *,
    docling_version: str,
) -> str:
    """Hash every setting that can change chunking output."""
    if not docling_version.strip():
        raise ValueError("docling_version must not be blank")

    payload = {
        "algorithm_version": CHUNKER_ALGORITHM_VERSION,
        "chunker_name": CHUNKER_NAME,
        "config": config.model_dump(mode="json"),
        "docling_version": docling_version,
    }
    canonical_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )

    return calculate_content_sha256(canonical_json.encode("utf-8"))


PositivePageNumber = Annotated[int, Field(gt=0)]


class DocumentChunk(BaseModel):
    """One searchable piece of a parsed document."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1
    chunk_id: UUID
    document_id: UUID
    document_version_id: UUID
    chunk_index: int = Field(ge=0)
    text: str
    contextualized_text: str
    token_count: int = Field(gt=0)
    headings: tuple[str, ...] = ()
    source_item_refs: tuple[str, ...] = ()
    page_numbers: tuple[PositivePageNumber, ...] = ()

    @field_validator("text", "contextualized_text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text values must not be blank")

        return value
