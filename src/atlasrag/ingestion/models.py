from enum import StrEnum
from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from atlasrag.ingestion.identity import (
    calculate_content_sha256,
    create_document_id,
    create_document_version_id,
)


class DocumentFormat(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"


class DocumentSource(BaseModel):
    """Stable identity supplied by a document source."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    source_namespace: str
    source_key: str

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        create_document_id(self.source_namespace, self.source_key)
        return self

    @computed_field
    @property
    def document_id(self) -> UUID:
        return create_document_id(self.source_namespace, self.source_key)


class DocumentVersion(BaseModel):
    """Immutable metadata for one exact source-document version."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    source: DocumentSource
    source_format: DocumentFormat
    content_sha256: str
    size_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_version_identity(self) -> Self:
        create_document_version_id(
            self.source.document_id,
            self.content_sha256,
        )
        return self

    @computed_field
    @property
    def document_version_id(self) -> UUID:
        return create_document_version_id(
            self.source.document_id,
            self.content_sha256,
        )

    @classmethod
    def from_content(
        cls,
        *,
        source: DocumentSource,
        source_format: DocumentFormat,
        content: bytes,
    ) -> Self:
        return cls(
            source=source,
            source_format=source_format,
            content_sha256=calculate_content_sha256(content),
            size_bytes=len(content),
        )
