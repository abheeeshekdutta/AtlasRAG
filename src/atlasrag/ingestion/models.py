from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, computed_field, model_validator

from atlasrag.ingestion.identity import create_document_id


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
