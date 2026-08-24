from importlib.metadata import version as distribution_version
from io import BytesIO
from typing import Literal

from docling.datamodel.base_models import (
    ConversionStatus,
    DocumentStream,
    InputFormat,
)
from docling.document_converter import DocumentConverter
from docling.exceptions import ConversionError
from docling_core.types.doc.document import DoclingDocument
from pydantic import BaseModel, ConfigDict

from atlasrag.ingestion.loader import LoadedDocument
from atlasrag.ingestion.models import DocumentVersion


class DocumentParsingError(RuntimeError):
    """Raised when a document cannot be converted completely."""


class ParsedDocument(BaseModel):
    """A parsed document coupled to its source-version lineage."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1
    version: DocumentVersion
    parser_name: str
    parser_version: str
    document: DoclingDocument


class DoclingParser:
    """Convert loaded documents into Docling's structured document model."""

    def __init__(self) -> None:
        self._converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.DOCX,
                InputFormat.MD,
            ]
        )
        self._parser_version = distribution_version("docling")

    def parse(self, loaded: LoadedDocument) -> ParsedDocument:
        stream = DocumentStream(
            name=loaded.path.name,
            stream=BytesIO(loaded.content),
        )

        try:
            result = self._converter.convert(
                stream,
                raises_on_error=True,
            )
        except ConversionError as error:
            raise DocumentParsingError(
                f"Docling conversion failed for {loaded.path.name}"
            ) from error

        if result.status != ConversionStatus.SUCCESS:
            details = "; ".join(str(error) for error in result.errors)
            raise DocumentParsingError(
                f"Docling conversion ended with status "
                f"{result.status.value}: {details or 'no details'}"
            )

        return ParsedDocument(
            version=loaded.version,
            parser_name="docling",
            parser_version=self._parser_version,
            document=result.document,
        )
